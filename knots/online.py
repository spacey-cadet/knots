"""
Online metrics: what listeners actually do with the picks, and whether that should start a retrain.

Everything here works on an append-only EVENT LOG (one JSON object per line) and on plain lists of events, so
it is testable without a server. The server is not wired to it yet: that needs a verified identity (derive it
from the OAuth token and pass a salted hash as `user`; never a caller-supplied id) and a place to host the log.

Events (no conversation text is ever logged, only catalog facts and ids):

  impression   a list was shown: session_id, status ("ok" | "relaxed" | "empty"), model, picks[pick_record()]
  feedback     the listener said a pick was good or not (explicit): session_id, url, good "y"|"n", user, source "user"
  engage       implicit positive: asked for more on a pick, opened or saved it (session_id, url, action)
  reformulate  implicit negative: re-asked the same thing in the same session after being shown picks

Two rules keep this honest:

  1. Only explicit `feedback` from a listener (source "user") is ever turned into a training label
     (`labels_from_events`). Engagement and reformulation can RAISE AN ALERT but never train anything.
  2. Alerts decide WHEN to try a retrain, never WHETHER the result ships: the champion/challenger gate in
     `retrain.gate` still has the last word, and a retrain still needs enough human labels from enough people.

Feedback only exists for picks that were shown, so it is biased toward what the model already favours. Every
pick record keeps its `position` and an `exploration` flag so that bias can be measured once an exploration
slot exists (`exploration` in the window metrics). Percentile metrics (novelty, unexpectedness) are only
meaningful against a FROZEN reference pool saved at promotion time: against the window's own picks they
average 0.5 by construction, so without a reference pool they are reported as unknown (None).
"""
from __future__ import annotations

import json
import math
import os
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import beyond as B

EVENT_TYPES = ("impression", "feedback", "engage", "reformulate")
STATUSES = ("ok", "relaxed", "empty")
DAY = 86400.0


@dataclass
class MonitorPolicy:
    window_days: float = float(os.environ.get("KNOTS_MONITOR_WINDOW_DAYS", "14"))
    min_feedback: int = int(os.environ.get("KNOTS_MONITOR_MIN_FEEDBACK", "40"))      # explicit answers before a quality alert
    quality_drop: float = 0.10          # the optimistic end of the current positive rate must be this far under baseline
    min_picks: int = 100                # shown picks on each side before comparing score distributions
    psi_alert: float = 0.25             # population stability index; 0.1-0.25 is a watch, above is a shift
    min_lists: int = 30                 # shown lists before a retrieval or filter-bubble alert
    retrieval_rise: float = 0.15        # rise in empty+relaxed share
    bubble_drop: float = 0.15           # fall in novelty / unexpectedness, or rise in repeat rate


# ----------------------------------------------------------------------------------------------- records
def pick_record(card: dict, position: int, exploration: bool = False) -> dict:
    """The compact, text-free facts about one shown pick that the metrics need."""
    ff = card.get("fit_features") or {}
    return {"url": card.get("url", ""), "type": card.get("type", ""), "matched_topic": card.get("matched_topic", ""),
            "position": int(position), "exploration": bool(exploration), "fit_score": card.get("fit_score"),
            "votes": card.get("votes"), "citations": card.get("citations"),
            "evidence_types": list(card.get("evidence_types") or []),
            "fit_features": {"overlap": ff.get("overlap"), "sim_session": ff.get("sim_session")}}


def impression_event(user: str, session_id: str, cards: list[dict], status: str = "ok", model: str = "",
                     exploration_urls: set[str] | frozenset = frozenset(), ts: float | None = None) -> dict:
    """`cards` in the order they were presented (position 1 first)."""
    return {"type": "impression", "ts": time.time() if ts is None else ts, "user": user, "session_id": session_id,
            "status": status, "model": model,
            "picks": [pick_record(c, i + 1, c.get("url") in exploration_urls) for i, c in enumerate(cards)]}


def validate(event: dict) -> dict:
    t = event.get("type")
    if t not in EVENT_TYPES:
        raise ValueError(f"unknown event type {t!r}")
    if not event.get("session_id"):
        raise ValueError("an event needs a session_id")
    if t == "feedback":
        if event.get("good") not in ("y", "n"):
            raise ValueError("feedback.good must be 'y' or 'n'")
        if not event.get("user") or not event.get("url"):
            raise ValueError("feedback needs a verified user and a url (labels are capped per person)")
    if t == "impression":
        if event.get("status", "ok") not in STATUSES:
            raise ValueError(f"impression.status must be one of {STATUSES}")
        if not isinstance(event.get("picks"), list):
            raise ValueError("an impression needs its picks")
    if t == "engage" and not event.get("url"):
        raise ValueError("engage needs a url")
    return event


class EventLog:
    """Append-only JSON-lines file. Small appends are single O_APPEND writes, so concurrent writers do not
    interleave a line. A damaged line is skipped on read (and counted), never fatal."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.skipped = 0

    def append(self, event: dict) -> dict:
        event = dict(validate(event))
        event.setdefault("ts", time.time())
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(event, separators=(",", ":")) + "\n").encode())
        finally:
            os.close(fd)
        return event

    def read(self, since: float | None = None) -> list[dict]:
        self.skipped, out = 0, []
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return []
        for line in lines:
            try:
                e = json.loads(line)
            except ValueError:
                self.skipped += 1
                continue
            if isinstance(e, dict) and (since is None or e.get("ts", 0) >= since):
                out.append(e)
        return out

    def forget(self, user: str) -> int:
        """Remove every event of one person (the log must honour forget_me). Returns how many were removed."""
        kept, removed = [], 0
        for line in (self.path.read_text().splitlines() if self.path.exists() else []):
            try:
                mine = json.loads(line).get("user") == user
            except ValueError:
                mine = False
            removed += mine
            if not mine:
                kept.append(line)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text("".join(k + "\n" for k in kept))
        os.replace(tmp, self.path)
        return removed


def window(events: list[dict], now: float | None = None, days: float = 14.0) -> list[dict]:
    now = time.time() if now is None else now
    return [e for e in events if e.get("ts", 0) >= now - days * DAY]


def labels_from_events(events: list[dict]) -> list[dict]:
    """The ONLY route from online events to training labels: explicit listener feedback. Implicit signals
    (engage, reformulate) never become labels."""
    return [{"user": e["user"], "session_id": e["session_id"], "url": e["url"], "good": e["good"],
             "source": "user", "ts": e.get("ts", 0)}
            for e in events if e.get("type") == "feedback" and e.get("source", "user") == "user"
            and e.get("good") in ("y", "n") and e.get("user") and e.get("url")]


# ----------------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float] | None:
    """90% Wilson interval for a proportion; None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def psi(reference: list[float], current: list[float], bins: int = 10) -> float | None:
    """Population stability index of `current` against `reference` (bins are the reference's quantiles).
    None when the reference has no spread to bin."""
    ref = sorted(x for x in reference if isinstance(x, (int, float)) and x == x)
    cur = [x for x in current if isinstance(x, (int, float)) and x == x]
    if len(set(ref)) < 2 or not cur:
        return None
    edges = sorted({ref[min(len(ref) - 1, int(len(ref) * i / bins))] for i in range(1, bins)})

    def shares(xs: list[float]) -> list[float]:
        counts = [0] * (len(edges) + 1)
        for x in xs:
            counts[bisect_right(edges, x)] += 1
        return [max(c / len(xs), 1e-4) for c in counts]
    r, c = shares(ref), shares(cur)
    return round(sum((ci - ri) * math.log(ci / ri) for ri, ci in zip(r, c)), 4)


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _latest_feedback(events: list[dict]) -> list[dict]:
    """Latest answer per (person, session, pick), listener feedback only."""
    latest: dict[tuple, dict] = {}
    for e in sorted((e for e in events if e.get("type") == "feedback" and e.get("source", "user") == "user"
                     and e.get("good") in ("y", "n")), key=lambda e: e.get("ts", 0)):
        latest[(e.get("user"), e.get("session_id"), e.get("url"))] = e
    return list(latest.values())


# ----------------------------------------------------------------------------------------------- metrics
def window_metrics(events: list[dict], pool: B.Pool | None = None) -> dict:
    """Online metrics for the events given (filter them with `window` first). Missing values are None.

    `pool` is the frozen reference pool for the percentile metrics; without one, novelty and unexpectedness
    are None (against the window's own picks they would average 0.5 whatever was shown)."""
    imps = sorted((e for e in events if e.get("type") == "impression"), key=lambda e: e.get("ts", 0))
    sessions = {e["session_id"] for e in imps}
    picks_by_key = {(e["session_id"], p.get("url")): p for e in imps for p in e.get("picks", [])}
    n_picks = sum(len(e.get("picks", [])) for e in imps)
    fb = _latest_feedback(events)
    pos = sum(1 for e in fb if e["good"] == "y")

    def rate(rows: list[dict]) -> tuple[float | None, int]:
        return _rate(sum(1 for e in rows if e["good"] == "y"), len(rows)), len(rows)

    def pick_of(e: dict) -> dict:
        return picks_by_key.get((e["session_id"], e["url"])) or {}

    top3 = [e for e in fb if pick_of(e).get("position", e.get("position", 99)) <= 3]
    is_explore = {id(e): bool(pick_of(e).get("exploration", e.get("exploration"))) for e in fb}
    explore = [e for e in fb if is_explore[id(e)]]
    ranked = [e for e in fb if not is_explore[id(e)]]
    ci = wilson(pos, len(fb))
    engaged = {e["session_id"] for e in events if e.get("type") == "engage"} & sessions
    reformed = {e["session_id"] for e in events if e.get("type") == "reformulate"} & sessions
    seen: dict[str, set[str]] = defaultdict(set)
    repeats = repeatable = 0
    for e in imps:
        u = e.get("user")
        if not u:
            continue
        for p in e.get("picks", []):
            if seen[u]:
                repeatable += 1
                repeats += p.get("url") in seen[u]
        seen[u].update(p.get("url") for p in e.get("picks", []))
    lists = [e.get("picks", []) for e in imps if e.get("picks")]
    beyond = {"format_spread": None, "topic_spread": None, "novelty": None, "unexpectedness": None,
              "coverage": B.coverage(lists)}
    if lists:
        per = [B.list_metrics(lst, None, pool or B.Pool([]), None) for lst in lists]
        for name in ("format_spread", "topic_spread") + (("novelty", "unexpectedness") if pool else ()):
            vals = [m[name] for m in per if m[name] == m[name]]
            beyond[name] = round(sum(vals) / len(vals), 4) if vals else None
    (p_rank, n_rank), (p_exp, n_exp), (p_top3, n_top3) = rate(ranked), rate(explore), rate(top3)
    return {
        "lists": len(imps), "picks": n_picks, "sessions": len(sessions), "users": len({e.get("user") for e in imps if e.get("user")}),
        "empty_rate": _rate(sum(1 for e in imps if e.get("status") == "empty"), len(imps)),
        "relaxed_rate": _rate(sum(1 for e in imps if e.get("status") == "relaxed"), len(imps)),
        "feedback": {"n": len(fb), "people": len({e["user"] for e in fb}), "per_pick_rate": _rate(len(fb), n_picks),
                     "positive_rate": _rate(pos, len(fb)),
                     "ci90": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
                     "positive_rate_top3": p_top3, "n_top3": n_top3},
        "exploration": {"positive_rate_ranked": p_rank, "n_ranked": n_rank, "positive_rate_explore": p_exp, "n_explore": n_exp},
        "engagement_rate": _rate(len(engaged), len(sessions)),
        "reformulation_rate": _rate(len(reformed), len(sessions)),
        "repeat_rate": _rate(repeats, repeatable),
        "beyond": beyond,
    }


def make_baseline(events: list[dict]) -> dict:
    """Freeze what 'normal' looked like when a model was promoted: pass this to `Registry.promote(baseline=...)`.
    Holds the positive rate, the score distribution, the beyond-accuracy levels and the reference pool."""
    imps = [e for e in events if e.get("type") == "impression"]
    picks = [p for e in imps for p in e.get("picks", [])]
    pool = B.Pool(picks)
    m = window_metrics(events, pool)
    scores = [p["fit_score"] for p in picks if isinstance(p.get("fit_score"), (int, float))]
    step = max(1, len(scores) // 500)
    return {"created": time.time(), "metrics": m, "scores": [round(s, 4) for s in scores[::step]], "pool": pool.to_dict()}


# ----------------------------------------------------------------------------------------------- triggers
def alerts(events: list[dict], baseline: dict | None, policy: MonitorPolicy | None = None) -> list[dict]:
    """What looks wrong in `events` (already windowed) compared with the baseline frozen at promotion.

    `retrain_relevant` says whether a retrain could plausibly help: a quality drop or a shift in the score
    distribution can be a stale ranker; an empty/relaxed spike is a retrieval or catalog problem, and a
    narrowing of what is shown is a feedback-loop warning, so neither of those should by itself start a retrain."""
    policy = policy or MonitorPolicy()
    if not baseline:
        return []
    base = baseline.get("metrics") or {}
    pool = B.Pool.from_dict(baseline["pool"]) if baseline.get("pool") else None
    m = window_metrics(events, pool)
    out = []
    fb, bfb = m["feedback"], (base.get("feedback") or {})
    if fb["n"] >= policy.min_feedback and bfb.get("positive_rate") is not None and fb["ci90"]:
        if fb["ci90"][1] < bfb["positive_rate"] - policy.quality_drop:
            out.append({"kind": "quality_drop", "retrain_relevant": True,
                        "detail": f"listeners said good {fb['positive_rate']:.0%} of the time "
                                  f"(90% range {fb['ci90'][0]:.0%}-{fb['ci90'][1]:.0%}, {fb['n']} answers) against "
                                  f"{bfb['positive_rate']:.0%} when this model was promoted"})
    imps = [e for e in events if e.get("type") == "impression"]
    cur_scores = [p["fit_score"] for e in imps for p in e.get("picks", []) if isinstance(p.get("fit_score"), (int, float))]
    ref_scores = baseline.get("scores") or []
    if len(cur_scores) >= policy.min_picks and len(ref_scores) >= policy.min_picks:
        v = psi(ref_scores, cur_scores)
        if v is not None and v >= policy.psi_alert:
            out.append({"kind": "score_drift", "retrain_relevant": True,
                        "detail": f"the fit scores being served have shifted (PSI {v:.2f}, alert at {policy.psi_alert})"})
    if m["lists"] >= policy.min_lists:
        bad = (m["empty_rate"] or 0) + (m["relaxed_rate"] or 0)
        bbad = (base.get("empty_rate") or 0) + (base.get("relaxed_rate") or 0)
        if bad - bbad >= policy.retrieval_rise:
            out.append({"kind": "retrieval", "retrain_relevant": False,
                        "detail": f"{bad:.0%} of lists were empty or relaxed (was {bbad:.0%}): check the catalogs, not the ranker"})
        bb, cb = base.get("beyond") or {}, m["beyond"]
        narrowed = [n for n in ("novelty", "unexpectedness")
                    if bb.get(n) is not None and cb[n] is not None and bb[n] - cb[n] >= policy.bubble_drop]
        rep, brep = m["repeat_rate"], base.get("repeat_rate")
        if rep is not None and rep - (brep or 0) >= policy.bubble_drop:
            narrowed.append("repeat_rate")
        if narrowed:
            out.append({"kind": "filter_bubble", "retrain_relevant": False,
                        "detail": "what is shown has narrowed (" + ", ".join(narrowed) + "): consider the exploration slot"})
    return out


def retrain_decision(labels: list[dict], events: list[dict], state: dict, policy=None, monitor: MonitorPolicy | None = None,
                     now: float | None = None) -> dict:
    """One call for 'should we try a retrain now?': label volume plus online alerts.

    Returns {"retrain", "reason", "alerts", "metrics"}. An alert that a retrain could fix lowers the number of
    NEW labels needed (Policy.alert_min_new) but never the total labels or the number of distinct people."""
    from .retrain import Policy, should_retrain
    policy, monitor = policy or Policy(), monitor or MonitorPolicy()
    recent = window(events, now, monitor.window_days)
    found = alerts(recent, state.get("baseline"), monitor)
    fixable = [a for a in found if a["retrain_relevant"]]
    go, why = should_retrain(labels, state, policy, alerts=fixable)
    return {"retrain": go, "reason": why, "alerts": found,
            "metrics": window_metrics(recent, B.Pool.from_dict(state["baseline"]["pool"])
                                      if (state.get("baseline") or {}).get("pool") else None)}

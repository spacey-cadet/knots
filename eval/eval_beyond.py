"""Beyond accuracy: how diverse, how well GROUNDED, and how far do the picks reach?

    python eval/eval_beyond.py                      # all extractors, train sessions; writes metrics_beyond.json
    python eval/eval_beyond.py --extractor bedrock  # one extractor only
    python eval/eval_beyond.py --no-embed           # skip the embedding call (no intra-list diversity)

Uses the same lists, orderings and session-resampled 90% intervals as `eval_rank.py`, at the headline cut (top 3
by default), so the numbers sit next to P@3 and can be read as a trade-off. Definitions are in `knots/beyond.py`
and docs/METRICS.md. Two blocks per extractor:

  labelled   the lists `eval_rank.py` scores (every pick labelled). HEADLINE: P@3, diversity, Grounding@3 (how much
             independent evidence backs each good pick), theme coverage, and the overreach rate, which is a GATE
             (a change that raises it is rejected, whatever it does to diversity). DIAGNOSTICS, never to be
             maximised: novelty, unexpectedness, serendipity, format spread, interest coverage.
  all shown  every train-session list, labelled or not: the label-free metrics only, so missing labels cannot
             shrink the sample

`supported` is read from a pick-level `supported` column in labels.csv when you add one. Until then it is a PROXY:
a pick is unsupported when it is labelled not good with reason "relevance" (the connection itself was wrong). Under
that proxy every good pick is supported, so the factor changes nothing for Grounding and only the overreach rate
uses it. Theme quotes and theme coverage use the stored per-theme quotes when the results have them, and otherwise
the human-supported themes in themes.csv, linked to picks by topic; the report says how many it could link.

Popularity and "how obvious" are percentiles against every pick shown in the run (the pool), so a number means
"more or less than a typical shown pick". Coverage has no catalog denominator: it is how spread out and how
repeated the shown items are. Holdout sessions are reported separately and never used to choose anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))
import eval_rank as ER  # noqa: E402
import metrics as M  # noqa: E402
from knots import beyond as B  # noqa: E402
from knots import grounding as G  # noqa: E402

OUT = ROOT / "eval" / "results" / "metrics_beyond.json"
K = 3
HEADLINE = ("p@3", "ild", "grounding", "theme_coverage", "overreach")
DIAGNOSTICS = ("novelty", "unexpectedness", "serendipity", "format_spread", "interest_coverage", "evidence_mean")
COLUMNS = HEADLINE + DIAGNOSTICS
LABEL_FREE = ("ild", "format_spread", "novelty", "unexpectedness", "interest_coverage", "evidence_mean")
SIM_SENSITIVITY = (0.25, 0.30, 0.35)
COVERAGE_COLUMNS = ("unique_share", "gini", "top_item_share")
RANDOM = "random (chance)"
ALL_SHOWN = "as read aloud (all picks)"

DEFINITIONS = {
    "ild": "mean pairwise cosine distance between the picks' vectors (higher = more varied)",
    "format_spread": "entropy of the picks' formats over log(list length); 1 = every pick a different format",
    "novelty": "1 - popularity percentile within the source (votes, views or citations); 1 = rarely seen",
    "unexpectedness": "1 - percentile of closeness to the person's own words (keyword overlap, session similarity)",
    "serendipity": "mean of good * unexpectedness over the picks",
    "grounding": "mean over the top-k picks of good * supported * min(E, 3) / 3, E = independent evidence types "
                 "(TMDB keyword, quoted theme, overview similarity; co-mention is not available yet). Distance earns nothing",
    "theme_coverage": "share of the session's linkable supported themes served by at least one good, supported pick",
    "overreach": "GATE: share of shown picks labelled not supported (proxy: bad with reason 'relevance'); lower is better",
    "evidence_mean": "mean number of evidence types per pick (0 to 3)",
    "interest_coverage": "share of the session's own interests that at least one shown pick answers",
    "unique_share": "distinct items / slots shown across all lists (1 = no item is ever repeated); point value, no interval",
    "gini": "inequality of how often each shown item appears (0 = all equally often)",
    "top_item_share": "the most repeated item's share of all slots",
}


# -------------------------------------------------------------------------------------------- lists
def shown_lists(results: list[dict], splits: dict, extractor: str | None) -> list[tuple[dict, list[dict]]]:
    """Every list from a train session that has picks, labelled or not."""
    out = []
    for e in results:
        if not e.get("ok") or (extractor and e["extractor"] != extractor):
            continue
        if splits.get(e["session_id"], "train") == "holdout":
            continue
        cards = ER.displayed_order(e)
        if cards:
            out.append((e, cards))
    return out


def ordered(cards: list[dict], how: str) -> list[dict]:
    key = ER.ORDERINGS[how]
    return cards if key is None else sorted(cards, key=key, reverse=True)


def embed_cards(cards: list[dict]) -> tuple[dict[str, list[float]], str]:
    """One vector per distinct pick, all from one embedder (the name comes back so the report can say which)."""
    from knots.embeddings import embed_many
    distinct: dict[str, dict] = {}
    for c in cards:
        distinct.setdefault(c["url"], c)
    urls = list(distinct)
    vecs, name = embed_many([B.card_text(distinct[u]) for u in urls])
    return dict(zip(urls, vecs)), name


def _norm(s) -> str:
    return " ".join(str(s or "").lower().split())


class Context:
    """What Grounding needs besides the picks: why bad picks were bad, optional pick-level `supported` labels, the
    human-supported themes, and the similarity threshold."""

    def __init__(self, reasons=None, supported=None, themes=None, sim_min: float | None = None):
        self.reasons = reasons or {}        # (session, extractor, url) -> reason
        self.supported = supported or {}    # (session, extractor, url) -> "y"/"n", from an optional pick-level column
        self.themes = themes or {}          # (session, extractor) -> [{"topic", "theme"}] human-supported
        self.sim_min = sim_min

    def with_sim(self, sim_min: float) -> "Context":
        return Context(self.reasons, self.supported, self.themes, sim_min)

    def is_supported(self, key: tuple, good: int) -> int:
        if key in self.supported:
            return 1 if self.supported[key] == "y" else 0
        return 0 if (not good and self.reasons.get(key, "") == "relevance") else 1

    def quoted(self, e: dict, card: dict) -> bool | None:
        """Does a quoted, kept theme stand behind this pick? None = the card stores the answer itself."""
        if card.get("evidence_types") is not None:
            return None
        topic = _norm(card.get("matched_topic"))
        for i in e.get("interests") or []:
            if _norm(i.get("topic")) == topic and any((i.get("theme_evidence") or {}).get(t)
                                                      for t in i.get("implicit_themes") or []):
                return True
        return any(_norm(t["topic"]) == topic for t in self.themes.get((e["session_id"], e["extractor"]), []))

    def types(self, e: dict, card: dict) -> list[str]:
        return G.card_types(card, self.quoted(e, card), self.sim_min)

    def list_themes(self, e: dict) -> list[tuple[str, str]]:
        """(interest topic, theme) pairs for this list: the entry's own quoted themes when it has them, otherwise the
        human-supported themes from themes.csv whose topic is one of the entry's interests."""
        own = [(_norm(i.get("topic")), t) for i in e.get("interests") or []
               for t in i.get("implicit_themes") or [] if (i.get("theme_evidence") or {}).get(t)]
        if own:
            return own
        topics = {_norm(i.get("topic")) for i in e.get("interests") or []}
        return [(_norm(t["topic"]), t["theme"]) for t in self.themes.get((e["session_id"], e["extractor"]), [])
                if _norm(t["topic"]) in topics]


def _row(e: dict, cards: list[dict], lab: dict | None, pool: B.Pool, vec: dict | None, ctx: Context | None = None) -> dict:
    sid, ext = e["session_id"], e["extractor"]
    rels = None if lab is None else [1 if lab[(sid, ext, c["url"])] == "y" else 0 for c in cards]
    m = B.list_metrics(cards, rels, pool, [vec[c["url"]] for c in cards] if vec else None)
    m["interest_coverage"] = _nan(B.interest_coverage(
        cards, [i["topic"] for i in e.get("interests") or [] if i.get("in_current", True)]))
    if rels is not None:
        m["p@3"] = sum(rels) / len(rels)
    if ctx is not None and cards:
        types = [ctx.types(e, c) for c in cards]
        m["evidence_mean"] = sum(len(t) for t in types) / len(cards)
        if rels is not None:
            sup = [ctx.is_supported((sid, ext, c["url"]), r) for c, r in zip(cards, rels)]
            m["grounding"] = sum(G.pick_grounding(r, u, len(t)) for r, u, t in zip(rels, sup, types)) / len(cards)
            m["overreach"] = sum(1 - u for u in sup) / len(cards)
            themes = ctx.list_themes(e)
            served = sum(1 for topic, _ in themes if any(
                _norm(c.get("matched_topic")) == topic and r and u for c, r, u in zip(cards, rels, sup)))
            m["theme_coverage"] = served / len(themes) if themes else float("nan")
    return {"session": sid, "metrics": m, "cards": cards}


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def rows_for(lists, lab, pool, vec, how: str, k: int | None, ctx: Context | None = None) -> list[dict]:
    return [_row(e, ordered(cards, how)[:k] if k else ordered(cards, how), lab, pool, vec, ctx) for e, cards in lists]


def random_rows(lists, lab, pool, vec, k: int, draws: int = 100, seed: int = 0, ctx: Context | None = None):
    """The same lists in random order, each metric averaged over many shuffles (the chance baseline).
    Returns (rows, the picks of each shuffle as whole-population draws, for coverage)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out, population = [], [[] for _ in range(draws)]
    for e, cards in lists:
        acc: dict[str, list[float]] = defaultdict(list)
        for d in range(draws):
            order = list(cards)
            rng.shuffle(order)
            population[d].append(order[:k])
            for name, v in _row(e, order[:k], lab, pool, vec, ctx)["metrics"].items():
                acc[name].append(v)
        out.append({"session": e["session_id"], "cards": cards[:k],
                    "metrics": {n: M._nanmean(v) for n, v in acc.items()}})
    return out, population


# ---------------------------------------------------------------------------------------- summaries
def coverage_summary(rows: list[dict]) -> dict:
    """Coverage of the shown lists. Point values only: resampling sessions with replacement repeats sessions, which
    manufactures repeated items and biases exactly what coverage measures, so no interval is given. Repetition also
    grows with the number of slots, so compare rows from the SAME lists, not across extractors or scopes."""
    return B.coverage([r["cards"] for r in rows])


def random_coverage(draws: list[list[list[dict]]]) -> dict:
    """The chance baseline's coverage: each statistic averaged over whole-population random orderings."""
    covs = [B.coverage(d) for d in draws]
    return {k: round(sum(c[k] for c in covs) / len(covs), 4) for k in covs[0]} if covs else B.coverage([])


def evaluate(lists, lab: dict | None, pool: B.Pool, vec: dict | None, k: int = K, ctx: Context | None = None) -> dict | None:
    if not lists:
        return None
    out = {}
    for how in ER.ORDERINGS:
        rows = rows_for(lists, lab, pool, vec, how, k, ctx)
        out[how] = _summ(rows, lab is None)
    out[ALL_SHOWN] = _summ(rows_for(lists, lab, pool, vec, "as read aloud", None, ctx), lab is None)
    rr, population = random_rows(lists, lab, pool, vec, k, ctx=ctx)
    out[RANDOM] = _summ(rr, lab is None)
    out[RANDOM]["coverage"] = random_coverage(population)
    return out


def _summ(rows: list[dict], label_free: bool) -> dict:
    wanted = LABEL_FREE if label_free else COLUMNS
    names = tuple(n for n in wanted if rows and n in rows[0]["metrics"])      # grounding needs a Context
    per = [{"session": r["session"], "metrics": {n: r["metrics"][n] for n in names}} for r in rows]
    s = M.summarize_rankings(per)
    res = {"lists": s["lists"], "sessions": s["sessions"], **{n: s[n] for n in names}}
    if "theme_coverage" in names:       # how many lists had a linkable theme at all: the honest denominator
        res["theme_coverage_lists"] = sum(1 for r in rows if r["metrics"]["theme_coverage"] == r["metrics"]["theme_coverage"])
    res["coverage"] = coverage_summary(rows)
    return res


def _fmt(v: dict | None) -> str:
    if not v or v["mean"] != v["mean"]:
        return "n/a"
    return f"{v['mean']:.2f} [{v['ci90'][0]:.2f}, {v['ci90'][1]:.2f}]"


def table(res: dict, names: tuple[str, ...]) -> list[str]:
    names = tuple(n for n in names if any(n in v for v in res.values()))
    rows = ["| Order | " + " | ".join(names) + " |", "| --- " * (len(names) + 1) + "|"]
    for order, v in res.items():
        rows.append(f"| {order} | " + " | ".join(_fmt(v.get(n)) for n in names) + " |")
    return rows


def evidence_report(lists, lab: dict, ctx: Context, k: int = K) -> dict:
    """What the evidence could and could not see, and how much the similarity threshold matters (the picks are the
    top k as read aloud; no intervals: this is a check on the definition, not a result)."""
    picks = [(e, c) for e, cards in lists for c in cards[:k]]
    counts = {t: 0 for t in G.TYPES}
    for e, c in picks:
        for t in ctx.types(e, c):
            counts[t] += 1
    linkable = sum(1 for e, c in picks if ctx.quoted(e, c) is not False)
    sens = {}
    for t in SIM_SENSITIVITY:
        rows = rows_for(lists, lab, B.Pool([]), None, "as read aloud", k, ctx.with_sim(t))
        vals = [r["metrics"]["grounding"] for r in rows]
        sens[f"{t:.2f}"] = round(sum(vals) / len(vals), 3) if vals else None
    return {"picks": len(picks), "fired": counts, "picks_with_a_quoted_theme": sum(1 for e, c in picks if ctx.quoted(e, c) is not False and ctx.quoted(e, c) is not None) + sum(1 for e, c in picks if ctx.quoted(e, c) is None and "theme_quote" in (c.get("evidence_types") or [])),
            "grounding_by_sim_threshold": sens, "default_sim_threshold": ctx.sim_min or G.sim_min()}


def coverage_table(res: dict) -> list[str]:
    rows = ["| Order | slots | distinct items | " + " | ".join(COVERAGE_COLUMNS) + " |",
            "| --- " * (3 + len(COVERAGE_COLUMNS)) + "|"]
    for order, v in res.items():
        c = v["coverage"]
        rows.append(f"| {order} | {c['slots']:g} | {c['distinct_items']:g} | "
                    + " | ".join(f"{c[n]:.2f}" for n in COVERAGE_COLUMNS) + " |")
    return rows


# ------------------------------------------------------------------------------------------- main
def load_context(labels: list[dict]) -> Context:
    """Reasons and optional pick-level `supported` labels from labels.csv, plus the human-supported themes."""
    reasons, supported = {}, {}
    for r in labels:
        key = (r["session_id"], r["extractor"], r["url"])
        reasons[key] = (r.get("reason") or "").strip().lower()
        if (r.get("supported") or "").strip().lower() in ("y", "n"):
            supported[key] = r["supported"].strip().lower()
    themes: dict = {}
    try:
        for r in csv.DictReader(open(ER.LABELS.parent / "themes.csv", encoding="utf-8-sig")):
            if (r.get("supported") or "").strip().lower() == "y":
                themes.setdefault((r["session_id"], r["extractor"]), []).append({"topic": r["topic"], "theme": r["theme"]})
    except OSError:
        pass
    return Context(reasons, supported, themes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default=None, help="only this extractor (e.g. bedrock); default: all and each")
    ap.add_argument("--no-embed", action="store_true", help="skip embedding the picks (no intra-list diversity)")
    args = ap.parse_args(argv)
    try:
        results = json.loads(ER.RESULTS.read_text())["results"]
    except (OSError, ValueError, KeyError):
        print("No eval/results/results.json: run `python eval/run_eval.py run` first.")
        return 1
    labels = list(csv.DictReader(open(ER.LABELS, encoding="utf-8-sig")))
    ctx = load_context(labels)
    splits = {s["id"]: s.get("split", "train") for s in json.loads(ER.SESSIONS.read_text())}
    all_cards = [c for e in results if e.get("ok") for c in ER.displayed_order(e)]
    pool = B.Pool(all_cards)
    vec, embedder = (None, None) if args.no_embed else embed_cards(all_cards)
    out: dict = {"embedder": M.embedder_info(), "ild_embedder": embedder, "k": K, "pool": pool.describe(),
                 "definitions": DEFINITIONS}
    if embedder == "hashed":
        print("Note: picks were embedded with the offline hashed embedder (lexical only), so ild is a word-overlap "
              "measure here, not a semantic one.")
    scopes = [args.extractor] if args.extractor else [None, *sorted({e["extractor"] for e in results if e.get("ok")})]
    for scope in scopes:
        built = ER.build_lists(results, labels, splits, scope)
        label = scope or "all extractors"
        shown = shown_lists(results, splits, scope)
        sc = {"labelled": evaluate(built["train"], built["lab"], pool, vec, ctx=ctx),
              "labelled_lists": len(built["train"]),
              "all_shown": evaluate(shown, None, pool, vec, ctx=ctx), "all_shown_lists": len(shown),
              "holdout": evaluate(built["holdout"], built["lab"], pool, vec, ctx=ctx)}
        if built["train"]:
            sc["evidence"] = evidence_report(built["train"], built["lab"], ctx)
        out[label] = sc
        print(f"\n== {label}: top {K}; {len(built['train'])} labelled lists, {len(shown)} lists in all ==")
        if sc["labelled"]:
            print("  HEADLINE (overreach is a gate: lower is better, and it must not rise):")
            print("\n".join("  " + r for r in table(sc["labelled"], HEADLINE)))
            print("  DIAGNOSTICS (describe what was shown; never maximise):")
            print("\n".join("  " + r for r in table(sc["labelled"], DIAGNOSTICS)))
            ev = sc.get("evidence")
            if ev:
                print(f"  evidence fired over {ev['picks']} picks: {ev['fired']}; Grounding@{K} at similarity "
                      f"threshold {ev['grounding_by_sim_threshold']} (default {ev['default_sim_threshold']})")
                print(f"  theme coverage was measurable on {sc['labelled']['as read aloud'].get('theme_coverage_lists', 0)}"
                      f" of {sc['labelled_lists']} lists; theme_quote and theme coverage are understated until "
                      "`run_eval.py run` is re-run so picks carry their theme quotes.")
        if sc["all_shown"]:
            print("  every train list (label-free metrics):")
            print("\n".join("  " + r for r in table(sc["all_shown"], LABEL_FREE)))
            print("  coverage of what was shown:")
            print("\n".join("  " + r for r in coverage_table(sc["all_shown"])))
        print(f"  holdout: " + ("measured, see metrics_beyond.json" if sc["holdout"] else "not measured yet"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""How good is the order of the recommendations people are actually given?

    python eval/eval_rank.py                      # all extractors, train sessions; writes metrics_ranking.json
    python eval/eval_rank.py --extractor bedrock  # one extractor only

Takes every (session, extractor) list of shown picks whose picks are ALL labelled (`good` = y/n), orders them
several ways, and reports precision@k, hit rate, MRR, MAP and NDCG@k with 90% intervals from resampling
sessions. The orderings:

  as read aloud     the order the voice line and the cards present them (lead medium first)
  fit score         the ranker's own score, best first (what the model thinks is best)
  tag overlap       the catalog's keyword-overlap score alone (the old signal)
  session fit       similarity to what was said, alone
  rating            TMDB rating alone
  popularity        votes / citations alone (checks whether popular means good)
  random            the same picks in random order, averaged over many shuffles (chance)

Only shown picks are labelled, so this measures the ORDER and the precision of the shown set. It cannot see a
better item the system never retrieved. Holdout sessions are reported separately and never used to choose
anything. A list with a still-unlabelled pick is left out and named, so missing labels cannot flatter a number.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import metrics as M  # noqa: E402

RESULTS = ROOT / "eval" / "results" / "results.json"
LABELS = ROOT / "eval" / "labels.csv"
SESSIONS = ROOT / "eval" / "sessions.json"
OUT = ROOT / "eval" / "results" / "metrics_ranking.json"

BUCKET_OF_MEDIUM = {"film": "watch", "video": "explore", "paper": "read"}
KS = (1, 3, 5)
HEADLINE = ("p@1", "p@3", "map", "ndcg@5")
LABELLED = ("y", "n")


def displayed_order(entry: dict) -> list[dict]:
    """Cards in the order they are read: the lead interest's best medium first, then the rest."""
    picks = entry["plan"]["picks"]
    lead = (entry.get("interests") or [{}])[0].get("mediums") or []
    buckets = [BUCKET_OF_MEDIUM[m] for m in lead if m in BUCKET_OF_MEDIUM]
    buckets += [b for b in ("watch", "read", "explore") if b not in buckets]
    return [c for b in buckets for c in picks.get(b, [])]


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) else float("-inf")


ORDERINGS = {
    "as read aloud": None,                                                    # displayed order itself
    "fit score": lambda c: _num(c.get("fit_score")),
    "tag overlap": lambda c: _num((c.get("fit_features") or {}).get("overlap")),
    "session fit": lambda c: _num((c.get("fit_features") or {}).get("sim_session")),
    "rating": lambda c: _num(c.get("rating")),
    "popularity": lambda c: _num(c.get("votes") or c.get("citations") or 0),
}


def build_lists(results: list[dict], labels: list[dict], splits: dict, extractor: str | None = None) -> dict:
    """Returns {"train": {ordering: [group]}, "holdout": {...}, "excluded": {...}}."""
    lab = {(x["session_id"], x["extractor"], x["url"]): x["good"] for x in labels}
    train: list[tuple[dict, list[dict]]] = []
    excluded = {"partly_unlabelled": [], "holdout_lists": 0, "holdout_labelled_lists": 0, "no_picks": []}
    holdout: list[tuple[dict, list[dict]]] = []
    for e in results:
        if not e.get("ok") or (extractor and e["extractor"] != extractor):
            continue
        cards = displayed_order(e)
        if not cards:
            excluded["no_picks"].append([e["session_id"], e["extractor"]])
            continue
        marks = [lab.get((e["session_id"], e["extractor"], c["url"]), "") for c in cards]
        if splits.get(e["session_id"], "train") == "holdout":
            excluded["holdout_lists"] += 1
            if all(m in LABELLED for m in marks):
                excluded["holdout_labelled_lists"] += 1
                holdout.append((e, cards))
            continue
        if not all(m in LABELLED for m in marks):
            if any(m in LABELLED for m in marks):
                excluded["partly_unlabelled"].append([e["session_id"], e["extractor"],
                                                      sum(m in LABELLED for m in marks), len(cards)])
            else:
                excluded["partly_unlabelled"].append([e["session_id"], e["extractor"], 0, len(cards)])
            continue
        train.append((e, cards))
    return {"train": train, "holdout": holdout, "excluded": excluded, "lab": lab}


def groups_for(lists, lab: dict, how: str) -> list[dict]:
    out = []
    key = ORDERINGS[how]
    for e, cards in lists:
        ordered = cards if key is None else sorted(cards, key=key, reverse=True)   # stable: ties keep read-aloud order
        out.append({"session": e["session_id"],
                    "rels": [1 if lab[(e["session_id"], e["extractor"], c["url"])] == "y" else 0 for c in ordered]})
    return out


def evaluate(lists, lab: dict) -> dict | None:
    if not lists:
        return None
    res = {how: M.summarize_rankings(groups_for(lists, lab, how), KS) for how in ORDERINGS}
    res["random (chance)"] = M.summarize_rankings(M.expected_random(groups_for(lists, lab, "as read aloud"), KS), KS)
    return res


def _fmt(v: dict) -> str:
    return f"{v['mean']:.2f} [{v['ci90'][0]:.2f}, {v['ci90'][1]:.2f}]"


def table(res: dict) -> list[str]:
    head = "| Order | " + " | ".join(h.upper() for h in HEADLINE) + " |"
    rows = [head, "| --- " * (len(HEADLINE) + 1) + "|"]
    for name, v in res.items():
        rows.append(f"| {name} | " + " | ".join(_fmt(v[h]) for h in HEADLINE) + " |")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default=None, help="only this extractor (e.g. bedrock); default: report each and both")
    args = ap.parse_args(argv)
    try:
        results = json.loads(RESULTS.read_text())["results"]
    except (OSError, ValueError, KeyError):
        print("No eval/results/results.json: run `python eval/run_eval.py run` first.")
        return 1
    labels = list(csv.DictReader(open(LABELS, encoding="utf-8-sig")))
    splits = {s["id"]: s.get("split", "train") for s in json.loads(SESSIONS.read_text())}
    out: dict = {"embedder": M.embedder_info(), "ks": list(KS), "headline": list(HEADLINE)}
    scopes = [args.extractor] if args.extractor else [None, *sorted({e["extractor"] for e in results if e.get("ok")})]
    for scope in scopes:
        built = build_lists(results, labels, splits, scope)
        label = scope or "all extractors"
        train = built["train"]
        res = evaluate(train, built["lab"])
        out[label] = {"train": res, "lists": len(train),
                      "sessions": len({e["session_id"] for e, _ in train}),
                      "excluded": built["excluded"],
                      "holdout": evaluate(built["holdout"], built["lab"])}
        print(f"\n== {label}: {len(train)} complete lists from {out[label]['sessions']} train sessions ==")
        if not res:
            print("  nothing labelled yet")
            continue
        b = res["as read aloud"]
        print(f"  precision of everything shown: {_fmt(b['set_precision'])}; "
              f"lists with no good pick at all: {b['lists_without_a_good_pick']} of {b['lists']}")
        print("\n".join("  " + r for r in table(res)))
        ex = built["excluded"]
        if ex["partly_unlabelled"]:
            print(f"  left out, picks still unlabelled: " + ", ".join(
                f"{s}/{x} ({k}/{n})" for s, x, k, n in ex["partly_unlabelled"]))
        print(f"  holdout: {ex['holdout_labelled_lists']} of {ex['holdout_lists']} lists labelled"
              + ("" if ex["holdout_labelled_lists"] else " (not measured yet)"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

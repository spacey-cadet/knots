"""Beyond accuracy: how diverse, novel, spread out, serendipitous and creative are the picks people are given?

    python eval/eval_beyond.py                      # all extractors, train sessions; writes metrics_beyond.json
    python eval/eval_beyond.py --extractor bedrock  # one extractor only
    python eval/eval_beyond.py --no-embed           # skip the embedding call (no intra-list diversity)

Uses the same lists, orderings and session-resampled 90% intervals as `eval_rank.py`, at the headline cut (top 3
by default), so the numbers sit next to P@3 and can be read as a trade-off. Definitions are in `knots/beyond.py`
and docs/METRICS.md. Two blocks per extractor:

  labelled   the lists `eval_rank.py` scores (every pick labelled): adds serendipity and creativity, which need
             a verdict on each pick, and P@3 for context
  all shown  every train-session list, labelled or not: the label-free metrics only (diversity, novelty,
             unexpectedness, coverage), so missing labels cannot shrink the sample

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

OUT = ROOT / "eval" / "results" / "metrics_beyond.json"
K = 3
COLUMNS = ("p@3", "ild", "format_spread", "novelty", "unexpectedness", "serendipity", "creativity", "interest_coverage")
LABEL_FREE = ("ild", "format_spread", "novelty", "unexpectedness", "interest_coverage")
COVERAGE_COLUMNS = ("unique_share", "gini", "top_item_share")
RANDOM = "random (chance)"
ALL_SHOWN = "as read aloud (all picks)"

DEFINITIONS = {
    "ild": "mean pairwise cosine distance between the picks' vectors (higher = more varied)",
    "format_spread": "entropy of the picks' formats over log(list length); 1 = every pick a different format",
    "novelty": "1 - popularity percentile within the source (votes, views or citations); 1 = rarely seen",
    "unexpectedness": "1 - percentile of closeness to the person's own words (keyword overlap, session similarity)",
    "serendipity": "mean of good * unexpectedness over the picks",
    "creativity": "mean of good * sqrt(novelty * unexpectedness) over the picks",
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


def _row(e: dict, cards: list[dict], lab: dict | None, pool: B.Pool, vec: dict | None) -> dict:
    sid, ext = e["session_id"], e["extractor"]
    rels = None if lab is None else [1 if lab[(sid, ext, c["url"])] == "y" else 0 for c in cards]
    m = B.list_metrics(cards, rels, pool, [vec[c["url"]] for c in cards] if vec else None)
    m["interest_coverage"] = _nan(B.interest_coverage(
        cards, [i["topic"] for i in e.get("interests") or [] if i.get("in_current", True)]))
    if rels is not None:
        m["p@3"] = sum(rels) / len(rels)
    return {"session": sid, "metrics": m, "cards": cards}


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def rows_for(lists, lab, pool, vec, how: str, k: int | None) -> list[dict]:
    return [_row(e, ordered(cards, how)[:k] if k else ordered(cards, how), lab, pool, vec) for e, cards in lists]


def random_rows(lists, lab, pool, vec, k: int, draws: int = 100, seed: int = 0):
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
            for name, v in _row(e, order[:k], lab, pool, vec)["metrics"].items():
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


def evaluate(lists, lab: dict | None, pool: B.Pool, vec: dict | None, k: int = K) -> dict | None:
    if not lists:
        return None
    out = {}
    for how in ER.ORDERINGS:
        rows = rows_for(lists, lab, pool, vec, how, k)
        out[how] = _summ(rows, lab is None)
    out[ALL_SHOWN] = _summ(rows_for(lists, lab, pool, vec, "as read aloud", None), lab is None)
    rr, population = random_rows(lists, lab, pool, vec, k)
    out[RANDOM] = _summ(rr, lab is None)
    out[RANDOM]["coverage"] = random_coverage(population)
    return out


def _summ(rows: list[dict], label_free: bool) -> dict:
    names = LABEL_FREE if label_free else COLUMNS
    per = [{"session": r["session"], "metrics": {n: r["metrics"][n] for n in names}} for r in rows]
    s = M.summarize_rankings(per)
    res = {"lists": s["lists"], "sessions": s["sessions"], **{n: s[n] for n in names}}
    res["coverage"] = coverage_summary(rows)
    return res


def _fmt(v: dict) -> str:
    return "n/a" if v["mean"] != v["mean"] else f"{v['mean']:.2f} [{v['ci90'][0]:.2f}, {v['ci90'][1]:.2f}]"


def table(res: dict, names: tuple[str, ...]) -> list[str]:
    rows = ["| Order | " + " | ".join(names) + " |", "| --- " * (len(names) + 1) + "|"]
    for order, v in res.items():
        rows.append(f"| {order} | " + " | ".join(_fmt(v[n]) for n in names) + " |")
    return rows


def coverage_table(res: dict) -> list[str]:
    rows = ["| Order | slots | distinct items | " + " | ".join(COVERAGE_COLUMNS) + " |",
            "| --- " * (3 + len(COVERAGE_COLUMNS)) + "|"]
    for order, v in res.items():
        c = v["coverage"]
        rows.append(f"| {order} | {c['slots']:g} | {c['distinct_items']:g} | "
                    + " | ".join(f"{c[n]:.2f}" for n in COVERAGE_COLUMNS) + " |")
    return rows


# ------------------------------------------------------------------------------------------- main
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
        sc = {"labelled": evaluate(built["train"], built["lab"], pool, vec),
              "labelled_lists": len(built["train"]),
              "all_shown": evaluate(shown, None, pool, vec), "all_shown_lists": len(shown),
              "holdout": evaluate(built["holdout"], built["lab"], pool, vec)}
        out[label] = sc
        print(f"\n== {label}: top {K}; {len(built['train'])} labelled lists, {len(shown)} lists in all ==")
        if sc["labelled"]:
            print("  labelled lists (with serendipity and creativity):")
            print("\n".join("  " + r for r in table(sc["labelled"], COLUMNS)))
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

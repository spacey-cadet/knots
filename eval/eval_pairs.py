"""How often is a watch-and-learn pair one you would actually watch and read together?

    python eval/eval_pairs.py        # reads eval/pairs.csv (label column `good`), writes metrics_pairs.json

Reports the success rate with a session-resampled 90% interval, for train and holdout sessions
separately, and whether a higher pair_score really means a better pair (AUC). Pairs are only
generated when both halves are sourced, so a low count of pairs is expected, not a bug.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import metrics as M  # noqa: E402

OUT = ROOT / "eval" / "results" / "metrics_pairs.json"


def summarize(rows: list[dict]) -> dict | None:
    rows = [r for r in rows if r["good"] in ("y", "n")]
    if not rows:
        return None
    y = np.array([r["good"] == "y" for r in rows], int)
    g = np.array([r["session_id"] for r in rows])
    s = np.array([float(r["pair_score"] or 0) for r in rows])
    rate = lambda sc, yy: float(np.mean(yy))  # noqa: E731
    lo, hi = M.boot_ci(s, y, g, stat=rate)
    out = {"pairs": len(y), "sessions": len(set(g.tolist())), "success_rate": round(float(y.mean()), 3),
           "success_ci90": [round(lo, 3), round(hi, 3)]}
    out["score_auc"] = round(M.auc(s, y), 3) if len(set(y.tolist())) > 1 else None
    return out


def main() -> int:
    path = ROOT / "eval" / "pairs.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig"))) if path.exists() else []
    out = {"all": summarize(rows),
           "train": summarize([r for r in rows if r.get("split", "train") != "holdout"]),
           "holdout": summarize([r for r in rows if r.get("split") == "holdout"]),
           "generated": len(rows), "unlabelled": sum(r["good"] not in ("y", "n") for r in rows),
           "embedder": M.embedder_info()}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    if not out["all"]:
        print(f"{len(rows)} pairs generated, none labelled yet. Run `run_eval.py run`, then label the Pairs tab.")
        return 1
    for k in ("train", "holdout"):
        v = out[k]
        print(f"{k:8s}", "no labelled pairs" if not v else
              f"{v['pairs']} pairs / {v['sessions']} sessions: success {v['success_rate']:.0%} "
              f"{v['success_ci90']}, pair_score AUC {v['score_auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

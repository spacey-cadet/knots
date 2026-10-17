"""Fit the ranker weights (knots/fitmodel.py) from your pick labels, and evaluate on unseen sessions.

    python eval/train_rank.py            # report + eval/results/metrics_rank.json
    python eval/train_rank.py --write    # also save knots/brain/data/rank_weights.json

Needs cards that carry `fit_features` (written by `run_eval.py run` from this version on). Labels
made before that cannot be used: the features were never stored, and reconstructing them would be
a guess. Re-run the eval, then relabel.

Rules: leave-one-session-out; holdout-split sessions are never trained on; the weights are written
only with MIN_SESSIONS or more labelled TRAIN sessions (override --force); calibration (Platt) is
fitted on the held-out scores. The prior and the old overlap score are reported next to the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from knots import fitmodel  # noqa: E402
import metrics as M  # noqa: E402

MIN_SESSIONS = 20
METRICS = ROOT / "eval" / "results" / "metrics_rank.json"
KINDS = ("watch", "read", "explore")


def load_rows(labels=None, results=None):
    """Labelled picks joined to their stored fit features. Returns (rows, n_labelled, n_without_features)."""
    labels = labels or list(csv.DictReader(open(ROOT / "eval/labels.csv", encoding="utf-8-sig")))
    results = results or json.loads((ROOT / "eval/results/results.json").read_text())["results"]
    cards = {}
    for r in results:
        if r.get("ok"):
            for kind in KINDS:
                for c in r["plan"]["picks"].get(kind, []):
                    cards[(r["session_id"], r["extractor"], c["url"])] = c
    rows, labelled, missing = [], 0, 0
    for x in labels:
        if x.get("good") not in ("y", "n") or x.get("live") != "yes":
            continue
        labelled += 1
        c = cards.get((x["session_id"], x["extractor"], x["url"]))
        if not c or not c.get("fit_features"):
            missing += 1
            continue
        rows.append({"session": x["session_id"], "split": x.get("split", "train"), "type": x["type"],
                     "y": int(x["good"] == "y"), "old": float(c.get("relevance") or 0.0),
                     "x": [float(c["fit_features"][f]) for f in fitmodel.FEATURES]})
    return rows, labelled, missing


def evaluate(rows, C=0.3):
    X = np.array([r["x"] for r in rows]); y = np.array([r["y"] for r in rows])
    g = np.array([r["session"] for r in rows]); old = np.array([r["old"] for r in rows])
    prior = X @ np.array([fitmodel.DEFAULT_WEIGHTS[f] for f in fitmodel.FEATURES])
    loso = np.zeros(len(y))
    for s in sorted(set(g)):
        te = g == s
        m = LogisticRegression(C=C, max_iter=1000).fit(X[~te], y[~te])
        loso[te] = X[te] @ m.coef_[0]
    out = {}
    for name, s in (("old overlap score", old), ("hand-set prior", prior), ("fitted (leave-session-out)", loso)):
        lo, hi = M.boot_ci(s, y, g)
        out[name] = {"auc": round(M.auc(s, y), 3), "auc_ci90": [round(lo, 3), round(hi, 3)],
                     "precision_at_3": round(M.precision_at_k(s, y, g, 3), 3)}
    a, b = M.fit_platt(loso, y)
    cal = 1 / (1 + np.exp(-(a * loso + b)))
    return X, y, g, loso, out, (a, b), M.reliability(cal, y), M.ece(cal, y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--C", type=float, default=0.3)
    args = ap.parse_args()
    rows, labelled, missing = load_rows()
    train = [r for r in rows if r["split"] != "holdout"]
    hold = [r for r in rows if r["split"] == "holdout"]
    METRICS.parent.mkdir(exist_ok=True)
    if len(train) < 10 or len({r["y"] for r in train}) < 2:
        msg = (f"{labelled} labelled live picks, but only {len(rows)} carry stored fit features "
               f"({missing} do not). Re-run `python eval/run_eval.py run`, then label the new picks.")
        print(msg)
        METRICS.write_text(json.dumps({"status": "not_trained", "reason": msg, "embedder": M.embedder_info()}, indent=1))
        return 1
    X, y, g, loso, res, (a, b), rel, ece = evaluate(train, args.C)
    sessions = len(set(g))
    print(f"{len(y)} train picks in {sessions} sessions, {y.mean():.0%} good; {len(hold)} holdout picks\n")
    print(f"{'ranker (held-out sessions)':32s} {'AUC':>5s}  {'90% interval':>14s}  {'P@3':>5s}")
    for k, v in res.items():
        print(f"{k:32s} {v['auc']:5.2f}  {str(v['auc_ci90']):>14s}  {v['precision_at_3']:5.2f}")
    print(f"{'random':32s} {0.5:5.2f}  {'':>14s}  {y.mean():5.2f}")
    print(f"\ncalibration a={a:.2f} b={b:.2f}, expected calibration error {ece:.3f}")
    full = LogisticRegression(C=args.C, max_iter=1000).fit(X, y)
    w = {f: round(float(c), 4) for f, c in zip(fitmodel.FEATURES, full.coef_[0])}
    print("weights fit on all train rows:", w)
    by_type = {t: round(float(np.mean([r["y"] for r in train if r["type"] == t])), 3)
               for t in sorted({r["type"] for r in train})}
    holdout = None
    if len(hold) >= 10 and len({r["y"] for r in hold}) > 1:
        hx = np.array([r["x"] for r in hold]); hy = np.array([r["y"] for r in hold]); hg = np.array([r["session"] for r in hold])
        hs = hx @ full.coef_[0]
        lo, hi = M.boot_ci(hs, hy, hg)
        holdout = {"n": len(hy), "auc": round(M.auc(hs, hy), 3), "auc_ci90": [round(lo, 3), round(hi, 3)],
                   "precision_at_3": round(M.precision_at_k(hs, hy, hg, 3), 3)}
        print("HOLDOUT:", holdout)
    METRICS.write_text(json.dumps({
        "status": "trained", "train_picks": len(y), "train_sessions": sessions, "good_rate": round(float(y.mean()), 3),
        "good_rate_by_type": by_type, "loso": res, "weights": w,
        "calibration": {"a": round(a, 4), "b": round(b, 4), "ece": round(ece, 4), "reliability": rel},
        "holdout": holdout, "embedder": M.embedder_info()}, indent=1))
    if args.write:
        if sessions < MIN_SESSIONS and not args.force:
            print(f"\nNot written: only {sessions} train sessions (< {MIN_SESSIONS}). Label more, or pass --force.")
            return 2
        out = fitmodel.WEIGHTS_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"weights": w, "calibration": {"a": a, "b": b},
                                   "meta": {"picks": len(y), "sessions": sessions, "C": args.C,
                                            "date": date.today().isoformat(), "loso_auc": res["fitted (leave-session-out)"]["auc"]}},
                                  indent=1))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

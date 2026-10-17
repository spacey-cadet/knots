"""Train and honestly evaluate the theme judge on the author's labels (eval/themes.csv).

    python eval/train_brain.py            # report only (also writes eval/results/metrics_brain.json)
    python eval/train_brain.py --save     # also fit on all TRAIN rows and write knots/brain/data/*

What it builds: the critic (logistic model over interpretable features), the nearest-labelled-theme
examples, and a Platt calibration + threshold for their average (see knots/brain/judge.py).

Honesty rules baked in:
  - Leave-one-session-out CV: the model that scores a session never saw it.
  - Rows from sessions marked `holdout` (sessions.json) are NEVER trained on or tuned on. If enough
    are labelled they are scored separately, and that number is the one to trust.
  - The critic's feature lists were written after reading the training labels, so its CV number is
    optimistic. Only the holdout is a clean test.
  - The threshold is a fixed rule (0.5 on the calibrated scale), not tuned on these labels.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from knots import calibration  # noqa: E402
from knots.brain.features import FEATURE_NAMES, features  # noqa: E402
from knots.brain.judge import logit  # noqa: E402
from knots.brain.neighbors import Neighbors  # noqa: E402
import metrics as M  # noqa: E402

DATA = ROOT / "knots" / "brain" / "data"
METRICS = ROOT / "eval" / "results" / "metrics_brain.json"
THRESHOLD = 0.5
MIN_HOLDOUT = 10


def load(split: str | None = "train"):
    """Labelled theme rows; `split` filters to train/holdout (None = all)."""
    rows = [r for r in csv.DictReader(open(ROOT / "eval/themes.csv", encoding="utf-8-sig"))
            if r["supported"] in ("y", "n") and (split is None or r.get("split", "train") == split)]
    text = lambda r: r["session_text"].replace(" / ", " ")  # noqa: E731
    X = np.array([features(text(r), r["theme"]) for r in rows]) if rows else np.zeros((0, len(FEATURE_NAMES)))
    y = np.array([1 if r["supported"] == "y" else 0 for r in rows])
    g = np.array([r["session_id"] for r in rows])
    return rows, X, y, g


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000).fit((X - mu) / sd, y)
    return m, mu, sd


def _nn(rows, y):
    return Neighbors([r["theme"] for r in rows], y.tolist(), [r["session_id"] for r in rows])


def _stats(name, raw, y, g, thr=None):
    lo, hi = M.boot_ci(raw, y, g)
    within = [M.auc(raw[g == s], y[g == s]) for s in sorted(set(g)) if len(set(y[g == s])) > 1]
    out = {"auc": round(M.auc(raw, y), 3), "auc_ci90": [round(lo, 3), round(hi, 3)],
           "within_session_auc": round(float(np.nanmean(within)), 3) if within else None}
    print(f"  {name:34s} AUC {out['auc']:.2f} [{lo:.2f}, {hi:.2f}]   within-session {out['within_session_auc']}")
    return out


def main() -> int:
    rows, X, y, g = load("train")
    hrows, hX, hy, hg = load("holdout")
    print(f"{len(y)} train labels in {len(set(g))} sessions, base rate supported {y.mean():.0%}; "
          f"{len(hy)} holdout labels")

    # leave-one-session-out for the critic, the neighbours and their average
    pc, pn = np.zeros(len(y)), np.zeros(len(y))
    nn_all = _nn(rows, y)
    for s in sorted(set(g)):
        te = g == s
        m, mu, sd = fit(X[~te], y[~te])
        pc[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
        pn[te] = [(nn_all.score(r["theme"], exclude_group=s) + 1) / 2 for r, t in zip(rows, te) if t]
    ens = (pc + pn) / 2
    print("\nleave-one-session-out (train sessions):")
    res = {"critic": _stats("critic (features; optimistic)", pc, y, g),
           "neighbors": _stats("nearest labelled themes", pn, y, g),
           "ensemble": _stats("average of the two", ens, y, g)}

    a, b = M.fit_platt([logit(v) for v in ens], y)
    cal = np.array([calibration.apply(logit(v), a, b) for v in ens])
    rel = M.reliability(cal, y)
    print(f"\ncalibration (Platt a={a:.2f}, b={b:.2f}); expected calibration error {M.ece(cal, y):.3f}")
    print("  predicted vs observed:", "  ".join(f"{r['bin']}: {r['predicted']:.2f}/{r['observed']:.2f} (n={r['n']})" for r in rel))
    print(f"\nkeeping themes with calibrated p >= {THRESHOLD}:")
    for t in (0.4, THRESHOLD, 0.6):
        keep = cal >= t
        prec = y[keep].mean() if keep.any() else float("nan")
        print(f"  p>={t}: keeps {keep.sum():3d}/{len(y)}  precision {prec:.0%}  recall {y[keep].sum() / max(1, y.sum()):.0%}")
    keep = cal >= THRESHOLD
    res["at_threshold"] = {"threshold": THRESHOLD, "kept": int(keep.sum()), "of": len(y),
                           "precision": round(float(y[keep].mean()), 3) if keep.any() else None,
                           "base_rate": round(float(y.mean()), 3)}

    # the clean test: holdout sessions, with models fitted on train rows only
    m, mu, sd = fit(X, y)
    holdout = None
    if len(hy) >= MIN_HOLDOUT and len(set(hy)) > 1:
        hp = np.array([(m.predict_proba((hX[i:i + 1] - mu) / sd)[0, 1] + (nn_all.score(r["theme"]) + 1) / 2) / 2
                       for i, r in enumerate(hrows)])
        print("\nHOLDOUT (never trained or tuned on; this is the number to trust):")
        holdout = _stats("ensemble", hp, hy, hg)
        hc = np.array([calibration.apply(logit(v), a, b) for v in hp])
        k = hc >= THRESHOLD
        holdout.update(n=int(len(hy)), sessions=int(len(set(hg))), base_rate=round(float(hy.mean()), 3),
                       precision_at_threshold=round(float(hy[k].mean()), 3) if k.any() else None,
                       kept=int(k.sum()), reliability=M.reliability(hc, hy))
        print(f"  keeps {int(k.sum())}/{len(hy)} at p>={THRESHOLD}, precision "
              f"{holdout['precision_at_threshold']} (base rate {holdout['base_rate']})")
    else:
        print(f"\nHOLDOUT: {len(hy)} labelled themes (need {MIN_HOLDOUT}+ with both classes). Label the "
              "holdout sessions (sessions.json split=holdout) to get a clean number.")

    out = {"train_labels": int(len(y)), "train_sessions": int(len(set(g))), "loso": res,
           "calibration": {"a": round(a, 4), "b": round(b, 4), "ece": round(M.ece(cal, y), 4), "reliability": rel},
           "holdout": holdout}
    METRICS.parent.mkdir(exist_ok=True)
    METRICS.write_text(json.dumps(out, indent=1))

    m_all, mu, sd = fit(X, y)
    print("\nweights:", {n: round(float(w), 2) for n, w in zip(FEATURE_NAMES, m_all.coef_[0])})
    if "--save" in sys.argv:
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / "theme_critic.json").write_text(json.dumps(
            {"features": FEATURE_NAMES, "mean": mu.tolist(), "std": sd.tolist(),
             "w": m_all.coef_[0].tolist(), "b": float(m_all.intercept_[0])}, indent=1))
        (DATA / "theme_examples.jsonl").write_text("".join(
            json.dumps({k: r.get(k, "") for k in ("session_id", "session_text", "theme", "supported", "error_type", "note")},
                       ensure_ascii=False) + "\n" for r in rows))
        (DATA / "theme_judge.json").write_text(json.dumps(
            {"calibration": {"a": a, "b": b}, "threshold": THRESHOLD,
             "meta": {"train_labels": int(len(y)), "train_sessions": int(len(set(g))),
                      "loso_ensemble_auc": res["ensemble"]["auc"]}}, indent=1))
        print("saved", DATA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

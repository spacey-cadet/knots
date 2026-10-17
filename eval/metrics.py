"""Shared evaluation helpers: AUC, precision@k, session-bootstrap intervals, reliability tables.

Rows from one session are highly correlated, so uncertainty is estimated by resampling WHOLE
sessions, and models are always scored on sessions they were not trained on."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(scores, y) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, scores)) if len(set(y.tolist())) > 1 else float("nan")


def boot_ci(scores, y, groups, n: int = 2000, seed: int = 0, stat=auc) -> tuple[float, float]:
    """90% interval of `stat` from resampling whole sessions."""
    scores, y, groups = np.asarray(scores), np.asarray(y), np.asarray(groups)
    rng, sessions, vals = np.random.default_rng(seed), sorted(set(groups.tolist())), []
    for _ in range(n):
        pick = rng.choice(sessions, len(sessions))
        idx = np.concatenate([np.where(groups == s)[0] for s in pick])
        v = stat(scores[idx], y[idx])
        if v == v:                       # skip NaN (a resample with one class)
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [5, 95])
    return float(lo), float(hi)


def precision_at_k(scores, y, groups, k: int = 3) -> float:
    scores, y, groups = np.asarray(scores), np.asarray(y), np.asarray(groups)
    out = []
    for s in sorted(set(groups.tolist())):
        i = np.where(groups == s)[0]
        out.append(y[i[np.argsort(-scores[i])[:k]]].mean())
    return float(np.mean(out)) if out else float("nan")


def reliability(p, y, bins: int = 5) -> list[dict]:
    """Per probability bin: how many, mean predicted, observed rate. Honest calibration check."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if m.any():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                         "predicted": round(float(p[m].mean()), 3), "observed": round(float(y[m].mean()), 3)})
    return rows


def ece(p, y, bins: int = 5) -> float:
    """Expected calibration error over `bins` equal-width bins (lower is better)."""
    rows = reliability(p, y, bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rows) / n) if n else float("nan")


def fit_platt(scores, y) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(scores, float).reshape(-1, 1), y)
    return float(m.coef_[0][0]), float(m.intercept_[0])

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


def embedder_info(results_path=None) -> dict:
    """Which embedder produced eval/results/results.json (and so every ranker and pair number built from it).

    {"recorded": False} for a results file written before this was tracked, so nothing is guessed."""
    import json
    from pathlib import Path
    path = Path(results_path) if results_path else Path(__file__).resolve().parent / "results" / "results.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"recorded": False}
    cfg = data.get("embedder")
    if not cfg:
        return {"recorded": False}
    runs: dict[str, int] = {}
    fell_back = 0
    for e in data.get("results", []):
        used = e.get("embedders_used") or {}
        if not e.get("ok") or not used:
            continue
        for name in used:
            runs[name] = runs.get(name, 0) + 1
        if cfg.get("provider") != "hashed" and used.get("hashed"):
            fell_back += 1
    names = sorted(runs)
    provider = names[0] if len(names) == 1 else ("mixed" if names else cfg.get("provider"))
    return {"recorded": True, "provider": provider, "requested": cfg.get("requested"),
            "model_id": cfg.get("model_id"), "dimensions": cfg.get("dimensions"),
            "runs_by_embedder": runs, "fell_back_runs": fell_back}


# ---------------------------------------------------------------------------------------------
# List-wise ranking metrics. A "group" is one ranked list of picks shown for one session, as 0/1 labels
# in the order they were ranked. Only shown picks are labelled, so these measure the ORDER of what was
# shown and the precision of the shown set, not whether better candidates were never retrieved.
# P@k uses the first min(k, len) items: a list shorter than k is judged on what it contains.
# AP and NDCG are undefined for a list with nothing good in it (no order can score there), so they are
# averaged over lists with at least one good pick, and that count is reported.
# ---------------------------------------------------------------------------------------------
import math as _math

NAN = float("nan")


def p_at_k(rels, k: int) -> float:
    top = list(rels)[:k]
    return sum(top) / len(top) if top else NAN


def hit_at_k(rels, k: int) -> float:
    return float(any(list(rels)[:k]))


def mrr(rels) -> float:
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def average_precision(rels) -> float:
    hits, total = 0, 0.0
    for i, r in enumerate(rels, 1):
        if r:
            hits += 1
            total += hits / i
    return total / hits if hits else NAN


def ndcg_at_k(rels, k: int) -> float:
    rels = [float(r) for r in rels]

    def dcg(xs):
        return sum(x / _math.log2(i + 2) for i, x in enumerate(xs[:k]))
    ideal = dcg(sorted(rels, reverse=True))
    return dcg(rels) / ideal if ideal else NAN


def group_metrics(rels, ks=(1, 3, 5)) -> dict[str, float]:
    out = {"set_precision": p_at_k(rels, len(rels) or 1), "mrr": mrr(rels), "map": average_precision(rels)}
    for k in ks:
        out[f"p@{k}"] = p_at_k(rels, k)
        out[f"hit@{k}"] = hit_at_k(rels, k)
        out[f"ndcg@{k}"] = ndcg_at_k(rels, k)
    return out


def _nanmean(v) -> float:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else NAN


def summarize_rankings(groups: list[dict], ks=(1, 3, 5), n_boot: int = 2000, seed: int = 0) -> dict:
    """groups: [{"session": id, "rels": [0/1 in ranked order]}], or [{"session": id, "metrics": {...}}] when the
    per-list metrics are already computed (the random baseline). Means with a 90% interval from resampling
    SESSIONS (several lists from one session are correlated, so the session is the unit that is resampled)."""
    per = [g["metrics"] if "metrics" in g else group_metrics(g["rels"], ks) for g in groups]
    names = list(per[0]) if per else []
    sessions = sorted({g["session"] for g in groups})
    by_session = {s: [i for i, g in enumerate(groups) if g["session"] == s] for s in sessions}
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([by_session[sessions[j]] for j in rng.integers(0, len(sessions), len(sessions))]).astype(int)
             for _ in range(n_boot)] if sessions else []
    out: dict = {"lists": len(groups), "sessions": len(sessions),
                 "lists_without_a_good_pick": sum(1 for p in per if p["map"] != p["map"])}
    for name in names:
        vals = np.array([p[name] for p in per], float)
        boots = [_nanmean(vals[d]) for d in draws]
        boots = [b for b in boots if b == b]
        lo, hi = (float(np.percentile(boots, 5)), float(np.percentile(boots, 95))) if boots else (NAN, NAN)
        out[name] = {"mean": round(_nanmean(vals), 3), "ci90": [round(lo, 3), round(hi, 3)]}
    return out


def expected_random(groups: list[dict], ks=(1, 3, 5), draws: int = 300, seed: int = 0) -> list[dict]:
    """The same lists in random order: each list's metrics averaged over many shuffles (the chance baseline)."""
    rng = np.random.default_rng(seed)
    out = []
    for g in groups:
        acc: dict[str, list[float]] = {}
        for _ in range(draws):
            order = list(g["rels"])
            rng.shuffle(order)
            for k, v in group_metrics(order, ks).items():
                acc.setdefault(k, []).append(v)
        out.append({"session": g["session"], "metrics": {k: _nanmean(v) for k, v in acc.items()}})
    return out

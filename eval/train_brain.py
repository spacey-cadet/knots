"""Train and honestly evaluate the theme critic on the author's labels (eval/themes.csv).

Leave-one-session-out CV: the model that scores a session's themes never saw that session.
    python eval/train_brain.py            # report only
    python eval/train_brain.py --save     # also fit on everything and write knots/brain/data/theme_critic.json
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from knots.brain.features import FEATURE_NAMES, features  # noqa: E402


def load():
    rows = [r for r in csv.DictReader(open(ROOT / "eval/themes.csv", encoding="utf-8-sig"))
            if r["supported"] in ("y", "n")]
    X = np.array([features(r["session_text"].replace(" / ", " "), r["theme"]) for r in rows])
    y = np.array([1 if r["supported"] == "y" else 0 for r in rows])
    g = np.array([r["session_id"] for r in rows])
    return rows, X, y, g


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000).fit((X - mu) / sd, y)
    return m, mu, sd


def main():
    rows, X, y, g = load()
    p = np.zeros(len(y))
    for s in set(g):
        te = g == s
        m, mu, sd = fit(X[~te], y[~te])
        p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    print(f"{len(y)} labels, base rate supported {y.mean():.0%}")
    print(f"leave-one-session-out AUC {roc_auc_score(y, p):.2f}")
    for t in (0.4, 0.5, 0.6):
        keep = p >= t
        print(f"keep p>={t}: keeps {keep.sum():3d}/{len(y)}  precision {y[keep].mean():.0%}  recall {(y[keep]).sum()/y.sum():.0%}")
    m, mu, sd = fit(X, y)
    print("weights:", {n: round(float(w), 2) for n, w in zip(FEATURE_NAMES, m.coef_[0])})
    if "--save" in sys.argv:
        out = ROOT / "knots/brain/data/theme_critic.json"
        out.write_text(json.dumps({"features": FEATURE_NAMES, "mean": mu.tolist(), "std": sd.tolist(),
                                   "w": m.coef_[0].tolist(), "b": float(m.intercept_[0])}, indent=1))
        print("saved", out)
        ex = ROOT / "knots/brain/data/theme_examples.jsonl"
        ex.write_text("".join(json.dumps({k: r[k] for k in ("session_id", "session_text", "theme", "supported", "note")},
                                         ensure_ascii=False) + "\n" for r in rows))
        print("saved", ex)


main()

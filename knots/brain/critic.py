"""Theme critic: a logistic model over interpretable features, trained on the author's labels
(eval/train_brain.py). Scores how likely the author would accept a theme as supported by the text."""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

from .features import features

DATA = Path(__file__).parent / "data"
THRESHOLD = float(os.environ.get("KNOTS_THEME_THRESHOLD", "0.4"))  # CV: precision 62% at 0.4 vs 44% base


@lru_cache(maxsize=1)
def _model() -> dict | None:
    try:
        return json.loads((DATA / "theme_critic.json").read_text())
    except (OSError, ValueError):
        return None


def score(session_text: str, theme: str) -> float | None:
    """P(author accepts this theme), or None if no trained critic is shipped."""
    m = _model()
    if not m:
        return None
    x = features(session_text, theme)
    z = m["b"] + sum(w * (v - mu) / sd for w, v, mu, sd in zip(m["w"], x, m["mean"], m["std"]))
    return 1 / (1 + math.exp(-z))


def split(session_text: str, themes: list[str]) -> tuple[list[str], list[str]]:
    """(kept, dropped). Off with KNOTS_BRAIN=off; never drops when there is no critic."""
    if os.environ.get("KNOTS_BRAIN", "on").lower() == "off" or not _model():
        return list(themes), []
    kept, dropped = [], []
    for t in themes:
        (kept if score(session_text, t) >= THRESHOLD else dropped).append(t)
    return kept, dropped

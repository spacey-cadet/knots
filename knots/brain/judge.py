"""
The theme judge: the critic (interpretable features) and the nearest-labelled-theme vote,
averaged, then calibrated.

  raw       mean of the available scores, each on 0..1 (critic probability; neighbour vote
            mapped from [-1, 1])
  p         Platt-calibrated probability, ONLY if a calibration was fitted on held-out scores
            and shipped (knots/brain/data/theme_judge.json). Without it `p` is None and no
            confidence is shown: an uncalibrated number would be a made-up number.
  decision  keep a theme when its score clears the threshold, else move it to dropped_themes.

Off with KNOTS_BRAIN=off. With no shipped data the judge never drops anything.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

from .. import calibration
from . import critic
from .neighbors import Neighbors

DATA = critic.DATA


def _off() -> bool:
    return os.environ.get("KNOTS_BRAIN", "on").lower() == "off"


@lru_cache(maxsize=1)
def _meta() -> dict:
    try:
        return json.loads((DATA / "theme_judge.json").read_text())
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def _neighbors() -> Neighbors | None:
    try:
        with open(DATA / "theme_examples.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except OSError:
        return None
    rows = [r for r in rows if r.get("supported") in ("y", "n")]
    if len(rows) < 20:           # too few examples to vote
        return None
    return Neighbors([r["theme"] for r in rows], [1 if r["supported"] == "y" else 0 for r in rows],
                     [r["session_id"] for r in rows])


def reload() -> None:
    """Drop cached artifacts (after retraining, and in tests)."""
    for fn in (_meta, _neighbors, critic._model):
        getattr(fn, "cache_clear", lambda: None)()


def raw_score(session_text: str, theme: str, exclude_group: str | None = None,
              use_critic: bool = True, use_neighbors: bool = True) -> float | None:
    parts = []
    if use_critic:
        pc = critic.score(session_text, theme)
        if pc is not None:
            parts.append(pc)
    nn = _neighbors() if use_neighbors else None
    if nn is not None:
        v = nn.score(theme, exclude_group=exclude_group)
        if v is not None:
            parts.append((v + 1.0) / 2.0)
    return sum(parts) / len(parts) if parts else None


def logit(x: float) -> float:
    x = min(1 - 1e-4, max(1e-4, x))
    return math.log(x / (1 - x))


def probability(session_text: str, theme: str) -> float | None:
    """Calibrated P(supported), or None when there is no judge or it is uncalibrated."""
    raw = raw_score(session_text, theme)
    cal = calibration.from_meta(_meta())
    if raw is None or cal is None:
        return None
    return calibration.apply(logit(raw), *cal)


def threshold() -> float:
    env = os.environ.get("KNOTS_THEME_THRESHOLD")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:
        return float(_meta()["threshold"])
    except (KeyError, TypeError, ValueError):
        return critic.THRESHOLD


def assess(session_text: str, themes: list[str]) -> tuple[list[str], list[str], dict[str, float]]:
    """(kept, dropped, confidence). `confidence` maps kept themes to a calibrated probability and
    is empty when the judge is uncalibrated. Never drops when off or when there is no judge."""
    if _off():
        return list(themes), [], {}
    cal = calibration.from_meta(_meta())
    thr = threshold()
    kept, dropped, conf = [], [], {}
    for t in themes:
        raw = raw_score(session_text, t)
        if raw is None:
            kept.append(t)
            continue
        value = calibration.apply(logit(raw), *cal) if cal else raw
        if value >= thr:
            kept.append(t)
            if cal:
                conf[t] = round(value, 2)
        else:
            dropped.append(t)
    return kept, dropped, conf

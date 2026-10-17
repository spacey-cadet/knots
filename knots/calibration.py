"""
Calibration: turning a model's raw score into "when it says 80%, is it right ~80% of the time?".

Platt scaling: p = sigmoid(a * score + b). The pair (a, b) is fitted on HELD-OUT scores
(leave-one-session-out) by the training scripts in eval/, and shipped next to the model. At
serving time only `apply` runs. A model with no shipped (a, b) is *uncalibrated*, and callers
must show no probability for it rather than pretend.
"""
from __future__ import annotations

import math


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


def apply(score: float, a: float, b: float) -> float:
    return sigmoid(a * score + b)


def from_meta(meta: dict | None) -> tuple[float, float] | None:
    """(a, b) from a model artifact's `calibration` block, or None if it has none."""
    c = (meta or {}).get("calibration")
    try:
        return float(c["a"]), float(c["b"])
    except (TypeError, KeyError, ValueError):
        return None

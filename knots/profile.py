"""
A user profile that learns from history without growing, and a read on what kind of session this is.

Bounded by construction: at most MAX_SLOTS interest slots (default 20), each one a topic label, ONE centroid
vector (an exponential moving average of everything folded into it), a decayed weight and a last-seen time. New
evidence merges into the nearest slot; a slot nobody has touched for a while decays and is pruned; when full,
the lightest slot is evicted. Raw text is never stored. Size is O(MAX_SLOTS * dim) forever.

The profile is a PRIOR, never the decider: `prior()` is capped, so what the person said in this session always
outweighs what they liked before. Vectors from different embedders are never compared (`compatible`).

learn_vs_play (0 = pure entertainment, 1 = pure learning) is read from the session's own wording first and
only nudged by a slow average of past sessions. No cues at all reads as a neutral 0.5, not as a guess.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

MAX_SLOTS = int(os.environ.get("KNOTS_PROFILE_SLOTS", "20"))
HALF_LIFE_DAYS = float(os.environ.get("KNOTS_PROFILE_HALF_LIFE_DAYS", "45"))
MERGE_SIM = 0.80            # cosine above which new evidence joins an existing slot
ALPHA = 0.20                # how far a slot's centroid moves toward new evidence
PRUNE_BELOW = 0.05
PRIOR_CAP = 0.15            # the profile can add at most this much to a score (scores are 0..1)
MODE_ALPHA = 0.10           # how fast the long-run learn/play average moves
SESSION_WEIGHT = 0.75       # the session's own wording versus the long-run average

_LEARN = ("explain", "how does", "how do", "why ", "why?", "research", "paper", "understand", "learn", "what is",
          "what's", "study", "theory", "history of", "mechanism", "tell me more", "deep dive", "concept")
_PLAY = ("something fun", "funny", "chill", "relax", "binge", "entertain", "bored", "movie night", "cozy",
         "light", "switch off", "unwind", "popcorn", "just want to watch", "mindless")


@dataclass
class Slot:
    topic: str
    centroid: list[float]
    weight: float = 1.0
    last_seen: float = 0.0
    n: int = 1


@dataclass
class Profile:
    embedder: str = ""
    slots: list[Slot] = field(default_factory=list)
    learn_play: float = 0.5                  # long-run average
    updated: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(embedder=d.get("embedder", ""), learn_play=float(d.get("learn_play", 0.5)),
                   updated=float(d.get("updated", 0.0)), slots=[Slot(**s) for s in d.get("slots", [])])

    def compatible(self, embedder: str) -> bool:
        return not self.slots or self.embedder == embedder


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def _decay(p: Profile, now: float) -> None:
    if p.updated and now > p.updated:
        f = 0.5 ** ((now - p.updated) / (HALF_LIFE_DAYS * 86400))
        for s in p.slots:
            s.weight *= f
    p.slots = [s for s in p.slots if s.weight >= PRUNE_BELOW]


def update(p: Profile, evidence: list[tuple[str, list[float], float]], embedder: str, now: float | None = None,
           session_mode: float | None = None) -> Profile:
    """Fold one session in. `evidence` = [(topic, vector, strength 0..1)]. Mutates and returns the profile."""
    now = time.time() if now is None else now
    if not p.compatible(embedder):
        p.slots, p.learn_play = [], 0.5            # different vector space: start clean rather than mix
    p.embedder = embedder
    _decay(p, now)
    for topic, vec, strength in evidence:
        if not vec:
            continue
        best = max(p.slots, key=lambda s: cosine(s.centroid, vec), default=None)
        if best is not None and cosine(best.centroid, vec) >= MERGE_SIM:
            best.centroid = [(1 - ALPHA) * c + ALPHA * v for c, v in zip(best.centroid, vec)]
            best.weight += strength
            best.last_seen, best.n = now, best.n + 1
            if strength >= 0.5:
                best.topic = topic
        else:
            p.slots.append(Slot(topic=topic, centroid=list(vec), weight=strength or 0.1, last_seen=now))
    p.slots.sort(key=lambda s: -s.weight)
    del p.slots[MAX_SLOTS:]
    if session_mode is not None:
        p.learn_play = (1 - MODE_ALPHA) * p.learn_play + MODE_ALPHA * session_mode
    p.updated = now
    return p


def prior(p: Profile, vec: list[float], embedder: str) -> float:
    """A bounded bonus (0..PRIOR_CAP) for how close `vec` is to what this person has come back to."""
    if not p.slots or not p.compatible(embedder) or not vec:
        return 0.0
    total = sum(s.weight for s in p.slots) or 1.0
    best = max(max(0.0, cosine(s.centroid, vec)) * min(1.0, s.weight / total * len(p.slots) / 2) for s in p.slots)
    return round(PRIOR_CAP * best, 4)


def session_mode(texts: list[str]) -> float | None:
    """0..1 from this session's wording; None when there is nothing to go on."""
    t = " ".join(texts).lower()
    learn = sum(t.count(c) for c in _LEARN)
    play = sum(t.count(c) for c in _PLAY)
    if learn + play == 0:
        return None
    return round(0.5 + 0.5 * (learn - play) / (learn + play + 1), 3)


def blend_mode(session: float | None, p: Profile | None) -> dict:
    avg = p.learn_play if p else 0.5
    v = avg if session is None else SESSION_WEIGHT * session + (1 - SESSION_WEIGHT) * avg
    v = round(v, 3)
    return {"learn_vs_play": v, "label": "learn" if v >= 0.65 else "play" if v <= 0.35 else "mixed",
            "from_session": session is not None}

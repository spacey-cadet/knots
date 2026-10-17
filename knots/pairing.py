"""
Watch-and-learn pairs: the unit that ties entertainment to learning.

A pair is one thing to WATCH (film, series, documentary) and one thing to LEARN FROM (paper or
video explainer) about the SAME interest, in an order that works as a double feature.

    pair_score = min(fit(W), fit(L)) * cos(W, L) * level_match(W, L)

  fit        percentile of the item's fit score within its group (watch or learn). `min` means a
             weak half cannot be rescued by a strong half.
  cos        embedding similarity of the two items' own text: they must cover the same idea, not
             just the same session.
  level      a ladder, not a match: the learning item should be as demanding as the watch item or
             somewhat more (drama -> explainer -> survey paper). Same level, a modest step up: 1.0;
             slightly down or a big jump: 0.7; otherwise 0.5.

Selection: same interest only; best score first; every item used once; at most one pair per
interest and MAX_PAIRS overall. A pair is shown only if BOTH cards are sourced (see receipts.py)
and the score clears a floor. The floors and the level rule are untuned design choices: the pair
labels in eval/pairs.csv (and eval/eval_pairs.py) are what will tell us whether they are right.
"""
from __future__ import annotations

from . import fitmodel, signals
from .embeddings import cosine
from .ranking import _concept, card_extras, medium_weight, render_card
from . import receipts

W_TYPES = ("movie", "series", "doc")
L_TYPES = ("paper", "video")
MAX_PAIRS = 3
PAIR_MIN_SIM = {"titan": 0.35, "cohere": 0.35, "hashed": 0.10}      # cos(W, L) floor by embedder
PAIR_MIN_SCORE = {"titan": 0.22, "cohere": 0.22, "hashed": 0.05}


def level_match(w_level: float, l_level: float) -> float:
    d = l_level - w_level
    if 0.0 <= d <= 0.5:
        return 1.0
    if -0.15 <= d < 0.0 or 0.5 < d <= 0.7:
        return 0.7
    return 0.5


def _percentiles(indices: list[int], scores: list[float]) -> dict[int, float]:
    order = sorted(indices, key=lambda i: scores[i])
    n = len(order)
    return {i: (1.0 if n == 1 else r / (n - 1)) for r, i in enumerate(order)}


def _shared_terms(w, l) -> list[str]:
    """Tags of the watch item that literally appear in the learn item's text. Facts, not guesses."""
    text = (l.title + " " + (l.abstract or l.blurb)).lower()
    return [t for t in w.matched_keywords if t.lower() in text][:3]


def _bridge(w, l, concept: str, shared: list[str]) -> str:
    role = "explainer" if l.content_type == "video" else "research"
    s = f"Watch {w.title}, then go to the {role}: {l.title}. Both are about {concept}."
    if shared:
        s += f" Both touch on: {', '.join(shared)}."
    return s


def build_pairs(items: list, interests: list, query: str = "", limit: int = MAX_PAIRS) -> list[dict]:
    by_topic = {i.topic: i for i in interests}
    W = [k for k, it in enumerate(items) if it.content_type in W_TYPES and it.topic in by_topic]
    L = [k for k, it in enumerate(items) if it.content_type in L_TYPES and it.topic in by_topic]
    if not W or not L:
        return []
    sc = fitmodel.score_items(query, items, interests, medium_weight)
    pw, pl = _percentiles(W, sc.scores), _percentiles(L, sc.scores)
    min_sim = PAIR_MIN_SIM.get(sc.embedder, 0.10)
    min_score = PAIR_MIN_SCORE.get(sc.embedder, 0.05)

    cands = []
    for w in W:
        for l in L:
            if items[w].topic != items[l].topic:
                continue
            cos = cosine(sc.vecs[w], sc.vecs[l])
            if cos < min_sim:
                continue
            lm = level_match(signals.level_of(items[w]), signals.level_of(items[l]))
            score = min(pw[w], pl[l]) * cos * lm
            if score >= min_score:
                cands.append((score, w, l, cos, lm))
    pairs, used, topics = [], set(), set()
    for score, w, l, cos, lm in sorted(cands, reverse=True):
        wi, li = items[w], items[l]
        if w in used or l in used or wi.topic in topics:
            continue
        interest = by_topic[wi.topic]
        wc = render_card(wi, interest, card_extras(sc, w, wi, interest, query))
        lc = render_card(li, interest, card_extras(sc, l, li, interest, query))
        if not (receipts.sourced(wc["receipts"]) and receipts.sourced(lc["receipts"])):
            continue                       # an unsourced half means no pair, not a weaker pair
        used.update((w, l))
        topics.add(wi.topic)
        concept = _concept(interest)
        pairs.append({"concept": concept, "score": round(score, 3),
                      "components": {"cos": round(cos, 3), "level_match": lm,
                                     "fit_watch": round(pw[w], 2), "fit_learn": round(pl[l], 2)},
                      "watch": wc, "learn": lc,
                      "bridge": _bridge(wi, li, concept, _shared_terms(wi, li))})
        if len(pairs) >= limit:
            break
    return pairs

"""
explain_recommendation + render_card: turns a matched ContentItem plus the
interest that produced it into a human-readable "why", then packages a set
of them into the structured card payload Alexa+ renders on a screened device
(and the `reasoning` line it can speak).

Reasons are built only from facts we hold: the concept and unstated theme the
extractor found, and the catalog tags that actually matched the item. We never
assert what a film "is about" beyond its tags, so a reason can't hallucinate.
"""

from __future__ import annotations

from . import fitmodel, receipts, signals
from .catalogs import ContentItem
from .extraction import Interest
from . import grounding
from .quality import BLOCKBUSTER_MIN_RELEVANCE, is_blockbuster, is_gem

# What the person does with each type. Used to group output for a screen/voice.
KIND = {"movie": "watch", "series": "watch", "doc": "watch", "video": "explore", "paper": "read"}


MEDIUM_OF = {"movie": "film", "series": "film", "doc": "film", "video": "video", "paper": "paper"}
MEDIUM_WEIGHT = (1.0, 0.85, 0.7)    # score multiplier by an interest's medium rank


def medium_weight(item: ContentItem, interest: Interest) -> float:
    medium = MEDIUM_OF.get(item.content_type, "film")
    order = [m for m in interest.mediums if m in MEDIUM_OF.values()] or ["film", "video", "paper"]
    rank = order.index(medium) if medium in order else len(MEDIUM_WEIGHT) - 1
    return MEDIUM_WEIGHT[min(rank, len(MEDIUM_WEIGHT) - 1)]


def _concept(interest: Interest) -> str:
    return interest.concepts[0] if interest.concepts else interest.topic


def explain_recommendation(item: ContentItem, interest: Interest) -> str:
    """Return one to three sentences of grounded reasoning, not a generic blurb."""
    topic, concept = interest.topic, _concept(interest)
    deeper = concept != topic            # the extractor found something beneath the surface topic
    theme = interest.implicit_themes[0] if interest.implicit_themes else ""

    if item.content_type == "doc":
        lead = (f"You've been asking about {topic}, and underneath it the idea of {concept}. "
                f"{item.title} is the real-world story." if deeper
                else f"Given your interest in {topic}, {item.title} covers the real-world story behind it.")
    elif item.content_type == "movie":
        lead = (f"Your questions about {topic} keep landing on {concept}; "
                f"{item.title} puts that idea on screen as fiction." if deeper
                else f"You've been engaging with {topic}; {item.title} explores similar themes on screen.")
    elif item.content_type == "series":
        lead = (f"Your questions about {topic} keep landing on {concept}; "
                f"{item.title} is a series that takes that idea on over many episodes." if deeper
                else f"You've been engaging with {topic}; {item.title} is a series that explores similar themes.")
    elif item.content_type == "video":
        lead = f"If you want {concept} explained rather than dramatised, {item.title} is a video explainer on it."
    else:  # paper
        lead = f"If you want the research behind {concept}, {item.title} is the academic angle."

    parts = [lead]
    if theme and item.content_type in ("movie", "series", "doc"):
        parts.append(f"It also speaks to something you didn't say out loud: {theme}.")
    if item.matched_keywords:
        parts.append(f"Matched on: {', '.join(item.matched_keywords[:3])}.")
    if item.challenge:
        parts.append(f"This one pushes back on your view: {item.challenge}")
    if is_blockbuster(item):
        parts.append(f"Widely watched and well rated ({item.rating:.1f}/10 from {item.popularity:,} votes).")
    elif is_gem(item):
        parts.append(f"Lesser-known, but well rated ({item.rating:.1f}/10 from {item.popularity:,} votes).")
    return " ".join(parts)


def card_extras(sc: "fitmodel.Scored", k: int, item: ContentItem, interest: Interest, query: str) -> dict:
    """Evidence and model facts for card k: receipts, fit features, calibrated confidence (or None)."""
    return {
        "sim": round(sc.sims[k], 4) if sc.query_used else None,
        "fit_features": {f: round(v, 3) for f, v in sc.features[k].items()},
        "fit_score": round(sc.scores[k], 3),
        "cleared_floor": fitmodel.cleared_floor(sc, k),   # internal: may the note claim this fits the person?
        "confidence": sc.confidence(k),          # None = no calibration fitted, so no number is shown
        "receipts": receipts.build(item, interest, query),
        "level": signals.level_of(item),
        "status": item.status or None, "citations": item.citations,
        "survey": item.is_survey or None, "challenge": item.challenge or None,
    }


def render_card(item: ContentItem, interest: Interest, extras: dict | None = None) -> dict:
    """Structured payload for one recommendation card."""
    card = {
        "title": item.title,
        "type": item.content_type,
        "kind": KIND.get(item.content_type, "explore"),
        "year": item.year,
        "url": item.source_url,
        "reasoning": explain_recommendation(item, interest),
        "matched_topic": interest.topic,
        "concept": _concept(interest),
        "evidence": item.matched_keywords[:5],
        "relevance": round(item.topic_overlap_score, 2),
        "medium_rank": (interest.mediums.index(MEDIUM_OF.get(item.content_type, "film")) + 1
                        if MEDIUM_OF.get(item.content_type, "film") in interest.mediums else None),
        "rating": item.rating or None,          # TMDB vote_average, films/docs only
        "votes": item.popularity or None,
        "gem": is_gem(item),
        "blockbuster": is_blockbuster(item),
    }
    if extras:
        card.update(extras)
    # which independent kinds of evidence back this link (see knots/grounding.py); stored so it can be measured offline
    card["evidence_types"] = grounding.evidence_types(item.content_type, item.matched_keywords, card.get("sim"),
                                                      grounding.quoted_theme(interest))
    return card


BLOCKBUSTER_POSITION = 1   # reserved slots go this deep (index), so a cap on the list cannot drop them
GEM_POSITION = 2


def _relevant_enough(item: ContentItem, best: dict[str, float]) -> bool:
    """A famous title only counts if it is nearly as relevant as the best match for its topic."""
    return item.topic_overlap_score >= BLOCKBUSTER_MIN_RELEVANCE * best.get(item.topic, 0.0)


CHALLENGE_POSITION = 3


def rank_and_render(items: list[ContentItem], interests: list[Interest], limit: int = 6,
                    query: str = "") -> list[dict]:
    """Pick the top items across all topics/types and render cards.

    Relevance decides the order (weighted by how well the medium suits the interest);
    rating is only a tie-break, and popularity is deliberately not a signal, or the same
    famous films would win every session. At most one item per (topic, type), so a topic
    yields one documentary (literal), one film (lateral), one series, one paper.

    Two slots are reserved, each only if a candidate exists:
      - a blockbuster: a widely seen, well-rated film or series that is ALSO strongly
        relevant to the topic (see quality.BLOCKBUSTER_MIN_RELEVANCE);
      - a gem: a lesser-known but highly rated film, series or doc.
    A third appears only if the opt-in challenge search found one: a well-rated title that
    complicates the person's own view.

    A candidate under the quality floor (fitmodel.floor, fitted from labels) is never shown, even to fill
    a slot or a reserved blockbuster/gem place: fewer, better picks.

    With `query` (the session text) order comes from fitmodel.py: similarity to what was said,
    quality, format fit and, for papers, authority. Every such card carries receipts, and a card
    with nothing to stand on (no matched tags and no summary) is dropped. Without a query the
    order is the original relevance-times-medium rule, unchanged."""
    interest_by_topic = {i.topic: i for i in interests}
    sc = fitmodel.score_items(query, items, interests, medium_weight) if query.strip() and items else None
    pos = {id(it): k for k, it in enumerate(items)}

    def fit(i: ContentItem) -> float:
        if sc is not None:
            return sc.scores[pos[id(i)]]
        it = interest_by_topic.get(i.topic)
        return i.topic_overlap_score * (medium_weight(i, it) if it else 1.0)

    def make(item: ContentItem) -> dict | None:
        interest = interest_by_topic[item.topic]
        if sc is None:
            return render_card(item, interest)
        if not fitmodel.passes_floor(sc, pos[id(item)]):
            return None                            # below the quality floor: fewer picks beats a weak one
        card = render_card(item, interest, card_extras(sc, pos[id(item)], item, interest, query))
        return card if receipts.sourced(card["receipts"]) else None

    scored = sorted(items, key=lambda i: (fit(i), i.rating), reverse=True)
    best: dict[str, float] = {}                    # best on-screen relevance per topic
    for i in scored:
        if MEDIUM_OF.get(i.content_type) == "film":
            best[i.topic] = max(best.get(i.topic, 0.0), i.topic_overlap_score)

    cards, seen, used_urls = [], set(), set()
    for item in scored:
        interest = interest_by_topic.get(item.topic)
        key = (item.topic, item.content_type)
        if interest is None or key in seen:
            continue
        card = make(item)
        if card is None:
            continue
        seen.add(key)
        used_urls.add(item.source_url)
        cards.append(card)

    def reserve(wanted, flag: str, position: int) -> None:
        if any(c[flag] for c in cards):
            return
        pick = next((i for i in scored if wanted(i) and i.source_url not in used_urls
                     and i.topic in interest_by_topic), None)
        card = make(pick) if pick is not None else None
        if card is not None:
            used_urls.add(pick.source_url)
            cards.insert(min(position, len(cards)), card)

    reserve(lambda i: is_blockbuster(i) and _relevant_enough(i, best), "blockbuster", BLOCKBUSTER_POSITION)
    reserve(is_gem, "gem", GEM_POSITION)
    if any(i.challenge for i in items) and not any(c.get("challenge") for c in cards):
        pick = next((i for i in scored if i.challenge and i.topic in interest_by_topic), None)
        card = make(pick) if pick is not None else None
        if card is not None:
            cards.insert(min(CHALLENGE_POSITION, len(cards)), card)

    # The allowance: a floor that leaves fewer than min_picks cards may be topped up from just under it. These
    # cards are flagged `relaxed`, rank after every card that cleared, and never claim a personal fit.
    need = fitmodel.min_picks() if sc is not None and fitmodel.floor_active() else 0
    for item in scored:
        if len(cards) >= need:
            break
        interest = interest_by_topic.get(item.topic)
        key = (item.topic, item.content_type)
        if interest is None or key in seen or item.source_url in used_urls:
            continue
        k = pos[id(item)]
        if not fitmodel.within_allowance(sc, k):
            continue
        card = render_card(item, interest, card_extras(sc, k, item, interest, query))
        if not receipts.sourced(card["receipts"]):
            continue
        card["relaxed"] = True
        seen.add(key)
        used_urls.add(item.source_url)
        cards.append(card)
    return cards[:limit]

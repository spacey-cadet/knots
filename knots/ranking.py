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

from .catalogs import ContentItem
from .extraction import Interest
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
    if interest.depth == "deep" and item.content_type in ("movie", "series", "doc"):
        parts.append("Since you're going deep on this, there's a paper that pairs with it.")
    if is_blockbuster(item):
        parts.append(f"Widely watched and well rated ({item.rating:.1f}/10 from {item.popularity:,} votes).")
    elif is_gem(item):
        parts.append(f"Lesser-known, but well rated ({item.rating:.1f}/10 from {item.popularity:,} votes).")
    return " ".join(parts)


def render_card(item: ContentItem, interest: Interest) -> dict:
    """Structured payload for one recommendation card."""
    return {
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


BLOCKBUSTER_POSITION = 1   # reserved slots go this deep (index), so a cap on the list cannot drop them
GEM_POSITION = 2


def _relevant_enough(item: ContentItem, best: dict[str, float]) -> bool:
    """A famous title only counts if it is nearly as relevant as the best match for its topic."""
    return item.topic_overlap_score >= BLOCKBUSTER_MIN_RELEVANCE * best.get(item.topic, 0.0)


def rank_and_render(items: list[ContentItem], interests: list[Interest], limit: int = 6) -> list[dict]:
    """Pick the top items across all topics/types and render cards.

    Relevance decides the order (weighted by how well the medium suits the interest);
    rating is only a tie-break, and popularity is deliberately not a signal, or the same
    famous films would win every session. At most one item per (topic, type), so a topic
    yields one documentary (literal), one film (lateral), one series, one paper.

    Two slots are reserved, each only if a candidate exists:
      - a blockbuster: a widely seen, well-rated film or series that is ALSO strongly
        relevant to the topic (see quality.BLOCKBUSTER_MIN_RELEVANCE);
      - a gem: a lesser-known but highly rated film, series or doc."""
    interest_by_topic = {i.topic: i for i in interests}

    def fit(i: ContentItem) -> float:
        it = interest_by_topic.get(i.topic)
        return i.topic_overlap_score * (medium_weight(i, it) if it else 1.0)

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
        seen.add(key)
        used_urls.add(item.source_url)
        cards.append(render_card(item, interest))

    def reserve(wanted, flag: str, position: int) -> None:
        if any(c[flag] for c in cards):
            return
        pick = next((i for i in scored if wanted(i) and i.source_url not in used_urls
                     and i.topic in interest_by_topic), None)
        if pick is not None:
            used_urls.add(pick.source_url)
            cards.insert(min(position, len(cards)), render_card(pick, interest_by_topic[pick.topic]))

    reserve(lambda i: is_blockbuster(i) and _relevant_enough(i, best), "blockbuster", BLOCKBUSTER_POSITION)
    reserve(is_gem, "gem", GEM_POSITION)
    return cards[:limit]

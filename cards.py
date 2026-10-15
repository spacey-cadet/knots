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

from catalogs import ContentItem, is_gem
from interests import Interest

# What the person does with each type. Used to group output for a screen/voice.
KIND = {"movie": "watch", "series": "watch", "doc": "watch", "video": "explore", "paper": "read"}


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
    if is_gem(item):
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
        "rating": item.rating or None,          # TMDB vote_average, films/docs only
        "votes": item.popularity or None,
        "gem": is_gem(item),
    }


GEM_POSITION = 2   # insert the reserved lesser-known pick at most this deep (index), so it survives any cap


def rank_and_render(items: list[ContentItem], interests: list[Interest], limit: int = 6) -> list[dict]:
    """Pick the top items across all topics/types and render cards.

    Relevance decides the order; rating is only a tie-break (items already passed
    the quality gate in catalogs). Popularity is deliberately not a signal, or
    the same famous films win every session. At most one item per
    (topic, type), so a topic yields one documentary (literal), one film
    (lateral), one paper. One slot is reserved for a lesser-known, highly rated
    film or doc when one exists."""
    interest_by_topic = {i.topic: i for i in interests}
    scored = sorted(items, key=lambda i: (i.topic_overlap_score, i.rating), reverse=True)

    cards, seen, used_urls = [], set(), set()
    for item in scored:
        interest = interest_by_topic.get(item.topic)
        key = (item.topic, item.content_type)
        if interest is None or key in seen:
            continue
        seen.add(key)
        used_urls.add(item.source_url)
        cards.append(render_card(item, interest))

    if not any(c["gem"] for c in cards):
        gem = next((i for i in scored if is_gem(i) and i.source_url not in used_urls
                    and i.topic in interest_by_topic), None)
        if gem is not None:
            cards.insert(min(GEM_POSITION, len(cards)), render_card(gem, interest_by_topic[gem.topic]))
    return cards[:limit]

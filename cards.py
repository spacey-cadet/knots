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

from catalogs import ContentItem
from interests import Interest

# What the person does with each type. Used to group output for a screen/voice.
KIND = {"movie": "watch", "doc": "watch", "video": "explore", "paper": "read"}


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
    elif item.content_type == "video":
        lead = f"For {concept}, this explainer breaks it down visually."
    else:  # paper
        lead = f"If you want the research behind {concept}, {item.title} is the academic angle."

    parts = [lead]
    if theme and item.content_type in ("movie", "doc"):
        parts.append(f"It also speaks to something you didn't say out loud: {theme}.")
    if item.matched_keywords:
        parts.append(f"Matched on: {', '.join(item.matched_keywords[:3])}.")
    if interest.depth == "deep" and item.content_type in ("movie", "doc"):
        parts.append("Since you're going deep on this, there's a paper that pairs with it.")
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
    }


def rank_and_render(items: list[ContentItem], interests: list[Interest], limit: int = 6) -> list[dict]:
    """Pick the top-scoring items across all topics/types and render cards.
    At most one item per (topic, type), so a topic yields one documentary
    (literal), one film (lateral), one paper, not three near-identical films."""
    interest_by_topic = {i.topic: i for i in interests}
    scored = sorted(items, key=lambda i: (i.topic_overlap_score, i.popularity), reverse=True)

    cards, seen = [], set()
    for item in scored:
        interest = interest_by_topic.get(item.topic)
        key = (item.topic, item.content_type)
        if interest is None or key in seen:
            continue
        seen.add(key)
        cards.append(render_card(item, interest))
        if len(cards) >= limit:
            break
    return cards

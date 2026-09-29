"""
explain_recommendation + render_card: turns a matched ContentItem plus the
interest that produced it into a human-readable "why", then packages a set
of them into the structured visual card payload Alexa+ would render on a
screened device.
"""

from __future__ import annotations

from catalogs import ContentItem
from interests import Interest

_TEMPLATES = {
    "movie": "You've been engaging with {topic} — {title} explores similar themes on screen.",
    "doc": "Given your interest in {topic}, {title} covers the real-world story behind it.",
    "video": "Since you've been researching {topic}, this explainer breaks it down visually.",
    "paper": "You've been going deep on {topic} — {title} is the academic angle on it.",
}


def explain_recommendation(item: ContentItem, interest: Interest) -> str:
    """Return one or two sentences of grounded reasoning, not a generic blurb."""
    template = _TEMPLATES.get(item.content_type, "Related to your interest in {topic}: {title}.")
    reason = template.format(topic=interest.topic, title=item.title)
    if interest.depth == "deep" and item.content_type != "paper":
        reason += " Since you've been going deep on this, you may want the paper too."
    return reason


def render_card(item: ContentItem, interest: Interest) -> dict:
    """Structured payload for one recommendation card."""
    return {
        "title": item.title,
        "type": item.content_type,
        "url": item.source_url,
        "reasoning": explain_recommendation(item, interest),
        "matched_topic": interest.topic,
        "relevance": round(item.topic_overlap_score, 2),
    }


def rank_and_render(items: list[ContentItem], interests: list[Interest], limit: int = 6) -> list[dict]:
    """Pick the top-scoring items across all topics/types and render cards."""
    interest_by_topic = {i.topic: i for i in interests}
    scored = sorted(items, key=lambda i: i.topic_overlap_score, reverse=True)

    cards, seen_types_per_topic = [], {}
    for item in scored:
        interest = interest_by_topic.get(item.topic)
        if interest is None:
            continue
        key = (item.topic, item.content_type)
        # avoid piling up 3 movies for the same topic; keep it varied
        if seen_types_per_topic.get(key, 0) >= 1:
            continue
        seen_types_per_topic[key] = seen_types_per_topic.get(key, 0) + 1
        cards.append(render_card(item, interest))
        if len(cards) >= limit:
            break
    return cards

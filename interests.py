"""
extract_interests: turns raw signal (Alexa+ conversation snippets, Reddit
activity) into a small list of topics with a depth and sentiment estimate.

v1 uses a lightweight heuristic extractor (keyword/phrase frequency + a
stopword filter) so the whole pipeline is runnable with zero API keys.
Swap `heuristic_extract` for a Bedrock or Claude API call later (see the
`llm_extract` stub below) without changing anything downstream — every
other tool only ever sees the `{topic, depth, sentiment}` shape.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Literal

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "i", "you", "he", "she", "it", "we", "they", "them", "him", "us",
    "my", "your", "his", "her", "its", "our", "their", "this", "that",
    "these", "those", "to", "of", "in", "on", "at", "for", "with", "about",
    "as", "by", "from", "up", "down", "into", "over", "after", "some",
    "than", "so", "just", "reading", "watching", "listening",
    "asking", "wondering", "thinking", "lately", "recently",
    # assistant/voice-interaction filler and question words — these dominate
    # raw Alexa+ transcript text but are never the actual topic
    "alexa", "what", "whats", "why", "how", "who", "when", "where", "which",
    "can", "tell", "explain", "keep", "seeing", "feed", "feeds", "threads",
    "thread", "me", "more", "like", "all", "know", "want", "think", "mean",
    "really", "kind", "sort", "let", "also", "get", "got", "much", "many",
    "people", "person", "things", "thing", "stuff",
}

Depth = Literal["casual", "deep"]
Sentiment = Literal["curious", "engaged", "skeptical", "neutral"]

DEEP_MARKERS = (
    "research", "paper", "study", "deep dive", "whitepaper", "thesis",
    "mechanism", "analysis", "technical", "how does", "why does",
)
SKEPTIC_MARKERS = ("scam", "bubble", "worried", "risk", "skeptical", "is it real")


@dataclass
class Interest:
    topic: str
    depth: Depth
    sentiment: Sentiment
    mentions: int

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_tokens(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s#]", " ", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]


def _extract_ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def heuristic_extract(session_texts: list[str], top_k: int = 5) -> list[Interest]:
    """
    session_texts: raw strings from Alexa+ conversation turns and/or Reddit
    post titles / comment bodies the user engaged with.

    Returns the top_k topics by mention frequency, each tagged with a rough
    depth (casual vs deep engagement) and sentiment, inferred from marker
    words in the surrounding text.
    """
    all_text = " ".join(session_texts)
    tokens = _clean_tokens(all_text)

    # A bigram only earns "topic" status if it repeats — a one-off pair like
    # "people buy" is noise, not a signal. Repeating bigrams read better as
    # topics than single words when they exist ("token economics" beats
    # "token" and "economics" separately).
    bigrams = _extract_ngrams(tokens, 2)
    bigram_counts = Counter(bigrams)
    strong_bigrams = {b: c for b, c in bigram_counts.items() if c >= 2}

    unigram_counts = Counter(tokens)
    words_in_strong_bigrams = {w for b in strong_bigrams for w in b.split()}

    counts: Counter = Counter(strong_bigrams)
    for word, c in unigram_counts.most_common():
        if word in words_in_strong_bigrams:
            continue  # already represented inside a stronger bigram
        counts[word] = c

    top = counts.most_common(top_k)
    interests: list[Interest] = []
    lowered_all = all_text.lower()

    for topic, mentions in top:
        depth: Depth = "deep" if any(m in lowered_all for m in DEEP_MARKERS) else "casual"
        sentiment: Sentiment = "neutral"
        if any(m in lowered_all for m in SKEPTIC_MARKERS):
            sentiment = "skeptical"
        elif mentions >= 3:
            sentiment = "engaged"
        else:
            sentiment = "curious"
        interests.append(Interest(topic=topic, depth=depth, sentiment=sentiment, mentions=mentions))

    return interests


def llm_extract(session_texts: list[str]) -> list[Interest]:
    """
    Stretch path: call Bedrock / Claude to do proper semantic topic
    extraction instead of frequency counting. Left unimplemented in v1 —
    wire this up for the AWS Builder mini challenge. Must return the same
    list[Interest] shape as heuristic_extract so nothing downstream changes.
    """
    raise NotImplementedError("Swap in a Bedrock/Claude call here for v2")

"""
The quality gate: which titles are good enough to recommend at all.

A gate, not a ranking signal. Fame is not quality, and relevance (not rating, not
popularity) decides the order among titles that pass. Everything here is a plain
number you can tune; nothing else in the pipeline hard-codes these thresholds.
"""

from __future__ import annotations

# Quality floor (a gate, not a ranking signal). Fame is not quality, and
# relevance, not rating, decides order among films that pass.
MIN_VOTE_COUNT = 100         # also sent to TMDB so discover drops junk early
MIN_RATING = 6.5
OLD_FILM_BEFORE = 1990       # older films must clear a higher bar
OLD_MIN_RATING = 7.5
OLD_MIN_VOTES = 1000
GEM_MAX_VOTES = 3000         # "lesser-known": few votes ...
GEM_MIN_RATING = 7.0         # ... but highly rated
# TV gets fewer votes than film and skews higher-rated, so its floor differs.
TV_MIN_VOTE_COUNT = 150
TV_MIN_RATING = 7.0
TV_GEM_MAX_VOTES = 1500


def passes_quality(rating: float, votes: int, year: str, series: bool = False) -> bool:
    """Floor for a TMDB film or series: rating plus a minimum vote count, higher for old titles."""
    try:
        old = bool(year) and int(year) < OLD_FILM_BEFORE
    except ValueError:
        old = False
    if old:
        return rating >= OLD_MIN_RATING and votes >= OLD_MIN_VOTES
    if series:
        return rating >= TV_MIN_RATING and votes >= TV_MIN_VOTE_COUNT
    return rating >= MIN_RATING and votes >= MIN_VOTE_COUNT


def is_gem(item: "ContentItem") -> bool:
    """Lesser-known but highly rated film/doc (the reserved slot in the picks)."""
    if item.content_type == "series":
        return item.popularity <= TV_GEM_MAX_VOTES and item.rating >= GEM_MIN_RATING
    return (item.content_type in ("movie", "doc") and item.popularity <= GEM_MAX_VOTES
            and item.rating >= GEM_MIN_RATING)

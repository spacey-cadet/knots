"""Knots: turns what a person is curious about into films, series, explainers and papers.

The whole recommendation algorithm is this package; its single entry point is `knots.engine`.
"""

from . import engine
from .catalogs import ContentItem
from .context import Conversation
from .engine import extract, forget, plan, recommend, run, search, what_remembered
from .extraction import Interest

__all__ = ["engine", "Conversation", "Interest", "ContentItem",
           "run", "plan", "recommend", "extract", "search", "what_remembered", "forget"]

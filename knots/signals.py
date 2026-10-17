"""
Cheap, deterministic signals about a candidate, used by ranking and pairing.

  is_survey(title, abstract)   a review/overview/tutorial paper: the accessible entry to a field
  reading_level(text)          0 (plain) .. 1 (dense), from Flesch reading ease
  level_of(item)               how demanding the item is, 0 (easy, lean-back) .. 1 (dense)
  authority(item)              0..1 for papers: citations (log scale) and survey status

All are heuristics. They are inputs to a ranker whose weights are learned from labels (or are
stated hand-set priors), never ground truth, and none of them is a popularity ranking for films.
"""
from __future__ import annotations

import math
import re

_SURVEY = re.compile(r"\b(survey|review|overview|tutorial|primer|state[- ]of[- ]the[- ]art|taxonomy|"
                     r"introduction to|an introduction|a guide|lecture notes)\b", re.I)
_SURVEY_ABSTRACT = re.compile(r"\b(?:this|we) (?:survey|review|overview|provide an overview|give an overview)\b", re.I)
_EASY_VIDEO = re.compile(r"\b(for beginners|explained|in \d+ minutes|simply|intro(?:duction)?|basics)\b", re.I)
_HARD_VIDEO = re.compile(r"\b(lecture|course|advanced|seminar|colloquium|talk at)\b", re.I)

CITATION_SCALE = math.log1p(5000)     # ~5,000 citations saturates the citation signal


def is_survey(title: str, abstract: str = "") -> bool:
    return bool(_SURVEY.search(title or "") or _SURVEY_ABSTRACT.search(abstract or ""))


def _syllables(word: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", word.lower())))


def reading_level(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text or "")
    sentences = max(1, len(re.findall(r"[.!?]+", text or "")))
    if len(words) < 8:
        return 0.5                      # too little text to judge
    fre = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (sum(map(_syllables, words)) / len(words))
    return round(min(1.0, max(0.0, (70.0 - fre) / 70.0)), 3)


def level_of(item) -> float:
    t = item.content_type
    if t == "paper":
        base = reading_level(item.abstract or item.blurb)
        return round(max(0.0, base - 0.15) if item.is_survey else min(1.0, base + 0.1), 3)
    if t == "video":
        v = 0.35
        v += 0.15 if item.duration_s > 1800 else 0.0
        v -= 0.10 if _EASY_VIDEO.search(item.title or "") else 0.0
        v += 0.20 if _HARD_VIDEO.search(item.title or "") else 0.0
        return round(min(1.0, max(0.0, v)), 3)
    return 0.3 if t == "doc" else 0.2    # drama is the easiest way in


def authority(item) -> float:
    """Paper authority in 0..1: log citations plus a survey bonus. 0 for non-papers and unknowns."""
    if item.content_type != "paper":
        return 0.0
    cites = math.log1p(max(0, item.citations or 0)) / CITATION_SCALE
    return round(min(1.0, 0.8 * min(1.0, cites) + (0.2 if item.is_survey else 0.0)), 3)

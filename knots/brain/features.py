"""
Features for the theme critic: cheap, deterministic, offline, no model call.

Each one encodes a failure pattern the author named while labelling the evaluation themes
("Archetype 1..5" in eval/themes.csv), so the critic learns THEIR taste, not a generic one:

  1  over-psychologizing    a feeling attributed to the person who never stated one
  2  terminological overreach  a specific term or concept that is not in their words
  3  narrative projection   a dramatic or ironic story the text does not tell
  5  binary imposition      an invented "X vs Y"

and the pattern of the themes they accepted: a stance or a mechanism stated in the person's own
terms ("whether choice is meaningful", "gap between knowledge and will").
"""

from __future__ import annotations

import re

FEATURE_NAMES = [
    "ground",             # share of the theme's content words that occur in the session text
    "ungrounded",         # content words of the theme missing from the session text
    "ungrounded_long",    # ... of which long words (jargon proxy)
    "affect_theme",       # feeling words in the theme (fear, anxiety, grief, unease ...)
    "feeling_stated",     # the person states a feeling of their own ("why do I feel lonely")
    "affect_unstated",    # a feeling in the theme that the person never stated (archetype 1)
    "vs_pattern",         # "X vs Y" / "versus" (archetype 5)
    "drama",              # dramatic markers (collapse, erosion, point of no return, loss of ...)
    "n_words",            # theme length
    "stance_pattern",     # "whether ...", "gap between", "limits of", "tension"
    "mechanism",          # structural vocabulary (system, incentive, mechanism, structure ...)
]

_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "into", "about", "over", "than", "their",
    "them", "they", "your", "what", "when", "which", "while", "where", "does", "have", "been",
    "being", "more", "most", "some", "such", "only", "just", "also", "very", "much", "many",
    "between", "without", "within", "against", "toward", "towards", "around", "across",
}
_WORD = re.compile(r"[a-z0-9']+")

_AFFECT = ("fear", "anxie", "anxio", "unease", "uneas", "dread", "grief", "griev", "sorro", "sadne", "lonel",
           "shame", "guilt", "worry", "worri", "resent", "longi", "yearn", "despa", "hopel", "panic", "terro",
           "insecu", "distru", "resig", "anger", "angry", "frust", "boredo", "bored", "uncert", "disillu")
_DRAMA = ("collaps", "erosio", "eroded", "erode", "disapp", "vanish", "surviv", "absurd", "crisis", "doom",
          "abyss", "apocal", "destin", "no return", "who control", "loss of", "personhood", "selfhood", "death")
_MECHANISM = ("system", "incentiv", "mechani", "structur", "econom", "institut", "polic", "infrastr", "platform",
              "algorith", "optimis", "optimiz", "verific", "signal", "amplif")
_FEELING_STATED = re.compile(
    r"\b(?:i feel|i felt|i'm feeling|i am feeling|feel(?:ing)? (?:more |so |really )?(?:lonely|sad|anxious|scared|"
    r"afraid|angry|depressed)|i(?: am|'m) (?:so |really )?(?:worried|scared|afraid|anxious|sad|lonely|angry)|"
    r"makes me (?:feel|sad|anxious|scared|angry)|it scares me|i hate|i miss)\b")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _stem(w: str) -> str:
    return w[:5] if len(w) > 5 else w


def _count(text: str, stems: tuple[str, ...]) -> int:
    low = text.lower()
    return sum(low.count(s) for s in stems)


def features(session_text: str, theme: str) -> list[float]:
    """Vector in FEATURE_NAMES order. `session_text` is what the person said (turns joined)."""
    theme_l = theme.lower()
    content = [w for w in _words(theme_l) if len(w) > 3 and w not in _STOP]
    seen = {_stem(w) for w in _words(session_text)}
    missing = [w for w in content if _stem(w) not in seen]
    ground = 1.0 - len(missing) / len(content) if content else 0.0

    affect = min(3, _count(theme_l, _AFFECT))
    stated = 1.0 if _FEELING_STATED.search(session_text.lower()) else 0.0
    return [
        ground,
        float(min(4, len(missing))),
        float(min(3, sum(1 for w in missing if len(w) >= 9))),
        float(affect),
        stated,
        float(affect and not stated),
        1.0 if re.search(r"\bvs\.?\b|\bversus\b", theme_l) else 0.0,
        float(min(3, _count(theme_l, _DRAMA))),
        float(min(10, len(_words(theme_l)))),
        1.0 if re.search(r"\bwhether\b|\bgap between\b|\blimits? of\b|\btension\b", theme_l) else 0.0,
        float(min(3, _count(theme_l, _MECHANISM))),
    ]

"""Few-shot guidance: the author's own accepted/rejected themes, retrieved by similarity to the
current text and injected into the extraction prompt, plus a short list of written rules.

The label is the verdict (supported y/n). A reviewer's free-text note about WHY a theme failed is a
diagnostic, not a label, so it is not shown to the model unless KNOTS_GUIDANCE_NOTES=1 (an experiment
whose effect cannot be measured on the same labels it was written from)."""
from __future__ import annotations

import json
import os

from ..embeddings import cosine, hashed_embedding
from .critic import DATA

RULES = """Theme rules learned from a reviewer's labels (follow them):
- State a stance or a mechanism in the person's own terms ("whether X is meaningful", "gap between A and B", \
how a system or incentive works). These were accepted.
- Do NOT attribute a feeling (fear, unease, grief, longing) the person never expressed.
- Do NOT introduce a technical term or concept the person did not use or clearly imply.
- Do NOT add a dramatic or ironic story arc (collapse, erosion, loss of selfhood) the text does not tell.
- Do NOT invent an "X vs Y" opposition the person did not draw.
- Fewer themes is better; return none rather than a stretch."""


def _examples() -> list[dict]:
    try:
        with open(DATA / "theme_examples.jsonl", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except OSError:
        return []


def guidance(session_text: str, k: int = 3, holdout: str | None = None) -> str:
    """Rules + k accepted and k rejected examples nearest to `session_text`.
    `holdout` excludes examples drawn from that session id (honest evaluation)."""
    if os.environ.get("KNOTS_BRAIN", "on").lower() == "off":
        return ""
    ex = [e for e in _examples() if e["session_id"] != holdout]
    if not ex:
        return RULES
    q = hashed_embedding(session_text)
    ex.sort(key=lambda e: -cosine(q, hashed_embedding(e["session_text"] + " " + e["theme"])))
    good = [e for e in ex if e["supported"] == "y"][:k]
    bad = [e for e in ex if e["supported"] == "n"][:k]
    lines = [RULES, "", "Accepted themes (from other conversations):"]
    lines += [f'- "{e["theme"]}"' for e in good]
    with_notes = os.environ.get("KNOTS_GUIDANCE_NOTES") == "1"
    lines.append("Rejected themes, with why:" if with_notes else "Rejected themes:")
    lines += [f'- "{e["theme"]}"' + (f' — {e["note"]}' if with_notes and e.get("note") else "") for e in bad]
    return "\n".join(lines)

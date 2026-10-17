#!/usr/bin/env python3
"""
Check the hand labels for problems, and (optionally) fill `error_type` from your notes.

    python eval/check_labels.py                      # report only
    python eval/check_labels.py --fill-error-types   # write error_type where a note names the error

It never changes a `supported` / `good` label. It reports things for YOU to decide:
  - a theme marked supported whose note describes an error (a likely mis-click)
  - an unsupported theme with no error_type, and a not-good pick with no reason
  - the same theme labelled twice in a session
  - holdout rows whose notes use the training rubric's template ("Archetype N"), which suggests
    they were drafted from it rather than judged fresh (the holdout must be your own judgement)
  - how many rows are still unlabelled, by split
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as R  # noqa: E402

# name in a note -> error_type. Order matters only for ties; the earliest mention wins.
PATTERNS = [
    ("over_psychologizing", r"over-?psycholog|psychologiz"),
    ("terminological_overreach", r"terminolog"),
    ("narrative_projection", r"narrative projection"),
    ("binary_imposition", r"binary imposition|binary opposition"),
    ("normative_expansion", r"normative expansion|normative"),
    ("causal_projection", r"causal projection"),
]
NEGATION = re.compile(r"\b(?:no|not|without|never|avoids?|free of)\b[^.;]{0,40}$", re.I)


def named_errors(note: str) -> list[str]:
    """Error types a note names affirmatively, in order of first mention (negated mentions skipped)."""
    hits = []
    for name, pat in PATTERNS:
        for m in re.finditer(pat, note, re.I):
            if not NEGATION.search(note[:m.start()]):
                hits.append((m.start(), name))
                break
    return [n for _, n in sorted(hits)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill-error-types", action="store_true")
    args = ap.parse_args()
    themes, picks, pairs = R.read_csv(R.THEMES), R.read_csv(R.LABELS), R.read_csv(R.PAIRS)
    problems = 0

    filled = 0
    for r in themes:
        if r["supported"] == "n" and not r.get("error_type"):
            names = named_errors(r["note"])
            if names and args.fill_error_types:
                r["error_type"] = names[0]
                filled += 1
    if args.fill_error_types:
        R.write_csv(R.THEMES, R.THEME_COLS, themes)
        print(f"Filled error_type on {filled} unsupported themes from your notes (edit freely).")

    print("\n== Themes ==")
    for r in themes:
        if r["supported"] == "y" and named_errors(r["note"]):
            problems += 1
            print(f"  CHECK supported=y but the note names an error ({', '.join(named_errors(r['note']))}): "
                  f"{r['session_id']} | {r['theme']}")
    unclassified = [r for r in themes if r["supported"] == "n" and not r.get("error_type")]
    print(f"  unsupported themes with no error_type: {len(unclassified)} of "
          f"{sum(r['supported'] == 'n' for r in themes)}")
    dup = Counter((r["session_id"], r["extractor"], r["theme"]) for r in themes)
    for k, n in dup.items():
        if n > 1:
            problems += 1
            print(f"  DUPLICATE x{n}: {k}")
    print("  error types:", dict(Counter(r["error_type"] or "(none)" for r in themes if r["supported"] == "n")))

    print("\n== Picks ==")
    nr = [r for r in picks if r["good"] == "n" and not r["reason"]]
    print(f"  not-good picks with no reason: {len(nr)}")

    print("\n== Holdout hygiene ==")
    for r in themes + picks + pairs:
        if r.get("split") == "holdout" and re.search(r"archetype\s*\d", r.get("note", ""), re.I):
            problems += 1
            print(f"  CHECK holdout note uses the training template: {r['session_id']} | {r.get('theme') or r.get('title')}")

    print("\n== Unlabelled, by split ==")
    for name, rows, col in (("themes", themes, "supported"), ("picks", picks, "good"), ("pairs", pairs, "good")):
        c = Counter((r.get("split", "train"), "labelled" if r[col] in ("y", "n") else "UNLABELLED") for r in rows)
        print(f"  {name:7s}", dict(c))
    print(f"\n{problems} thing(s) for you to check." if problems else "\nNo problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

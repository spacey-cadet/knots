#!/usr/bin/env python3
"""
Evaluation harness for Knots.

Runs the same sessions (eval/sessions.json) through one or more extractors,
saves the raw outputs, and builds two spreadsheets for you to label by hand:

  eval/labels.csv   one row per recommended pick  -> label `good` (y/n)
  eval/themes.csv   one row per "unstated theme"  -> label `supported` (y/n)

Then `score` turns your labels into a results table (eval/RESULTS.md).

Usage (from the repo root, venv active):

  python eval/run_eval.py run                              # heuristic + bedrock
  python eval/run_eval.py run --extractors heuristic       # offline only
  python eval/run_eval.py run --extractors "heuristic,bedrock=us.anthropic.claude-sonnet-4-6"
  python eval/run_eval.py score

`bedrock` runs in strict mode here, so a failure is recorded as an error
instead of silently falling back to the heuristic (which would fake a result).
Re-running `run` keeps the labels you already entered.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import interests as I  # noqa: E402
from catalogs import search_content  # noqa: E402
from partner import build_session_plan  # noqa: E402

SESSIONS = HERE / "sessions.json"
RESULTS = HERE / "results" / "results.json"
LABELS = HERE / "labels.csv"
THEMES = HERE / "themes.csv"
REPORT = HERE / "RESULTS.md"

LABEL_COLS = ["session_id", "session_text", "thread", "kind", "type", "title", "year",
              "url", "evidence", "reasoning", "live", "good", "note", "extractor"]
THEME_COLS = ["session_id", "session_text", "topic", "theme", "rationale",
              "supported", "note", "extractor"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_spec(spec: str) -> tuple[str, str | None]:
    """'heuristic' | 'bedrock' | 'bedrock=<model id>'."""
    if "=" in spec:
        mode, model = spec.split("=", 1)
        return mode.strip(), model.strip()
    return spec.strip(), None


def run_extractor(mode: str, model: str | None, texts: list[str]):
    if mode not in ("heuristic", "bedrock"):
        raise ValueError(f"unknown extractor '{mode}' (use heuristic or bedrock)")
    os.environ["EXTRACTOR"] = mode
    if mode == "bedrock":
        os.environ["EXTRACTOR_STRICT"] = "1"   # never silently fall back during an eval
    else:
        os.environ.pop("EXTRACTOR_STRICT", None)
    if model:
        os.environ["BEDROCK_MODEL_ID"] = model
    return I.extract_sync(texts)


def is_live(card: dict) -> bool:
    """False for the fixed offline mock items, which say nothing about the pipeline."""
    url, kind = card.get("url", ""), card.get("type", "")
    if kind in ("movie", "doc"):
        return "/movie/" in url            # live TMDB items; mocks are /tv/ links
    if kind == "paper":
        return "2201.00000" not in url     # the mock paper
    if kind == "video":
        return "search_query" not in url   # the mock video
    return False


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def yn(value: str) -> bool | None:
    v = (value or "").strip().lower()
    if v in ("y", "yes", "1", "true", "good"):
        return True
    if v in ("n", "no", "0", "false", "bad"):
        return False
    return None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    sessions = json.loads(SESSIONS.read_text())
    specs = [s.strip() for s in args.extractors.split(",") if s.strip()]
    tmdb_live = bool(os.environ.get("TMDB_API_KEY"))
    if not tmdb_live:
        print("WARNING: TMDB_API_KEY is not set. Films will be the fixed offline mock set, "
              "so every session gets the same films and the labels will not mean much.\n")

    results, errors = [], 0
    for sess in sessions:
        for spec in specs:
            mode, model = parse_spec(spec)
            entry = {"session_id": sess["id"], "extractor": spec, "texts": sess["texts"]}
            try:
                found, used = run_extractor(mode, model, sess["texts"])
                items = asyncio.run(search_content(found, include_videos=args.videos))
                plan = build_session_plan(found, items, extractor=used)
                entry.update(ok=True, used=used, interests=[i.to_dict() for i in found], plan=plan)
                n = sum(len(v) for v in plan["picks"].values())
                print(f"  {sess['id']}  {spec:<42} ok   {len(found)} interests, {n} picks")
            except Exception as e:  # noqa: BLE001 - record and keep going
                errors += 1
                entry.update(ok=False, error=f"{type(e).__name__}: {e}")
                print(f"  {sess['id']}  {spec:<42} FAIL {type(e).__name__}: {str(e)[:90]}")
            results.append(entry)

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"tmdb_live": tmdb_live, "results": results}, indent=2))

    build_label_sheets(results)
    print(f"\nSaved raw outputs to {RESULTS.relative_to(ROOT)}")
    print(f"Label picks in   {LABELS.relative_to(ROOT)}   (column: good = y/n)")
    print(f"Label themes in  {THEMES.relative_to(ROOT)}   (column: supported = y/n)")
    if errors:
        print(f"\n{errors} run(s) failed; see FAIL lines above. Fix and re-run: labels you entered are kept.")
    return 0


def build_label_sheets(results: list[dict]) -> None:
    old_labels = {(r["session_id"], r["extractor"], r["url"]): r for r in read_csv(LABELS)}
    old_themes = {(r["session_id"], r["extractor"], r["topic"], r["theme"]): r for r in read_csv(THEMES)}

    pick_rows, theme_rows = [], []
    for e in results:
        if not e.get("ok"):
            continue
        text = " / ".join(e["texts"])
        plan = e["plan"]
        thread = "; ".join(t["topic"] for t in plan["threads"])
        for kind, cards in plan["picks"].items():
            for c in cards:
                prev = old_labels.get((e["session_id"], e["extractor"], c["url"]), {})
                pick_rows.append({
                    "session_id": e["session_id"], "session_text": text, "thread": thread,
                    "kind": kind, "type": c["type"], "title": c["title"], "year": c["year"],
                    "url": c["url"], "evidence": ", ".join(c.get("evidence", [])),
                    "reasoning": c["reasoning"], "live": "yes" if is_live(c) else "no",
                    "good": prev.get("good", ""), "note": prev.get("note", ""),
                    "extractor": e["extractor"],
                })
        for it in e["interests"]:
            for theme in it.get("implicit_themes", []):
                prev = old_themes.get((e["session_id"], e["extractor"], it["topic"], theme), {})
                theme_rows.append({
                    "session_id": e["session_id"], "session_text": text, "topic": it["topic"],
                    "theme": theme, "rationale": it.get("rationale", ""),
                    "supported": prev.get("supported", ""), "note": prev.get("note", ""),
                    "extractor": e["extractor"],
                })

    # Shuffle within each session (fixed seed) so you can label without knowing which
    # extractor produced a row. Hide the last column (`extractor`) while labelling.
    for rows in (pick_rows, theme_rows):
        by_session = defaultdict(list)
        for r in rows:
            by_session[r["session_id"]].append(r)
        rows.clear()
        for sid in sorted(by_session):
            group = by_session[sid]
            random.Random(sid).shuffle(group)
            rows.extend(group)

    write_csv(LABELS, LABEL_COLS, pick_rows)
    write_csv(THEMES, THEME_COLS, theme_rows)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def pct(good: int, total: int) -> str:
    return f"{good}/{total} ({100 * good / total:.0f}%)" if total else "n/a"


def cmd_score(args) -> int:
    picks, themes = read_csv(LABELS), read_csv(THEMES)
    if not picks:
        print("No labels.csv yet. Run `python eval/run_eval.py run` first.")
        return 1

    extractors = sorted({r["extractor"] for r in picks} | {r["extractor"] for r in themes})
    stats = {x: defaultdict(int) for x in extractors}
    for r in picks:
        s = stats[r["extractor"]]
        if r["live"] != "yes" and not args.include_mock:
            s["mock_skipped"] += 1
            continue
        g = yn(r["good"])
        if g is None:
            s["unlabeled"] += 1
            continue
        for key in ("all", r["kind"]):
            s[f"{key}_n"] += 1
            s[f"{key}_good"] += int(g)
    for r in themes:
        s = stats[r["extractor"]]
        s["themes_found"] += 1
        sup = yn(r["supported"])
        if sup is None:
            s["themes_unlabeled"] += 1
        else:
            s["themes_n"] += 1
            s["themes_good"] += int(sup)

    lines = ["# Knots evaluation results", "",
             "Hand-labelled by the author. A pick is *good* if I would genuinely watch/read it "
             "for that session and the stated reason holds up. A theme is *supported* if the "
             "session text genuinely implies it.", "",
             "| Extractor | Good picks (all) | Watch | Read | Unstated themes found | Themes supported |",
             "| --- | --- | --- | --- | --- | --- |"]
    for x in extractors:
        s = stats[x]
        lines.append(
            f"| {x} | {pct(s['all_good'], s['all_n'])} | {pct(s['watch_good'], s['watch_n'])} | "
            f"{pct(s['read_good'], s['read_n'])} | {s['themes_found']} | "
            f"{pct(s['themes_good'], s['themes_n'])} |")
    lines.append("")
    for x in extractors:
        s = stats[x]
        notes = []
        if s["unlabeled"]:
            notes.append(f"{s['unlabeled']} picks still unlabeled")
        if s["themes_unlabeled"]:
            notes.append(f"{s['themes_unlabeled']} themes still unlabeled")
        if s["mock_skipped"]:
            notes.append(f"{s['mock_skipped']} offline mock picks excluded (use --include-mock to count them)")
        if notes:
            lines.append(f"- **{x}:** " + "; ".join(notes))
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text)
    print(text)
    print(f"Saved to {REPORT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run sessions through extractors, build label sheets")
    r.add_argument("--extractors", default="heuristic,bedrock",
                   help="comma list: heuristic, bedrock, or bedrock=<model id> (default: heuristic,bedrock)")
    r.add_argument("--videos", action="store_true", help="also query YouTube (needs YOUTUBE_API_KEY)")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("score", help="turn your labels into a results table")
    s.add_argument("--include-mock", action="store_true", help="count offline mock picks too")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

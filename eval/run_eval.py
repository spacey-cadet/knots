#!/usr/bin/env python3
"""
Evaluation harness for Knots.

Runs the same sessions (eval/sessions.json) through one or more extractors,
saves the raw outputs, and builds two spreadsheets for you to label by hand:

  eval/labels.csv   one row per recommended pick  -> label `good` (y/n), and when n,
                    a `reason`: relevance / quality / repeat / other
  eval/themes.csv   one row per "unstated theme"  -> label `supported` (y/n)

Then `score` turns your labels into a results table (eval/RESULTS.md).

Usage (from the repo root, venv active):

  python eval/run_eval.py run                              # heuristic + bedrock
  python eval/run_eval.py run --extractors heuristic       # offline only
  python eval/run_eval.py run --youtube-budget 95          # stop cleanly when the day's YouTube searches are spent
  python eval/run_eval.py run --resume                     # next day: redo only INCOMPLETE/missing runs
  python eval/run_eval.py run --extractors "heuristic,bedrock=us.anthropic.claude-sonnet-4-6"
  python eval/run_eval.py label                            # label in the browser
  python eval/run_eval.py score

`bedrock` runs in strict mode here, so a failure is recorded as an error
instead of silently falling back to the heuristic (which would fake a result).
Re-running `run` keeps the labels you already entered.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
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
# An eval can afford to wait; a flaky connection must not silently drop picks.
os.environ.setdefault("KNOTS_HTTP_TIMEOUT", "25")
os.environ.setdefault("KNOTS_ARXIV_CACHE", str(HERE / "results" / "arxiv_cache.json"))
# YouTube answers are cached too: the API allows only ~100 searches a day, and a re-run must never pay twice.
os.environ.setdefault("KNOTS_YOUTUBE_CACHE", str(HERE / "results" / "youtube_cache.json"))

from knots import catalogs, embeddings, engine, extraction, fitmodel  # noqa: E402

SESSIONS = HERE / "sessions.json"
RESULTS = HERE / "results" / "results.json"
LABELS = HERE / "labels.csv"
THEMES = HERE / "themes.csv"
REPORT = HERE / "RESULTS.md"

PAIRS = HERE / "pairs.csv"
# `split` is "train" or "holdout" (from sessions.json). Holdout rows are never used to train or
# tune anything; they exist only to measure the system honestly.
LABEL_COLS = ["session_id", "session_text", "thread", "kind", "type", "title", "year",
              "url", "evidence", "reasoning", "live", "good", "reason", "note", "extractor", "split"]
ERROR_TYPES = ("", "over_psychologizing", "terminological_overreach", "narrative_projection",
               "binary_imposition", "normative_expansion", "causal_projection", "other")
THEME_COLS = ["session_id", "session_text", "topic", "theme", "evidence", "rank", "rationale",
              "supported", "error_type", "note", "extractor", "split"]
THEMES_RUN = HERE / "results" / "themes_run.json"   # output of `run_eval.py themes`; the freshest theme source
PAIR_COLS = ["session_id", "session_text", "pair_id", "concept", "watch_title", "watch_url",
             "learn_title", "learn_url", "pair_score", "bridge", "good", "note", "extractor", "split"]


def session_splits() -> dict[str, str]:
    return {s["id"]: s.get("split", "train") for s in json.loads(SESSIONS.read_text())}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_spec(spec: str) -> tuple[str, str | None]:
    """'heuristic' | 'bedrock' | 'bedrock=<model id>'."""
    if "=" in spec:
        mode, model = spec.split("=", 1)
        return mode.strip(), model.strip()
    return spec.strip(), None


def set_extractor(mode: str, model: str | None) -> None:
    if mode not in ("heuristic", "bedrock"):
        raise ValueError(f"unknown extractor '{mode}' (use heuristic or bedrock)")
    os.environ["EXTRACTOR"] = mode
    if mode == "bedrock":
        os.environ["EXTRACTOR_STRICT"] = "1"   # never silently fall back during an eval
    else:
        os.environ.pop("EXTRACTOR_STRICT", None)
    if model:
        os.environ["BEDROCK_MODEL_ID"] = model


def is_live(card: dict) -> bool:
    """False for the fixed offline mock items, which say nothing about the pipeline."""
    return card.get("url", "") not in catalogs.mock_urls()


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

    videos = bool(os.environ.get("YOUTUBE_API_KEY")) and not args.no_videos
    if not videos:
        print("NOTE: YouTube is off (no YOUTUBE_API_KEY, or --no-videos), so no explainers will appear.\n")
    if args.youtube_budget is not None:
        os.environ["KNOTS_YOUTUBE_BUDGET"] = str(args.youtube_budget)

    def complete(e) -> bool:
        return bool(e and e.get("ok") and not e.get("catalog_failures"))

    previous = {}
    if args.resume:
        try:
            for e in json.loads(RESULTS.read_text()).get("results", []):
                previous[(e["session_id"], e["extractor"])] = e
        except (OSError, ValueError):
            print("NOTE: --resume found no earlier results.json; running everything.\n")
    order = [(s["id"], spec) for s in sessions for spec in specs]
    entries: dict[tuple, dict] = {}

    def save() -> None:
        """Written after every run, so an interrupted run loses nothing."""
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps({"tmdb_live": tmdb_live, "embedder": embeddings.embedder_config(),
                                       "results": [entries[k] for k in order if k in entries]}, indent=2))

    errors, kept, stopped = 0, 0, None
    for sess in sessions:
        for spec in specs:
            key = (sess["id"], spec)
            old = previous.get(key)
            if old and complete(old) and old.get("texts") == sess["texts"]:
                entries[key] = old
                kept += 1
                print(f"  {sess['id']}  {spec:<42} kept (complete from an earlier run)")
                continue
            if videos and not stopped and catalogs.youtube_searches_left() == 0:
                why = catalogs.youtube_blocked_reason() or f"the daily budget of {catalogs.youtube_budget()} searches is spent"
                stopped = (sess["id"], spec, why)
            if stopped:
                if old:                      # keep whatever an earlier run produced for it, flagged as stale
                    entries[key] = old
                continue
            mode, model = parse_spec(spec)
            entry = {"session_id": sess["id"], "extractor": spec, "texts": sess["texts"]}
            try:
                catalogs.FAILURES.clear()
                fitmodel.USED_EMBEDDERS.clear()
                set_extractor(mode, model)
                extraction._HOLDOUT = sess["id"]  # hide this session's own labelled themes from few-shot
                result = asyncio.run(engine.run(session_texts=sess["texts"], include_videos=videos))
                found, used, plan = result.interests, result.extractor, result.plan()
                failed = list(catalogs.FAILURES)
                entry.update(ok=True, used=used, interests=[i.to_dict() for i in found], plan=plan,
                             catalog_failures=failed, embedders_used=dict(fitmodel.USED_EMBEDDERS))
                n = sum(len(v) for v in plan["picks"].values())
                warn = f"  INCOMPLETE: {len(failed)} catalog call(s) failed" if failed else ""
                print(f"  {sess['id']}  {spec:<42} ok   {len(found)} interests, {n} picks{warn}")
            except Exception as e:  # noqa: BLE001 - record and keep going
                errors += 1
                entry.update(ok=False, error=f"{type(e).__name__}: {e}")
                print(f"  {sess['id']}  {spec:<42} FAIL {type(e).__name__}: {str(e)[:90]}")
            entries[key] = entry
            save()

    results = [entries[k] for k in order if k in entries]
    incomplete = sum(1 for e in results if e.get("ok") and e.get("catalog_failures"))
    pending = [k for k in order if k not in entries or not complete(entries[k])]
    cfg = embeddings.embedder_config()
    fell_back = sum(1 for e in results if e.get("ok") and cfg["provider"] != "hashed"
                    and e.get("embedders_used", {}).get("hashed"))
    save()
    if fell_back:
        print(f"\nWARNING: {fell_back} run(s) fell back to the hashed embedder although {cfg['provider']} was "
              "requested (throttling or an error). Their rankings and pairs are not comparable with the rest; "
              "re-run `run` before labelling or training.")

    build_label_sheets(results)
    print(f"\nSaved raw outputs to {RESULTS.relative_to(ROOT)}")
    print(f"Label picks in   {LABELS.relative_to(ROOT)}   (column: good = y/n)")
    print(f"Label themes in  {THEMES.relative_to(ROOT)}   (column: supported = y/n, error_type when n)")
    print(f"Label pairs in   {PAIRS.relative_to(ROOT)}   (column: good = y/n: would you watch and read these together?)")
    if stopped:
        print(f"\nSTOPPED before {stopped[0]} / {stopped[1]}: {stopped[2]}. Everything searched so far is cached "
              f"({os.environ['KNOTS_YOUTUBE_CACHE']}). After the quota resets (midnight Pacific time), run\n"
              "  python eval/run_eval.py run --resume\n"
              "and only the missing runs are done; cached searches cost nothing.")
    if incomplete:
        print(f"\n{incomplete} run(s) marked INCOMPLETE: catalog calls failed (network, rate limit or quota), "
              "so those sessions are missing picks. `run --resume` redoes only those; searches already answered "
              "are cached (arXiv and YouTube), so only the failures are retried.")
    if pending:
        print(f"{len(pending)} of {len(order)} run(s) are not complete yet. Do not label or score until this is 0.")
    if errors:
        print(f"\n{errors} run(s) failed; see FAIL lines above. Fix and re-run: labels you entered are kept.")
    return 0


def cmd_themes(args) -> int:
    """Extraction only: no catalogs, no YouTube, no embeddings of items. Writes themes_run.json and rebuilds
    themes.csv. results.json and labels.csv (picks) are left exactly as they were."""
    sessions = json.loads(SESSIONS.read_text())
    specs = [s.strip() for s in args.extractors.split(",") if s.strip()]
    entries, errors = [], 0
    for sess in sessions:
        for spec in specs:
            mode, model = parse_spec(spec)
            entry = {"session_id": sess["id"], "extractor": spec, "texts": sess["texts"]}
            try:
                set_extractor(mode, model)
                extraction._HOLDOUT = sess["id"]
                found, used = asyncio.run(extraction.extract(sess["texts"]))
                entry.update(ok=True, used=used, interests=[i.to_dict() for i in found])
                n = sum(len(i.implicit_themes) for i in found)
                dropped = sum(len(i.dropped_themes) for i in found)
                print(f"  {sess['id']}  {spec:<42} ok   {n} themes kept, {dropped} dropped")
            except Exception as e:  # noqa: BLE001
                errors += 1
                entry.update(ok=False, error=f"{type(e).__name__}: {e}")
                print(f"  {sess['id']}  {spec:<42} FAIL {type(e).__name__}: {str(e)[:90]}")
            entries.append(entry)
    THEMES_RUN.parent.mkdir(exist_ok=True)
    THEMES_RUN.write_text(json.dumps({"results": entries}, indent=2))
    build_label_sheets([], themes_only=True)
    print(f"\nSaved {THEMES_RUN.relative_to(ROOT)}; label themes in {THEMES.relative_to(ROOT)} "
          "(columns: supported = y/n; the quote it rests on is in `evidence`). Picks and their labels are untouched.")
    return 1 if errors else 0


def build_label_sheets(results: list[dict], themes_only: bool = False) -> None:
    old_labels = {(r["session_id"], r["extractor"], r["url"]): r for r in read_csv(LABELS)}
    old_themes = {(r["session_id"], r["extractor"], r["topic"], r["theme"]): r for r in read_csv(THEMES)}
    old_pairs = {(r["session_id"], r["extractor"], r["pair_id"]): r for r in read_csv(PAIRS)}
    splits = session_splits()

    pick_rows, theme_rows, pair_rows = [], [], []
    theme_source = [e for e in results if e.get("ok")]
    if THEMES_RUN.exists():                      # themes are extracted and labelled apart from picks (see cmd_themes)
        try:
            theme_source = json.loads(THEMES_RUN.read_text())["results"]
        except (OSError, ValueError, KeyError):
            pass
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
                    "good": prev.get("good", ""), "reason": prev.get("reason", ""),
                    "note": prev.get("note", ""),
                    "extractor": e["extractor"], "split": splits.get(e["session_id"], "train"),
                })
        for n, pr in enumerate(plan.get("pairs", []), start=1):
            pid = hashlib.sha1((pr["watch"]["url"] + "|" + pr["learn"]["url"]).encode()).hexdigest()[:10]
            prev = old_pairs.get((e["session_id"], e["extractor"], pid), {})
            pair_rows.append({
                "session_id": e["session_id"], "session_text": text, "pair_id": pid,
                "concept": pr["concept"], "watch_title": pr["watch"]["title"], "watch_url": pr["watch"]["url"],
                "learn_title": pr["learn"]["title"], "learn_url": pr["learn"]["url"],
                "pair_score": pr["score"], "bridge": pr["bridge"],
                "good": prev.get("good", ""), "note": prev.get("note", ""),
                "extractor": e["extractor"], "split": splits.get(e["session_id"], "train"),
            })
    for e in theme_source:
        text = " / ".join(e["texts"])
        for it in e["interests"]:
            for theme in it.get("implicit_themes", []):
                prev = old_themes.get((e["session_id"], e["extractor"], it["topic"], theme), {})
                theme_rows.append({
                    "session_id": e["session_id"], "session_text": text, "topic": it["topic"],
                    "theme": theme, "rationale": it.get("rationale", ""),
                    "evidence": (it.get("theme_evidence") or {}).get(theme, ""),
                    "rank": (it.get("theme_rank") or {}).get(theme, ""),
                    "supported": prev.get("supported", ""), "error_type": prev.get("error_type", ""),
                    "note": prev.get("note", ""),
                    "extractor": e["extractor"], "split": splits.get(e["session_id"], "train"),
                })

    # Shuffle within each session (fixed seed) so you can label without knowing which
    # extractor produced a row. Hide the last column (`extractor`) while labelling.
    for rows in (pick_rows, theme_rows, pair_rows):
        by_session = defaultdict(list)
        for r in rows:
            by_session[r["session_id"]].append(r)
        rows.clear()
        for sid in sorted(by_session):
            group = by_session[sid]
            random.Random(sid).shuffle(group)
            rows.extend(group)

    write_csv(THEMES, THEME_COLS, theme_rows)
    if not themes_only:                          # `themes` must never touch the pick or pair sheets
        write_csv(LABELS, LABEL_COLS, pick_rows)
        write_csv(PAIRS, PAIR_COLS, pair_rows)


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
        if not g:
            s[f"why_{(r.get('reason') or '').strip() or 'unspecified'}"] += 1
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
    lines += ["Why picks were not good (counts):", "",
              "| Extractor | Relevance | Quality | Repeat | Other | No reason given |",
              "| --- | --- | --- | --- | --- | --- |"]
    for x in extractors:
        s = stats[x]
        lines.append(f"| {x} | {s['why_relevance']} | {s['why_quality']} | {s['why_repeat']} | "
                     f"{s['why_other']} | {s['why_unspecified']} |")
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


# ---------------------------------------------------------------------------
# label: a small local web tool that writes your labels straight into the CSVs
# ---------------------------------------------------------------------------

LABEL_HTML = HERE / "label.html"
REASONS = ("", "relevance", "quality", "repeat", "other")   # why a pick was "not good"
# kind -> (csv path, columns, row key fields, label field)
SHEETS = {
    "pick": (LABELS, LABEL_COLS, ("session_id", "extractor", "url"), "good"),
    "theme": (THEMES, THEME_COLS, ("session_id", "extractor", "topic", "theme"), "supported"),
    "pair": (PAIRS, PAIR_COLS, ("session_id", "extractor", "pair_id"), "good"),
}


def row_id(kind: str, row: dict) -> str:
    _, _, key, _ = SHEETS[kind]
    return hashlib.sha1("\x1f".join(row.get(k, "") for k in key).encode()).hexdigest()[:12]


def cmd_label(args) -> int:
    import http.server
    import threading
    import webbrowser

    if not LABELS.exists() and not THEMES.exists() and not PAIRS.exists():
        print("No labels.csv / themes.csv yet. Run `python eval/run_eval.py run` first.")
        return 1

    lock = threading.Lock()
    port = args.port
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _guard(self) -> bool:
            # localhost only, and no cross-site writes
            if self.headers.get("Host", "") not in allowed_hosts:
                self._send(403, b"forbidden", "text/plain")
                return False
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] not in allowed_hosts:
                self._send(403, b"forbidden", "text/plain")
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            if self.path in ("/", "/label.html"):
                self._send(200, LABEL_HTML.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/data":
                out = {}
                with lock:
                    for kind, name in (("pick", "picks"), ("theme", "themes"), ("pair", "pairs")):
                        path, _, _, _ = SHEETS[kind]
                        rows = []
                        for r in read_csv(path):
                            r = dict(r)
                            r["id"] = row_id(kind, r)
                            r.pop("extractor", None)   # blind labelling: never sent to the page
                            rows.append(r)
                        out[name] = rows
                self._send(200, json.dumps(out).encode())
            elif self.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._guard():
                return
            if self.path != "/api/label" or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self._send(400, b"bad request", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                kind, rid = body["kind"], body["id"]
                value, note = str(body.get("value", "")), str(body.get("note", ""))[:500]
                reason = str(body.get("reason", ""))
                error_type = str(body.get("error_type", ""))
                if kind not in SHEETS or value not in ("", "y", "n"):
                    raise ValueError("bad kind or value")
                if reason not in REASONS or (reason and (kind != "pick" or value != "n")):
                    raise ValueError("bad reason")
                if error_type not in ERROR_TYPES or (error_type and (kind != "theme" or value != "n")):
                    raise ValueError("bad error_type")
            except Exception as e:  # noqa: BLE001
                self._send(400, f"bad request: {e}".encode(), "text/plain")
                return
            path, cols, _, label_col = SHEETS[kind]
            with lock:
                rows = read_csv(path)
                for r in rows:
                    if row_id(kind, r) == rid:
                        r[label_col], r["note"] = value, note
                        if kind == "pick":
                            r["reason"] = reason
                        if kind == "theme":
                            r["error_type"] = error_type
                        break
                else:
                    self._send(404, b"row not found (did you re-run the eval? reload the page)", "text/plain")
                    return
                tmp = path.with_suffix(".tmp")
                write_csv(tmp, cols, rows)
                tmp.replace(path)          # atomic: a crash can never leave a half-written CSV
            self._send(200, json.dumps({"ok": True}).encode())

    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        print(f"Cannot use port {port}: {e}. Try --port 8766.")
        return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"Labelling tool running at {url}\nLabels save straight into eval/labels.csv and eval/themes.csv.")
    print("Press Ctrl+C here when you are done.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Run `python eval/run_eval.py score` to see results.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run sessions through extractors, build label sheets")
    r.add_argument("--extractors", default="heuristic,bedrock",
                   help="comma list: heuristic, bedrock, or bedrock=<model id> (default: heuristic,bedrock)")
    r.add_argument("--no-videos", action="store_true",
                   help="skip YouTube (it is on by default when YOUTUBE_API_KEY is set)")
    r.add_argument("--resume", action="store_true",
                   help="keep runs that already completed in results.json; redo only INCOMPLETE, failed or missing ones")
    r.add_argument("--youtube-budget", type=int, default=None, metavar="N",
                   help="live YouTube searches allowed per quota day (tracked on disk; resets midnight Pacific). "
                        "Stops cleanly when spent instead of failing runs one by one. Also KNOTS_YOUTUBE_BUDGET")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("themes", help="re-extract themes only (no catalogs); rebuilds themes.csv, keeps pick labels")
    t.add_argument("--extractors", default="bedrock", help="comma list, as for `run` (default: bedrock)")
    t.set_defaults(fn=cmd_themes)
    lb = sub.add_parser("label", help="open the local labelling tool (saves into the CSVs)")
    lb.add_argument("--port", type=int, default=8765)
    lb.add_argument("--no-browser", action="store_true", help="do not open a browser tab automatically")
    lb.set_defaults(fn=cmd_label)
    s = sub.add_parser("score", help="turn your labels into a results table")
    s.add_argument("--include-mock", action="store_true", help="count offline mock picks too")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    # The eval labels every candidate, so neither the quality floor nor the model-written notes may change what it sees.
    os.environ["KNOTS_FLOOR"] = "off"
    os.environ["NOTES_LLM"] = "off"
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

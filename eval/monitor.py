"""Read the online metrics and say whether a retrain is due.

    python eval/monitor.py                       # the last 14 days of the event log
    python eval/monitor.py --days 30 --json
    python eval/monitor.py --registry data/models --freeze-baseline
                                                 # record what "normal" looks like now, as the baseline later
                                                 # windows are compared with (do this when a model is promoted)

Reads KNOTS_EVENT_LOG (default data/events.jsonl). It only reads and reports; it never retrains. The retrain
itself still has to pass the champion/challenger gate (knots/retrain.py)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from knots import feedback, online, retrain  # noqa: E402


def freeze_baseline(registry: retrain.Registry, events: list[dict]) -> dict:
    """Store a baseline in the registry state without changing which model is champion."""
    state = registry.state()
    state["baseline"] = online.make_baseline(events)
    registry._save(state)
    return state["baseline"]


def report(events: list[dict], state: dict, days: float, now: float | None = None) -> dict:
    labels = online.labels_from_events(events)
    decision = online.retrain_decision(labels, events, state, now=now, monitor=online.MonitorPolicy(window_days=days))
    decision["labels_total"] = len(labels)
    decision["usable_labels"] = len(retrain.eligible(labels, retrain.Policy()))
    return decision


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--registry", default=None, help="model registry directory (holds the baseline and training state)")
    ap.add_argument("--freeze-baseline", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    log = feedback.event_log()
    events = log.read()
    if not events:
        print(f"No events in {log.path} yet: nothing has been shown to listeners.")
        return 0
    registry = retrain.Registry(args.registry) if args.registry else None
    if args.freeze_baseline:
        if not registry:
            print("--freeze-baseline needs --registry")
            return 2
        recent = online.window(events, time.time(), args.days)
        freeze_baseline(registry, recent)
        print(f"baseline frozen from {len(recent)} events into {registry.state_path}")
        return 0
    state = registry.state() if registry else {"trained_on": 0, "baseline": None}
    out = report(events, state, args.days)
    if args.json:
        print(json.dumps(out, indent=1, default=str))
        return 0
    m = out["metrics"]
    fb = m["feedback"]
    print(f"last {args.days:g} days: {m['lists']} lists, {m['users']} people, {fb['n']} answers from {fb['people']} people"
          + (f", {fb['positive_rate']:.0%} good" if fb["positive_rate"] is not None else ""))
    print(f"empty {m['empty_rate'] if m['empty_rate'] is not None else 'n/a'}, relaxed {m['relaxed_rate'] if m['relaxed_rate'] is not None else 'n/a'}, "
          f"engagement {m['engagement_rate']}, reformulation {m['reformulation_rate']}, repeat {m['repeat_rate']}")
    print(f"usable labels for training: {out['usable_labels']} (of {out['labels_total']})")
    print("baseline:", "frozen" if state.get("baseline") else "none yet (alerts need one: --freeze-baseline)")
    for a in out["alerts"]:
        print(f"  ALERT {a['kind']}{'' if a['retrain_relevant'] else ' (not a retrain problem)'}: {a['detail']}")
    print(f"retrain: {'YES' if out['retrain'] else 'not yet'}, {out['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

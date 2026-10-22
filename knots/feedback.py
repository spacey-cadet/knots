"""
Recording what was shown and what listeners said about it, into the event log that `knots.online` reads.

Two calls:

  record_impression(plan, user, session_id)   after a response is built: which picks were shown, in what order
  record_feedback(user, session_id, url, good)   a listener said a pick was good or not

Feedback is only accepted for a pick that was actually shown, to the SAME verified person, in that session. That
stops anyone inventing labels for items they were never shown or for someone else's session; combined with the
per-person cap in `retrain.eligible`, one person cannot steer a retrain. Anonymous callers (no verified identity)
can be shown picks, and their impressions count toward the online metrics, but they cannot leave feedback,
because a label with no person behind it could not be capped.

Nothing here stores conversation text: an impression holds catalog facts and ids only.

  KNOTS_EVENT_LOG   path of the JSON-lines file (default data/events.jsonl). On EC2 put it on the persistent disk.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from .online import EventLog, impression_event


def log_path() -> str:
    return os.environ.get("KNOTS_EVENT_LOG") or str(Path(__file__).resolve().parent.parent / "data" / "events.jsonl")


_log: EventLog | None = None


def event_log() -> EventLog:
    global _log
    if _log is None or str(_log.path) != log_path():
        _log = EventLog(log_path())
    return _log


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


def _shown_cards(plan: dict) -> list[dict]:
    """The plan's full cards, in the order the response presents them (that is the order a listener hears)."""
    by_url = {c.get("url"): c for group in (plan.get("picks") or {}).values() for c in group}
    shown = ((plan.get("response") or {}).get("picks")) or []
    return [by_url[p["url"]] for p in shown if p.get("url") in by_url]


def record_impression(plan: dict, user: str | None, session_id: str, model: str = "") -> bool:
    """Log that this plan's picks were shown. Never raises: telemetry must not break an answer."""
    try:
        status = (plan.get("response") or {}).get("status", "empty")     # no interests: no response was built
        event_log().append(impression_event(user or "", session_id, _shown_cards(plan), status,
                                            model or plan.get("extractor", "")))
        return True
    except Exception:                                              # noqa: BLE001
        return False


def record_feedback(user: str | None, session_id: str, url: str, good: bool) -> dict:
    """Validate and log one listener verdict. Returns {"recorded": bool, "reason": str}."""
    if not user:
        return {"recorded": False, "reason": "no verified identity: feedback needs a signed-in listener"}
    if not session_id or not url:
        return {"recorded": False, "reason": "a session_id and the pick's url are required"}
    shown = [e for e in event_log().read()
             if e.get("type") == "impression" and e.get("session_id") == session_id and e.get("user") == user]
    if not any(p.get("url") == url for e in shown for p in e.get("picks", [])):
        return {"recorded": False, "reason": "that pick was not shown to you in that session"}
    event_log().append({"type": "feedback", "ts": time.time(), "user": user, "session_id": session_id, "url": url,
                        "good": "y" if good else "n", "source": "user"})
    return {"recorded": True, "reason": "thanks, noted"}


def forget_events(user: str | None) -> int:
    """Remove every logged event of one person (called from forget_me)."""
    if not user or not Path(log_path()).exists():
        return 0
    return event_log().forget(user)

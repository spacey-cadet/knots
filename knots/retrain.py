"""
When to retrain, from which labels, and whether the result may ship.

Only HUMAN ground truth trains: labels with source "user" (listeners who said good / not good) or "expert"
(the author's own labelling). Model output, notes and error_type never do. A retrain starts when enough such
labels have arrived from enough different people (so one enthusiast cannot steer it), with each person capped.
The challenger then has to beat the champion on a FROZEN expert holdout, with a margin and no regression on the
second metric, before it is promoted. Every promoted version is kept; `rollback` restores the previous one.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

TRAINABLE_SOURCES = ("user", "expert")


@dataclass
class Policy:
    min_labels: int = int(os.environ.get("KNOTS_RETRAIN_MIN_LABELS", "200"))      # N
    min_users: int = int(os.environ.get("KNOTS_RETRAIN_MIN_USERS", "20"))         # M
    per_user_cap: int = int(os.environ.get("KNOTS_RETRAIN_PER_USER_CAP", "25"))
    min_new: int = int(os.environ.get("KNOTS_RETRAIN_MIN_NEW", "100"))            # new eligible labels since last train
    margin: float = 0.02                                                          # headline metric must improve by this
    tolerance: float = 0.01                                                       # the other metric may not drop more


def eligible(labels: list[dict], policy: Policy) -> list[dict]:
    """Latest label per (user, session, url), human sources only, at most per_user_cap per user (newest kept)."""
    latest: dict[tuple, dict] = {}
    for r in sorted(labels, key=lambda r: r.get("ts", 0)):
        if r.get("source") in TRAINABLE_SOURCES and r.get("good") in ("y", "n"):
            latest[(r.get("user", ""), r.get("session_id", ""), r.get("url", ""))] = r
    per_user: dict[str, list[dict]] = defaultdict(list)
    for r in latest.values():
        per_user[r.get("user", "")].append(r)
    out = []
    for rows in per_user.values():
        out.extend(sorted(rows, key=lambda r: -r.get("ts", 0))[:policy.per_user_cap])
    return out


def should_retrain(labels: list[dict], state: dict, policy: Policy | None = None) -> tuple[bool, str]:
    policy = policy or Policy()
    rows = eligible(labels, policy)
    users = len({r.get("user", "") for r in rows})
    if len(rows) < policy.min_labels:
        return False, f"{len(rows)} usable labels, need {policy.min_labels}"
    if users < policy.min_users:
        return False, f"labels come from {users} people, need {policy.min_users}"
    new = len(rows) - int(state.get("trained_on", 0))
    if new < policy.min_new:
        return False, f"only {new} new usable labels since the last training, need {policy.min_new}"
    return True, f"{len(rows)} usable labels from {users} people ({new} new)"


def gate(champion: dict, challenger: dict, policy: Policy | None = None, headline: str = "p@3",
         second: str = "ndcg@5") -> dict:
    """Metrics are {name: float} measured on the SAME frozen expert holdout."""
    policy = policy or Policy()
    gain = challenger[headline] - champion[headline]
    drop = champion[second] - challenger[second]
    ok = gain >= policy.margin and drop <= policy.tolerance
    why = (f"{headline} {champion[headline]:.3f} -> {challenger[headline]:.3f} ({gain:+.3f}), "
           f"{second} {champion[second]:.3f} -> {challenger[second]:.3f}")
    if not ok:
        why += (f"; needs {headline} +{policy.margin} and {second} no worse than -{policy.tolerance}")
    return {"promote": ok, "reason": why}


class Registry:
    """Versioned model artifacts in a directory; `state.json` says which is the champion."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"

    def state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"champion": None, "history": [], "trained_on": 0}

    def _save(self, st: dict) -> None:
        self.state_path.write_text(json.dumps(st, indent=1))

    def register(self, artifact: dict, metrics: dict, trained_on: int) -> str:
        st = self.state()
        version = f"v{len(list(self.root.glob('v*.json'))) + 1}"
        (self.root / f"{version}.json").write_text(json.dumps(
            {"version": version, "created": time.time(), "metrics": metrics, "trained_on": trained_on,
             "artifact": artifact}, indent=1))
        return version

    def promote(self, version: str, trained_on: int) -> None:
        st = self.state()
        if st["champion"]:
            st["history"].append(st["champion"])
        st["champion"], st["trained_on"] = version, trained_on
        self._save(st)

    def rollback(self) -> str | None:
        st = self.state()
        if not st["history"]:
            return None
        st["champion"] = st["history"].pop()
        self._save(st)
        return st["champion"]

    def champion(self) -> dict | None:
        c = self.state()["champion"]
        return json.loads((self.root / f"{c}.json").read_text()) if c else None

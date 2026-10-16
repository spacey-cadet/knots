"""
The memory: what Knots keeps about a person between conversations.

For each conversation it stores ONLY derived data, never the words themselves:
  - the extracted interests (topic, concepts, unstated themes, keywords, mediums)
  - an embedding of the conversation, for finding relevant past conversations
  - a timestamp and a content key (so re-sending a conversation is not double counted)

The raw turns are discarded as soon as they have been extracted and embedded.
The person can see what is kept (`summary`) and erase it (`forget`).

Storage is SQLite (stdlib, zero setup) at KNOTS_PROFILE_DB, default
data/knots_profile.db. `ProfileStore` is the only thing the rest of the code
touches, so a DynamoDB version is a drop-in replacement for deployment (a
container's local disk does not survive restarts).
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from interests import Interest

MAX_RECORDS_PER_USER = 500          # bounded: the oldest are dropped past this

_INTEREST_FIELDS = {f.name for f in fields(Interest)}


@dataclass
class Record:
    key: str
    timestamp: str                  # ISO 8601
    interests: list[Interest]
    vector: list[float]
    embedder: str

    def time(self) -> datetime:
        try:
            dt = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob)) if blob else []


def _interest_from_dict(d: dict) -> Interest:
    return Interest(**{k: v for k, v in d.items() if k in _INTEREST_FIELDS})


def default_path() -> str:
    return os.environ.get("KNOTS_PROFILE_DB") or str(Path(__file__).resolve().parent / "data" / "knots_profile.db")


class ProfileStore:
    def __init__(self, path: str | None = None):
        self.path = path or default_path()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS records (
                user_id TEXT NOT NULL, key TEXT NOT NULL, ts TEXT NOT NULL,
                interests TEXT NOT NULL, vec BLOB, embedder TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, key))""")
            db.execute("CREATE INDEX IF NOT EXISTS by_user_time ON records (user_id, ts)")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)     # one short-lived connection per call: thread-safe

    def save(self, user_id: str, key: str, timestamp: str, interests: list[Interest],
             vector: list[float], embedder: str) -> bool:
        """Insert or replace one conversation's record. Returns True if it was new."""
        payload = json.dumps([i.to_dict() for i in interests])
        with self._db() as db:
            existed = db.execute("SELECT 1 FROM records WHERE user_id=? AND key=?", (user_id, key)).fetchone()
            db.execute("INSERT OR REPLACE INTO records (user_id, key, ts, interests, vec, embedder) VALUES (?,?,?,?,?,?)",
                       (user_id, key, timestamp, payload, _pack(vector), embedder))
            db.execute("""DELETE FROM records WHERE user_id=? AND key NOT IN
                          (SELECT key FROM records WHERE user_id=? ORDER BY ts DESC LIMIT ?)""",
                       (user_id, user_id, MAX_RECORDS_PER_USER))
        return existed is None

    def records(self, user_id: str, limit: int | None = None) -> list[Record]:
        """Newest first."""
        sql = "SELECT key, ts, interests, vec, embedder FROM records WHERE user_id=? ORDER BY ts DESC"
        args: tuple = (user_id,)
        if limit:
            sql += " LIMIT ?"
            args += (limit,)
        with self._db() as db:
            rows = db.execute(sql, args).fetchall()
        out = []
        for key, ts, interests, vec, embedder in rows:
            try:
                found = [_interest_from_dict(d) for d in json.loads(interests)]
            except (ValueError, TypeError):
                continue
            out.append(Record(key=key, timestamp=ts, interests=found, vector=_unpack(vec), embedder=embedder))
        return out

    def forget(self, user_id: str) -> int:
        with self._db() as db:
            cur = db.execute("DELETE FROM records WHERE user_id=?", (user_id,))
            return cur.rowcount

    def count(self, user_id: str) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM records WHERE user_id=?", (user_id,)).fetchone()[0]


_store: ProfileStore | None = None


def get_store() -> ProfileStore:
    global _store
    if _store is None or _store.path != default_path():
        _store = ProfileStore()
    return _store

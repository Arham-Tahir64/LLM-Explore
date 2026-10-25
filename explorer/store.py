"""Small SQLite store. Profiles and observations are namespaced by model ID."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

DEMO_PROFILE = {
    "id": "demo",
    "name": "Qwen3.5 · 4B",
    "provider": "demo",
    "model": "synthetic-qwen",
    "endpoint": "",
    "key_env": "",
}


class Store:
    def __init__(self, directory):
        self.path = Path(directory) / "explore.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, body TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')), body TEXT NOT NULL)"
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def profiles(self):
        with self.connect() as db:
            return [DEMO_PROFILE.copy()] + [
                json.loads(row[0])
                for row in db.execute("SELECT body FROM profiles ORDER BY rowid")
            ]

    def profile(self, identifier):
        return next((p for p in self.profiles() if p["id"] == identifier), None)

    def save_profile(self, body):
        body = dict(body, id=uuid4().hex[:12])
        with self.connect() as db:
            db.execute(
                "INSERT INTO profiles VALUES (?, ?)", (body["id"], json.dumps(body))
            )
        return body

    def experiments(self):
        with self.connect() as db:
            return [
                dict(json.loads(body), id=identifier, created=created)
                for identifier, created, body in db.execute(
                    "SELECT id,created,body FROM experiments ORDER BY created DESC, rowid DESC LIMIT 200"
                )
            ]

    def save_experiment(self, body):
        identifier = uuid4().hex[:12]
        with self.connect() as db:
            db.execute(
                "INSERT INTO experiments (id,body) VALUES (?,?)",
                (identifier, json.dumps(body)),
            )
        return identifier

    def delete_experiment(self, identifier):
        with self.connect() as db:
            return (
                db.execute("DELETE FROM experiments WHERE id=?", (identifier,)).rowcount
                > 0
            )

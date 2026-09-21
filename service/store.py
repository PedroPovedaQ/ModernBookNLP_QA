import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from service.config import Settings


class QueueFull(Exception):
    pass


class BusyJob(Exception):
    pass


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.data_dir / "jobs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, content_hash TEXT NOT NULL,
                    model_version TEXT NOT NULL, text TEXT, status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL,
                    updated REAL NOT NULL, result TEXT, error TEXT,
                    UNIQUE(owner, content_hash, model_version)
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created);
                CREATE TABLE IF NOT EXISTS worker (id INTEGER PRIMARY KEY CHECK(id=1), heartbeat REAL, ready INTEGER);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA secure_delete=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def submit(self, owner: str, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._cleanup(db, now)
            row = db.execute(
                "SELECT * FROM jobs WHERE owner=? AND content_hash=? AND model_version=?",
                (owner, digest, self.settings.analysis_version),
            ).fetchone()
            if row:
                return self.public(row)
            pending = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(owner=?),0) FROM jobs WHERE status IN ('queued','running')",
                (owner,),
            ).fetchone()
            if (
                pending[0] >= self.settings.max_pending
                or pending[1] >= self.settings.max_client_pending
            ):
                raise QueueFull()
            # Bound terminal-record/disk growth even when clients submit unique small jobs.
            if db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] >= 1000:
                raise QueueFull()
            job_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO jobs(id,owner,content_hash,model_version,text,status,created,updated) VALUES(?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    owner,
                    digest,
                    self.settings.analysis_version,
                    text,
                    "queued",
                    now,
                    now,
                ),
            )
            return self.public(
                db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            )

    def get(self, job_id: str, owner: str):
        with self.connect() as db:
            self._cleanup(db, time.time())
            row = db.execute(
                "SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner)
            ).fetchone()
            return self.public(row) if row else None

    @staticmethod
    def public(row):
        return {
            **{
                k: row[k]
                for k in (
                    "id",
                    "status",
                    "content_hash",
                    "model_version",
                    "attempts",
                    "created",
                    "updated",
                    "error",
                )
            },
            "result": json.loads(row["result"]) if row["result"] else None,
        }

    def delete(self, job_id: str, owner: str):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM jobs WHERE id=? AND owner=?", (job_id, owner)
            ).fetchone()
            if not row:
                return False
            if row["status"] == "running":
                raise BusyJob()
            db.execute("DELETE FROM jobs WHERE id=? AND owner=?", (job_id, owner))
            return True

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._cleanup(db, time.time())
            db.execute(
                "UPDATE jobs SET status='failed', text=NULL, error='model_version_changed', updated=? WHERE status='queued' AND model_version!=?",
                (time.time(), self.settings.analysis_version),
            )
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created,id LIMIT 1"
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, updated=?, error=NULL WHERE id=?",
                (time.time(), row["id"]),
            )
            return dict(
                db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
            )

    def recover(self):
        # Only call while holding the exclusive worker process lock.
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=CASE WHEN attempts < ? THEN 'queued' ELSE 'failed' END, error='worker_interrupted', updated=? WHERE status='running'",
                (self.settings.max_attempts, time.time()),
            )
            db.execute("UPDATE jobs SET text=NULL WHERE status='failed'")

    def finish(self, job_id: str, result: dict):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='completed', result=?, text=NULL, error=NULL, updated=? WHERE id=? AND status='running'",
                (json.dumps(result, ensure_ascii=True), time.time(), job_id),
            )

    def fail(self, job_id: str, error: str):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=CASE WHEN attempts < ? THEN 'queued' ELSE 'failed' END, error=?, updated=? WHERE id=? AND status='running'",
                (self.settings.max_attempts, error, time.time(), job_id),
            )
            db.execute(
                "UPDATE jobs SET text=NULL WHERE id=? AND status='failed'", (job_id,)
            )

    def heartbeat(self, ready: bool):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO worker VALUES(1,?,?)", (time.time(), int(ready))
            )

    def ready(self):
        with self.connect() as db:
            row = db.execute("SELECT heartbeat,ready FROM worker WHERE id=1").fetchone()
            return bool(row and row["ready"] and time.time() - row["heartbeat"] < 15)

    def _cleanup(self, db, now):
        db.execute(
            "DELETE FROM jobs WHERE status IN ('completed','failed') AND updated < ?",
            (now - self.settings.retention_seconds,),
        )
        db.execute(
            "UPDATE jobs SET status='failed', text=NULL, error='queue_expired', updated=? WHERE status='queued' AND created < ?",
            (now, now - self.settings.retention_seconds),
        )

    def cleanup(self):
        with self.connect() as db:
            self._cleanup(db, time.time())
        with self.connect() as db:
            db.execute("PRAGMA wal_checkpoint(PASSIVE)")

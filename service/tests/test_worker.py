import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path

from service.config import Settings
from service.model import normalize_output
from service.store import Store
from service.worker import ModelProcess, worker_lock


def sleeping_child(connection, settings, parent_pid):
    connection.send(("ready", None))
    time.sleep(60)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.settings = Settings(data_dir=self.path)
        self.store = Store(self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_tampered_cached_weights_before_loading(self):
        from unittest.mock import patch

        from service.model import verified_weights

        (self.path / "model.bin").write_bytes(b"tampered")
        with patch("service.model.WEIGHTS", {"model.bin": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verified_weights(self.path)

    def test_exclusive_lock(self):
        with worker_lock(self.path / "worker.lock"):
            with self.assertRaises(BlockingIOError):
                with worker_lock(self.path / "worker.lock"):
                    pass

    def test_timeout_kills_child(self):
        model = ModelProcess(self.settings, target=sleeping_child)
        try:
            self.assertEqual(model.wait(15, self.store, threading.Event())[0], "ready")
            with self.assertRaises(TimeoutError):
                model.wait(0.1, self.store, threading.Event(), ready=True)
        finally:
            model.close()
        self.assertFalse(model.process.is_alive())

    def test_cleanup_and_readiness(self):
        self.assertFalse(self.store.ready())
        self.store.heartbeat(True)
        self.assertTrue(self.store.ready())
        job = self.store.submit("alice", "hello")["id"]
        self.store.claim()
        self.store.finish(job, {})
        with self.store.connect() as db:
            db.execute("UPDATE jobs SET updated=0")
        self.store.cleanup()
        self.assertIsNone(self.store.get(job, "alice"))

    def test_changed_settings_do_not_reuse_cache_or_mislabel_pending_jobs(self):
        from dataclasses import replace

        old_job = self.store.submit("alice", "hello")["id"]
        changed = Store(replace(self.settings, batch_size=1))
        new_job = changed.submit("alice", "hello")["id"]
        self.assertNotEqual(old_job, new_job)
        self.assertEqual(changed.claim()["id"], new_job)
        self.assertEqual(
            changed.get(old_job, "alice")["error"], "model_version_changed"
        )

    def test_exact_unicode_offsets(self):
        text = "😀 Alice said “Hi”."
        tokens = [
            ("😀", 0, 1),
            ("Alice", 2, 7),
            ("said", 8, 12),
            ("“", 13, 14),
            ("Hi", 14, 16),
            ("”", 16, 17),
            (".", 17, 18),
        ]
        with (self.path / "document.tokens").open("w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["word", "byte_onset", "byte_offset"])
            writer.writerows(tokens)
        (self.path / "document.entities").write_text(
            "COREF\tstart_token\tend_token\tprop\tcat\ttext\n1\t1\t1\tPROP\tPER\tAlice\n"
        )
        (self.path / "document.quotes").write_text(
            "quote_start\tquote_end\tmention_start\tmention_end\tmention_phrase\tchar_id\tquote\n3\t5\t1\t1\tAlice\t1\tunused\n"
        )
        result = normalize_output(text, self.path)
        quote = result["quotes"][0]
        self.assertEqual(text[quote["start"] : quote["end"]], "“Hi”")
        self.assertEqual((quote["start_utf16"], quote["end_utf16"]), (14, 18))
        self.assertEqual(result["characters"][0]["display_name"], "Alice")
        with self.assertRaises(ValueError):
            normalize_output(text.replace("Alice", "Bobby"), self.path)

    def test_polling_does_not_wait_for_a_queue_writer(self):
        from concurrent.futures import ThreadPoolExecutor

        job = self.store.submit("alice", "hello")["id"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.store.connect() as writer:
                writer.execute("BEGIN IMMEDIATE")
                future = pool.submit(self.store.get, job, "alice")
                try:
                    result = future.result(timeout=1)
                finally:
                    writer.rollback()
            self.assertEqual(result["status"], "queued")

    def test_concurrent_deduplication(self):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(
                pool.map(lambda _: self.store.submit("alice", "same")["id"], range(16))
            )
        self.assertEqual(len(set(ids)), 1)


if __name__ == "__main__":
    unittest.main()

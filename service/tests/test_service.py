import hashlib
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from service.api import create_app
from service.config import Settings
from service.store import Store

KEY = "test-key-" + "a" * 32
OTHER = "test-key-" + "b" * 32


class ServiceFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            data_dir=Path(self.temp.name),
            keys={
                hashlib.sha256(KEY.encode()).hexdigest(): "alice",
                hashlib.sha256(OTHER.encode()).hexdigest(): "bob",
            },
            max_chars=100,
            max_pending=2,
        )
        self.store = Store(self.settings)
        self.client = TestClient(create_app(self.settings))
        self.headers = {"Authorization": "Bearer " + KEY}

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def submit(self, text="“Hello,” said Alice.", headers=None):
        return self.client.post(
            "/v1/jobs", json={"text": text}, headers=headers or self.headers
        )


class ServiceTests(ServiceFixture, unittest.TestCase):
    def test_auth_and_ownership(self):
        self.assertEqual(
            self.client.post("/v1/jobs", json={"text": "hello"}).status_code, 401
        )
        job = self.submit().json()["data"]["id"]
        other = {"Authorization": "Bearer " + OTHER}
        self.assertEqual(
            self.client.get("/v1/jobs/" + job, headers=other).status_code, 404
        )
        self.assertEqual(
            self.client.delete("/v1/jobs/" + job, headers=other).status_code, 404
        )

    def test_dedup_and_persistence(self):
        first = self.submit().json()["data"]["id"]
        self.assertEqual(self.submit().json()["data"]["id"], first)
        self.assertEqual(Store(self.settings).get(first, "alice")["status"], "queued")
        self.assertNotEqual(
            self.submit(headers={"Authorization": "Bearer " + OTHER}).json()["data"][
                "id"
            ],
            first,
        )

    def test_limits(self):
        self.assertEqual(self.submit("x" * 101).status_code, 413)
        self.assertEqual(self.submit("   ").status_code, 422)
        self.assertEqual(self.submit("one").status_code, 202)
        self.assertEqual(self.submit("two").status_code, 202)
        self.assertEqual(self.submit("three").status_code, 429)

    def test_claim_recovery_and_attempt_limit(self):
        job = self.submit().json()["data"]["id"]
        first = self.store.claim()
        self.assertEqual(first["id"], job)
        self.assertIsNone(self.store.claim())
        self.store.recover()
        second = self.store.claim()
        self.assertEqual(second["attempts"], 2)
        self.store.recover()
        self.assertEqual(self.store.get(job, "alice")["status"], "failed")
        self.assertIsNone(self.store.claim())

    def test_completed_cache_and_delete(self):
        job = self.submit().json()["data"]["id"]
        self.store.claim()
        self.assertEqual(
            self.client.delete("/v1/jobs/" + job, headers=self.headers).status_code, 409
        )
        self.store.finish(job, {"quotes": [], "characters": []})
        result = self.client.get("/v1/jobs/" + job, headers=self.headers).json()["data"]
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("text", result)
        self.assertEqual(self.submit().json()["data"]["id"], job)
        self.assertEqual(
            self.client.delete("/v1/jobs/" + job, headers=self.headers).status_code, 204
        )
        self.assertEqual(
            self.client.get("/v1/jobs/" + job, headers=self.headers).status_code, 404
        )


if __name__ == "__main__":
    unittest.main()

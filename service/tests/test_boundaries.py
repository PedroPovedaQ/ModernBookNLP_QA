import asyncio
import json
import unittest

from test_service import ServiceFixture

from service.api import BodyLimit


class BoundaryTests(ServiceFixture, unittest.TestCase):
    def test_malformed_unicode_and_no_input_echo(self):
        for text in ["\ud800", "a\x00b"]:
            response = self.client.post(
                "/v1/jobs",
                content=json.dumps({"text": text}),
                headers={**self.headers, "Content-Type": "application/json"},
            )
            self.assertEqual(response.status_code, 422)
            self.assertNotIn(text, response.text)
        response = self.client.post(
            "/v1/jobs",
            json={"text": {"private": "secret passage"}},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("secret passage", response.text)

    def test_authenticated_schema(self):
        self.assertEqual(self.client.get("/v1/openapi.json").status_code, 401)
        self.assertIn(
            "/v1/jobs",
            self.client.get("/v1/openapi.json", headers=self.headers).json()["paths"],
        )

    def test_queue_expiration_and_input_erasure(self):
        job = self.submit().json()["data"]["id"]
        with self.store.connect() as db:
            db.execute("UPDATE jobs SET created=0 WHERE id=?", (job,))
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.store.get(job, "alice")["error"], "queue_expired")
        with self.store.connect() as db:
            self.assertIsNone(
                db.execute("SELECT text FROM jobs WHERE id=?", (job,)).fetchone()[0]
            )

    def test_failed_job_can_be_explicitly_deleted_then_retried(self):
        job = self.submit().json()["data"]["id"]
        for _ in range(self.settings.max_attempts):
            self.store.claim()
            self.store.fail(job, "analysis_failed_or_timed_out")
        self.assertEqual(self.store.get(job, "alice")["status"], "failed")
        self.assertEqual(self.submit().json()["data"]["id"], job)
        self.client.delete("/v1/jobs/" + job, headers=self.headers)
        self.assertNotEqual(self.submit().json()["data"]["id"], job)

    def test_client_quota_and_stale_readiness(self):
        from dataclasses import replace

        from fastapi.testclient import TestClient

        from service.api import create_app

        with TestClient(
            create_app(replace(self.settings, max_pending=10, max_client_pending=1))
        ) as client:
            self.assertEqual(
                client.post(
                    "/v1/jobs", json={"text": "one"}, headers=self.headers
                ).status_code,
                202,
            )
            self.assertEqual(
                client.post(
                    "/v1/jobs", json={"text": "two"}, headers=self.headers
                ).status_code,
                429,
            )
        self.store.heartbeat(True)
        with self.store.connect() as db:
            db.execute("UPDATE worker SET heartbeat=0")
        self.assertEqual(self.client.get("/readyz").status_code, 503)


class StreamingLimitTests(unittest.TestCase):
    def test_chunked_body_limit_precedes_application(self):
        async def scenario():
            called = []
            sent = []
            messages = iter(
                [
                    {"type": "http.request", "body": b"123", "more_body": True},
                    {"type": "http.request", "body": b"456", "more_body": False},
                ]
            )

            async def app(scope, receive, send):
                called.append(True)

            async def receive():
                return next(messages)

            async def send(message):
                sent.append(message)

            await BodyLimit(app, 5)({"type": "http"}, receive, send)
            self.assertEqual(called, [])
            self.assertEqual(sent[0]["status"], 413)

        asyncio.run(scenario())

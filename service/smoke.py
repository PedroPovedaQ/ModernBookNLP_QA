"""Exercise a real HTTP job. Reads a key file without printing credentials."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    text = 'Alice walked into the garden. "Hello, Bob," said Alice.\r\nBob smiled. "Good morning, Alice," said Bob.'

    def request(path, body=None):
        req = urllib.request.Request(
            args.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    start = time.monotonic()
    job = request("/v1/jobs", {"text": text})["data"]
    print("Submitted", job["id"], flush=True)
    while time.monotonic() - start < 1000:
        job = request("/v1/jobs/" + job["id"])["data"]
        if job["status"] == "failed":
            raise RuntimeError(job["error"])
        if job["status"] == "completed":
            assert len(job["result"]["quotes"]) == 2, job
            for quote in job["result"]["quotes"]:
                assert text[quote["start"] : quote["end"]] == quote["text"]
            assert request("/v1/jobs", {"text": text})["data"]["id"] == job["id"]
            args.output.write_text(
                json.dumps(
                    {"elapsed_seconds": time.monotonic() - start, "job": job}, indent=2
                )
            )
            print("Completed and cached", job["id"], flush=True)
            return
        time.sleep(3)
    raise TimeoutError("Smoke deadline exceeded")


if __name__ == "__main__":
    main()

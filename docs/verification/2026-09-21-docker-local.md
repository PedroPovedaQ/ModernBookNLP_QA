# Local Docker verification — 2026-09-21

## Verdict

PASS for bounded Linux ARM64 Docker inference and graceful restart persistence. No public deployment was performed. This supersedes the earlier unverified container result in `self-hosted-api-review.md`.

Reused the unchanged Dockerfile, Compose configuration and previously built image `sha256:3c2a834d85aef8e511bd181da6743c69860a1b3f5ee89c8e3750e0ed2ceaf532`. Production modules match reviewed commit `a9581352163256eb6a38f3320712641a89db97df`; no service code changed. The existing 19-test and code-review receipts therefore remain applicable. No new unit tests were needed for this runtime verification/documentation follow-up. Execution was inline under the session tool mapping.

## Observed results

- Both local and HTTPS Compose configurations pass `docker compose config --quiet`.
- Real authenticated HTTP smoke completed with Alice/Bob attribution, two exact quotes, CRLF-preserving offsets and content deduplication.
- Model initialization: 204.9 seconds, including remaining base-model cache population. Inference: 28.208 seconds for the 28-word fixture.
- Graceful Docker restart retained the completed job and identical result. Missing credentials returned HTTP 401.
- A different, freshly submitted 25-word emoji/CRLF fixture completed after restart. Both expected speaker names, code-point spans and UTF-16 slices matched the original source.
- Cached restart initialization: 104.4 seconds. Fresh inference: 17.631 seconds. `/readyz` returned 503 while loading and 200 once ready.
- Maximum observed cgroup memory across these runs: 5,164,863,488 bytes (4.81 GiB), including file cache. Final anonymous memory approximately 3.45 GiB; file cache approximately 1.31 GiB. No OOM events or kills. This is not a minimum-RAM measurement or a full-book bound.
- API left running at `http://127.0.0.1:8787`; model and jobs persist in `modernbooknlp_booknlp-data`.

## Evidence

Local artifacts are in `/Users/pedro.poveda/.codex/artifacts/ensemble-benchmark-20260920/`:

- `service-docker-smoke-20260921.json`
- `service-docker-restart-20260921.json`
- `service-docker-runtime-20260921.log`
- `service-container-validation-status.json`
- `docker-restart-check.py` (HTTP verification script; reads an external key file)

## Deployment boundary

An 8 GiB CPU VPS is the preferred first paid pilot; a 4 GiB host has little anonymous-memory headroom and has not been validated. This Mac had heavy memory pressure, so timings are observations rather than VPS predictions. Only ARM64 is verified; build and repeat the smoke on the target architecture. Full chapters, input-boundary load, abrupt host loss, backups/restores and sustained concurrency remain unmeasured. The test establishes graceful restart persistence, not zero-downtime or high availability.

Use one Docker Compose instance, persistent local volume and the existing Caddy HTTPS overlay. Verify external readiness and a fresh authenticated job before connecting the application. Periodic inference probes must submit distinct text or delete their previous completed job so a cache hit cannot masquerade as fresh inference.

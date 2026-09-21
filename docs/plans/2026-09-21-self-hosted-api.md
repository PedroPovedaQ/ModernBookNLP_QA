---
title: Self-hosted quotation attribution API
date: 2026-09-21
type: feat
---
# Goal

Expose the existing ModernBookNLP Joint model as an authenticated, asynchronous, always-on service. User approved building the discussed service and researching hosting. Actual paid-host selection remains dependent on account/budget information.

## Product and technical contract

- POST /v1/jobs accepts original English prose and returns a job ID immediately. GET retrieves state/results; DELETE removes queued or terminal jobs. No caller-supplied paths or URLs.
- API keys are mapped from SHA-256 hashes to client identities; ownership is checked on all job operations. Backend-to-backend usage only; mobile apps must not embed these keys.
- Persistent SQLite queue on one local volume, separate API and worker processes. One worker exclusively locks the volume. Atomic claims, bounded attempts, recovery after restart, enforced inference timeout, admission limits and retention cleanup.
- Model loaded once in a supervised child process. Worker supervisor remains responsive to heartbeats and shutdown; timed-out child is terminated before another job starts.
- Content cache is scoped to client, text hash and model/pipeline version. Source text is erased after successful completion; terminal records expire. Errors do not expose text or filesystem paths.
- Results contain exact character/UTF-16 half-open quote and speaker-mention spans, model character IDs, display names and aliases. IDs are scoped to a result, not guaranteed stable across chapters/reanalysis. Voice selection stays with the consuming app. No false confidence score.
- Defaults constrain prose size and concurrency based on excerpt evidence. Full-book reconciliation, app playback/UI changes, automatic migration of existing voice assignments and multi-host scaling are outside this API increment.

## Implementation units

### U1. Durable API and queue
Files: service/config.py, service/store.py, service/api.py, service/tests/.
Verify auth, ownership, size/queue limits, concurrent deduplication, retry bounds, expiry and restart persistence. Start with failing tests for the new contract.

### U2. Model adapter and supervised worker
Files: service/model.py, service/worker.py, service/tests/.
Verify exact Unicode offsets using existing output characterization, scratch isolation, worker exclusivity, timeout termination, crash recovery and a real authenticated HTTP/model round trip. Keep pretrained inference unchanged.

### U3. Deployment and operating contract
Files: Dockerfile, compose.yaml, service requirements, docs/service.md, deploy/.
Provide a CPU container, persistent volume, health/readiness probes, HTTPS reverse-proxy example, secret provisioning and backup/recovery instructions. Build and test locally; research current always-on hosts with primary sources. Record actual local/container/live-host evidence separately. Commit/publish service-only changes to the user's fork after review; never include private benchmark app snapshots.

## Review and verification

Planning review (inline under session tool mapping): concurrency requires a process lock and atomic SQL, timeout requires killing inference rather than timing out an HTTP handler, ownership must derive from credentials, and cached IDs must not be sold as cross-chapter identities. These corrections are included above. Confidence: high for bounded single-host API; whole-book resource sizing and chosen hosting budget remain unresolved.

Definition of done: focused tests pass, real inference succeeds through HTTP, deployment artifacts build or an exact infrastructure failure is recorded, source reviewed, hosting recommendation grounded in current pricing and limits. An always-on public deployment is only complete after the chosen host is provisioned and independently health-checked; local success does not satisfy that claim.

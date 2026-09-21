# Self-hosted API review

Reviewed the new service against `origin/main` and the self-hosted API plan. The session's explicit tool mapping requires review work sequentially in the main thread; the CE correctness, security, reliability, API-contract, testing, maintainability, performance and adversarial lenses were applied that way. This is not an independent multi-agent or cross-model review.

## Actionable findings

No unresolved code findings from this pass. Resolved during implementation/review:

- Job polling performed retention writes and therefore waited behind queue writers. A concurrent regression reproduced the timeout; polling now uses a pure WAL read and filters expired terminal results. Cleanup remains in the worker and submission path.
- SQLite checkpointing inside the cleanup transaction failed with a locked-table error. Checkpoint now uses a separate connection after commit; expiry/readiness regression passes.
- Reading source with universal newline conversion would corrupt CRLF source offsets. Preserve UTF-8 source newlines; the real-model HTTP smoke validated exact quote spans after CRLF.
- Changing analysis settings could reuse results or process pending jobs under misleading metadata. Cache identity now includes thread/batch settings and incompatible queued jobs fail explicitly. Regression verifies distinct cache IDs and the old-job error.
- Removed duplicate inherited test execution so reported counts represent distinct tests. Malformed-Unicode tests now send escaped wire JSON instead of failing in the test client's encoder before reaching the API.

## Coverage

Authentication derives client identity from constant-time hash comparison; every job read/delete checks that identity. Bound request bodies, input validation, pending quotas, retained-record cap and private error messages were checked. No caller-controlled file paths, URL fetching or shell execution are exposed.

Queue transactions were checked against simultaneous deduplication, claim/delete races, crash recovery and maximum attempts. Exclusive worker lock, model child termination, parent-exit handling, cleanup, readiness and service supervisor behavior were inspected. A timed-out child was actually terminated by a process-level test.

Model outputs use asserted source spans and explicit UTF-16/code-point units. Display names do not imply stable identity across chapters. Weight checksums precede deserialization; a tampered-cache regression rejects mismatches. Tests also cover missing quotes, Unicode boundaries, queued expiry, failed-job resubmission and client separation. Dependency-light tests run in a fresh environment so the CI recipe cannot accidentally rely on the benchmark's ML environment.

Simplification pass: shared test setup now avoids duplicate inherited tests; unused imports removed and formatting standardized. Kept separate API, store, adapter and process supervisor because each owns a distinct failure boundary. No app UI or playback state is modified.

## Verdict and limits

Code is ready for bounded single-host testing/deployment, subject to the recorded container smoke and chosen-host verification. This does not certify whole-book performance, high availability, or consistent cross-chapter voices. Initial Hugging Face base-model downloads still follow upstream default revisions; preserve the cache and pin snapshots before treating runs as fully hermetic. API credentials represent backend clients; a consuming app must separately enforce end-user document ownership.

Public hosting remains unprovisioned until account, budget and domain/endpoint are selected. An authenticated local smoke or container build is not proof of a remotely available service.

## Container follow-up

Nineteen focused tests pass, including a regression that first reproduced polling blocked behind a queue writer. The updated Linux ARM64 CPU image builds. First uncached container startup exhausted the former 900-second initialization deadline; startup now allows 1800 seconds and logs elapsed initialization and failure type. This timeout increase is not a performance claim. Real container inference and restart smoke remain pending until their artifacts are recorded. Native real-model HTTP checks already passed.

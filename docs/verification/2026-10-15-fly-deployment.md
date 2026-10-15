# Fly pilot deployment verification — 2026-10-15

## Verdict

**PASS for the bounded single-machine pilot.** The ModernBookNLP service is serving
HTTPS from Fly in `iad` with one AMD64 machine and one durable volume. Liveness,
readiness, authentication, fresh inference, request deduplication, non-root volume
writes, and graceful restart persistence were verified against the live endpoint.

## Deployed artifact

- Source repository: `PedroPovedaQ/ModernBookNLP_QA`
- Source commit: `e0322db5ab2bd280856b94bed17649f4345cfa85`
- Fly app: `speedreadify-booknlp`
- Image tag: `registry.fly.io/speedreadify-booknlp:e0322db5ab2bd280856b94bed17649f4345cfa85`
- Image manifest: `sha256:45b03704997b9c043409d6a52aa7a831e1ade184fa5e67936a137b8bc74e84d2`
- Build path: Fly remote builder with `--build-only --remote-only`; no local Docker build
- Release: Fly release `v1`
- Endpoint: `https://speedreadify-booknlp.fly.dev`

The image tag is pinned to the reviewed source commit. The AMD64 runtime was checked
inside the running machine (`uname -m` returned `x86_64`), so the deployment was
verified on the target architecture rather than by health checks alone.

## Runtime shape and persistence

The app has exactly one running Machine and one attached volume:

- Machine: `879640a0019398` (`crimson-meadow-8679`), `iad`, started, checks `1/1`
- VM size: `shared-cpu-4x`, 4 shared vCPUs, 8192 MB RAM
- Volume: `vol_40ojdwgy5lm851p4`, `booknlp_data`, `iad`, encrypted, 30 GB
- Mount: `/data`
- Runtime user: `booknlp`, UID `10001`
- Fly configuration: [`fly.toml`](../../fly.toml)

The running process is non-root. As UID 10001, it created and removed a marker under
`/data`; the resulting file ownership was `10001:10001` with mode `0644`. This proves
the provider-mounted volume is writable by the service user and covers the jobs,
SQLite, model, and Hugging Face cache paths.

`auto_stop_machines = "off"`, `auto_start_machines = false`, and the absence of an
HA/scaling stanza keep the pilot to this single VM. No dedicated IPv4, managed
database, GPU, or additional runtime replica was provisioned. The remote build did
not leave an additional Fly builder app in the account.

## Credentials

The existing local server configuration was reused only after validating its single
client mapping. The service-side `BOOKNLP_API_KEY_HASHES` value was delivered through
Fly secrets. The raw client key remains outside the repository in a local file with
mode `0600`; it was not printed, committed, included in image configuration, or put
in this receipt. Clients must use HTTPS Bearer authentication.

## Live probes and inference

The public probes returned the expected responses after the model loaded:

- `GET /healthz` → HTTP `200`, `{"status":"alive"}`
- `GET /readyz` → HTTP `200`, `{"status":"ready"}`
- An unauthenticated `GET /v1/jobs/nonexistent` → HTTP `401`, `{"detail":"Invalid credentials"}`

The initial cold load used CPU and logged the model ready after 154.6 seconds; the
external readiness probe returned `200` by `23:11:24Z`. The existing authenticated
smoke request completed as job `5da778cf0bf54226896829d93f604f5d` with one attempt,
two quotes, and model version
`modern-joint-84f1f34-adapter-v1-threads4-batch2`. Its two quote spans matched the
submitted synthetic text in both code-point and UTF-16 coordinates. Repeating the
same POST returned the same job ID, proving request deduplication.

After the restart described below, a distinct synthetic English request completed as
job `c15d99941a994a7ebf05bc973e671214` in 3.500 seconds with one attempt, two quotes,
and aliases `Carol` and `Dana`. Both returned quote spans matched their source slices
in code-point and UTF-16 coordinates. Repeating that POST also returned the same job
ID. This is fresh post-restart inference, not a health-only check.

## Graceful restart and job persistence

The same Machine was restarted with a graceful `SIGINT` and a 40-second timeout.
Uvicorn shut down normally, the main child exited with code `0`, and Fly recorded
`requested_stop=true` with `oom_killed=false`. The same volume was reattached to the
same Machine ID. The model reloaded from the mounted cache and logged ready after
158.6 seconds; external `/readyz` returned `200` by `23:14:57Z`.

The pre-restart completed job `5da778cf0bf54226896829d93f604f5d` remained available
after restart with status `completed`, one attempt, and two quotes. Its canonical
result SHA-256 remained
`71c18dc6217ae18e787e8b3da76035c6ef99da46e888ff3796f887eb0e55d2b9`, identical to
the pre-restart value. The post-restart deduplication and fresh inference checks then
completed successfully.

## Cost and pilot limits

At the current Fly pricing basis, the always-on shape is approximately **$55.28 per
month before egress and snapshot charges**: shared-cpu-4x base allocation `$8.78`,
seven additional GiB of RAM at `$6/GiB` (`$42.00`), and a 30 GB volume at
`$0.15/GB` (`$4.50`). This is reported as the approved rough **$55/month** pilot
estimate on Fly's published 730-hour monthly basis. Five retained snapshots are
configured; storage or snapshot usage beyond the included allowance can change the
bill.

The service still has its existing 50,000-character request cap and analysis-scoped
IDs. Full-book memory pressure, sustained throughput/concurrency, abrupt host-loss
recovery, backup restoration, and HA behavior were not measured. This is a single
instance pilot and has not been integrated into SpeedReadify.

The configuration review found zero actionable findings; see
[`2026-10-15-fly-config-review.md`](2026-10-15-fly-config-review.md).

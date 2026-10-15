# Fly pilot configuration review — 2026-10-15

## Scope

- Reviewed only `/Users/pedro.poveda/.codex/artifacts/ensemble-benchmark-20260920/upstream/fly.toml`.
- Compared it with the previously reviewed `Dockerfile`, `service/serve.py`, and `service/config.py` in the same upstream checkout.
- No production files, Dockerfile, service code, or Fly state were changed. No Fly command, deployment, build, or test was run.

## Review

The configuration is compatible with the existing single-host service. `internal_port = 8000` matches the container command and `PORT` default; `force_https = true` provides the external HTTPS boundary; and the `SIGTERM`/`kill_timeout = 40` settings give `tini` and `service.serve` time to stop Uvicorn, the worker, and the model child. The container remains non-root (`booknlp`, UID 10001), and its `/data` mount covers the configured jobs, model, and Hugging Face cache directories (`fly.toml:5-14`; `Dockerfile:10-16`; `service/serve.py:17-53`; `service/config.py:45-74`).

The pilot shape is explicitly one shared 4-CPU/8-GB machine in `iad` with one 30-GB volume and five retained snapshots. There is no HA or scaling stanza, `auto_stop_machines = "off"` keeps the pilot running, and `auto_start_machines = false` avoids an automatic scale-from-zero path. `restart.policy = "on-failure"` matches the supervisor’s non-zero exit when either API or worker dies (`fly.toml:1-3,16-34`; `service/serve.py:39-53`). This keeps the cost and failure blast radius bounded to the approved single machine and volume.

The Fly check intentionally probes `/healthz`, which is liveness only. The service’s `/readyz` remains a separate readiness signal based on the worker heartbeat, so a healthy machine can still be loading the model; this matches the documented queue-during-startup behavior (`fly.toml:18-23`; `service/api.py:81-89`). The deployment owner must verify external `/readyz` and one authenticated job before treating the endpoint as inference-ready.

No credential is embedded in `fly.toml`. The service still fails startup without `BOOKNLP_API_KEY_HASHES`, so the deployment must inject that value through Fly secrets or equivalent provider secret storage (`service/api.py:48-51`; `service/config.py:51-75`). The `/v1/` routes remain authenticated while `/healthz` and `/readyz` are intentionally public probes.

The durable-volume design is coherent: SQLite and the worker lock live under `BOOKNLP_DATA_DIR=/data/jobs`, while model and Hugging Face caches also live under `/data`. The worker lock protects against two workers sharing a volume, and the one-machine configuration avoids concurrent volume attachment. Fly volume mount ownership is provider/runtime behavior; the deployment owner must record a real startup write proof as part of deployment verification rather than infer it from the image’s build-time `/data` ownership (`Dockerfile:10-12`; `service/worker.py:97-103`).

**Finding: none.** No actionable compatibility, single-instance, persistence, non-root startup, health/readiness, credential, restart, or cost-blast-radius issue was found in `fly.toml`.

## Deployment handoff limits

This is a configuration review only. Before calling the pilot deployed, verify the actual Fly machine has exactly one `shared-cpu-4x`/8-GB VM and one 30-GB `iad` volume, prove UID 10001 can create the jobs/cache directories on the mounted volume, check `/healthz` and `/readyz` separately, submit one authenticated job, restart the machine, and confirm the job/model-cache persistence. Keep the provider secret out of source, logs, command arguments, and chat.

## Verdict

**Zero actionable findings.** The reviewed Fly configuration is suitable for the conditionally approved single-machine pilot, subject to the deployment owner’s runtime write, readiness, authenticated-job, restart-persistence, and cost/account checks. No Fly mutation was performed.

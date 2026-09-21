# Self-hosted ModernBookNLP API

Run one authenticated HTTP API and one model worker on a host with a persistent local disk. Jobs survive restarts. The model stays loaded between jobs. This CPU deployment is intended for bounded English prose analysis, not real-time speech generation.

## Local Docker deployment

From the repository root:

```sh
python3 -m service.keygen --client speedreadify --directory ../booknlp-secrets
cp ../booknlp-secrets/server.env .env
docker compose up -d --build
curl --fail http://127.0.0.1:8787/healthz
curl --fail http://127.0.0.1:8787/readyz
```

Protect `.env` and the client-key file. Only the hash belongs on the server; put the raw client key in the calling backend's secret store. Neither belongs in an iOS app, browser bundle, URL, source control or chat. The generator refuses to overwrite existing credential files. The API fails startup when no key mapping is configured.

The first start downloads model files and base-model assets, which can take several minutes. `/healthz` indicates the HTTP process is alive. `/readyz` returns 503 until the worker finishes loading, or when its heartbeat is stale. Submissions can queue during loading. Monitor both probes: an HTTP 200 health check does not prove inference is ready.

The persistent named volume stores SQLite, the model cache and Hugging Face assets. `docker compose down` retains it; **do not use `down -v`** unless intentionally erasing queued jobs, results and model downloads. The API binds to loopback port 8787 by default. Docker's `unless-stopped` policy restarts after process failure and daemon restart; the host must remain awake and its Docker daemon must start on boot.

## Always-on Linux host and HTTPS

Provisional host sizing: 4 CPU cores, 16 GiB RAM and at least 30 GiB free SSD. The included container limit is 8 GiB with four CPU cores. The original Mac excerpt pilot used 3.26 GiB peak RSS; that is not a whole-book memory bound. Use a single instance. SQLite must stay on a local filesystem, not an NFS share or independently replicated disks.

Install Docker Engine with the Compose plugin using the host OS's official instructions. Clone this fork at the reviewed commit, generate credentials, then set `BOOKNLP_DOMAIN` in `.env` to a domain you control with DNS pointing at this host. Allow inbound TCP 80/443 and restricted administrative SSH; the API port remains loopback-only.

```sh
docker compose -f compose.yaml -f deploy/https.compose.yaml up -d --build
```

Caddy provides HTTPS termination and forwards to the private API container. Use the provider's persistent disk/volume; disable platform sleep/autostop and keep one instance running. A hosted HTTPS endpoint is not verified until its external `/readyz` and an authenticated real job pass. Do not use a temporary development tunnel as the availability strategy.

## HTTP contract

All `/v1/` endpoints require `Authorization: Bearer <key>`. `BOOKNLP_API_KEY_HASHES` is a JSON object mapping lowercase SHA-256 key digests to client names. Separate credentials for separate callers create separate job namespaces; keys mapped to the same client intentionally share that namespace. A SpeedReadify backend using one service credential must enforce its own per-user document ownership before proxying job IDs or results. This service authenticates clients, not SpeedReadify end users.

| Method and path | Behavior |
| --- | --- |
| POST `/v1/jobs` | JSON `{"text":"original prose"}` → 202 `{"data":job}`. Repeated identical text reuses that client's job within retention. |
| GET `/v1/jobs/{id}` | 200 `{"data":job}`; unknown or another client's ID → 404. |
| DELETE `/v1/jobs/{id}` | Removes queued or terminal job → 204; running job → 409. |
| GET `/v1/openapi.json` | Authenticated machine-readable route schema. |
| GET `/healthz` | Public liveness. |
| GET `/readyz` | Public worker readiness. |

A job contains `id`, `status`, `content_hash`, `model_version`, `attempts`, timestamps, `error` and `result`. Status is `queued`, `running`, `completed` or `failed`. There is no estimated percentage: the model does not expose reliable progress. Poll every 3–10 seconds with backoff on transport errors. Submission has content-based deduplication, so retrying a POST after a network failure does not duplicate inference. Cache identity includes the pipeline version, thread count and attribution batch size. Pending jobs from an incompatible configuration fail explicitly with `model_version_changed`; resubmit them after the upgrade. Failed jobs are also reused until deleted or expired; after diagnosing the cause, delete a failed job and resubmit to explicitly retry.

`result.quotes` contains the exact source text, character ID, speaker-mention span, and quote offsets. `start`/`end` are **half-open Unicode code point indices**. `start_utf16`/`end_utf16` are half-open UTF-16 indices suitable for Swift `NSRange`; never treat them as Swift Character or byte indices. Source text is not normalized; CRLF and emoji retain their positions. Token/source mismatch fails the job instead of returning corrupt highlights.

`result.characters` provides IDs, suggested display names and proper-name aliases. IDs are scoped to **one analysis**, not stable across separate chapters or model versions. The consuming app must save a document cast and reconcile aliases/merges before assigning voices. This API does not invent confidence values, promise canonical identity accuracy, perform voice assignment or alter existing playback.

Run the real HTTP smoke test without putting the key in command arguments:

```sh
python3 -m service.smoke --url http://127.0.0.1:8787 \
  --key-file ../booknlp-secrets/client-key.txt --output ../smoke-result.json
```

## Bounds and recovery

Environment settings (positive integers) apply to both API and worker:

| Variable | Default |
| --- | ---: |
| BOOKNLP_MAX_CHARS | 50000 |
| BOOKNLP_MAX_PENDING | 20 |
| BOOKNLP_MAX_CLIENT_PENDING | 5 |
| BOOKNLP_MAX_ATTEMPTS | 2 |
| BOOKNLP_TIMEOUT_SECONDS | 600 |
| BOOKNLP_STARTUP_SECONDS | 1800 |
| BOOKNLP_RETENTION_SECONDS | 86400 |
| BOOKNLP_THREADS | 4 |
| BOOKNLP_BATCH_SIZE | 2 |

Body size is bounded before JSON parsing; oversize prose → 413, invalid input → 422, queue/record capacity → 429 with Retry-After. At most 1000 retained jobs are admitted. Reverse-proxy request size is also capped; adjust it alongside the application bound if you deliberately raise limits. These bounds limit resource usage, not billing guarantees or full-book support.

The worker holds an exclusive filesystem lock. A second worker for the same volume exits rather than processing concurrently. On restart, abandoned running jobs are requeued if attempts remain. A supervisor enforces inference deadlines by terminating the model subprocess before retrying. Worker crashes after the final attempt become failed jobs. Retries can repeat computation after a crash but cannot publish multiple records for the same cached job.

The stored full input is cleared after completion or final failure. Completed/failed records expire after the retention interval; old queued jobs fail with `queue_expired`. Results contain quotations and therefore remain sensitive until deleted/expired. SQLite logical cleanup and scratch removal are not cryptographic erasure; volume snapshots/backups have their own retention. API errors omit original prose and server paths. Logs rotate in Compose; keep host storage monitored.

Back up the database using SQLite's backup API into a separately protected location, or stop the service before snapshotting the volume. Do not copy just the live `.sqlite3` file while WAL writes are active. Restore with the service stopped and preserve UID 10001 ownership. Models can be downloaded again; jobs/results cannot. Upgrades on this one-host deployment have brief downtime. Never change schema incompatibly without a tested migration/rollback.

## Development and provenance

Python 3.11. Install torch 2.6.0 from the official CPU wheel index, `service/requirements.lock`, the official spaCy `en_core_web_sm` 3.8.0 wheel and a compatible HTTP test client (`httpx==0.28.1`). Run `python -m unittest discover -s service/tests`.

Only two upstream behavior-neutral service changes are made: remove fixed debug pickle writes, and read source as explicit UTF-8 without newline conversion. Pretrained entity/coreference/joint weight hashes are verified before loading. Base Hugging Face model/tokenizer assets still follow upstream repository defaults on initial download; preserve the model volume for repeatability. A fully hermetic base-model snapshot is a separate reproducibility improvement.

No inference weights were trained or altered. Repository and pretrained model licenses are distinct: retain upstream notices and review the actual model/dataset terms for the intended deployment. The API service does not itself establish commercial licensing clearance.

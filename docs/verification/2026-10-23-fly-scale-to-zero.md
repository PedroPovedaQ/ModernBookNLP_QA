# Fly scale-to-zero verification — 2026-10-23

## Verdict

**PASS.** `speedreadify-booknlp` now suspends when idle and resumes on any request.
Idle cost fell from about $55/month to about $0.55/month. Output on a public-domain
excerpt is byte-identical to the always-on pilot.

## Shape

- Machine `871537b0410008`, `iad`, `performance-4x`, 8192 MB. Volume
  `vol_4oj60wqd3jn9ywqr` (`booknlp_jobs`, 1 GB) holds SQLite jobs only.
- `auto_stop_machines = "suspend"`, `auto_start_machines = true`, `min_machines_running = 0`.
- Model weights, HF configs and tokenizers are baked into a 2.6 GB image and checksum-verified
  at build time. Boot runs with `HF_HUB_OFFLINE=1` and does no downloads.
- `BOOKNLP_SKIP_BASE_WEIGHTS=1` builds encoders from their configs instead of reading
  base checkpoints. Every encoder key is overwritten by the fine-tuned weights (checked:
  no missing keys; ModernQA loads strictly; only computed rotary buffers remain).
- The old 30 GB volume `booknlp_data` held weights and pilot smoke jobs. It was destroyed
  after the baked image passed.

## Cold start breakdown

Fly's disk limit is the bottleneck. Cold rootfs and volume reads both measured
about 40 MB/s on `shared-cpu-4x` (documented cap 32 MiB/s). `performance-2x`/`4x` have
a 64 MiB/s cap.

| Phase | Before (shared-cpu-4x, weights on volume) | After (performance-4x, baked) |
| --- | --- | --- |
| VM start | 1.2 s | 0.5–1.7 s |
| torch import | 79 s, which includes SHA-256 over 2.3 GB and booknlp imports | 2.7 s, with re-hashing skipped |
| model init | 81 s, which includes 2.3 GB of base weights | 53 s |
| `/readyz` 200 after start | 158 s | 55–60 s cold, 0.8–2.3 s from suspend |

A background read-ahead was tried and reverted, because the imports share the same
disk bandwidth (90.7 s vs 85.5 s).

## Suspend vs stop

The docs list a 2 GB suspend limit. Suspend still worked on 8 GB with a volume: about
8 s to snapshot and 0.8–2.3 s to resume. The first job after a resume ran at normal
speed (60.7 s). The proxy suspended the idle Machine 3.75 and 4.5 minutes after the last
request. Health checks did not keep it awake. A deploy leaves the Machine stopped, so
the next request takes the cold path.

## Latency and accuracy

The excerpt was *A Scandal in Bohemia*, part I (Project Gutenberg #1661, 19,757
characters, 4,341 words). Every run below returned 138 quotes and 8 characters. Each
`result` was byte-identical to the pre-change baseline.

| Configuration | Job end-to-end |
| --- | --- |
| Before: always-on shared-cpu-4x with full burst balance | 43.9 s |
| shared-cpu-4x after a suspend/resume (burst balance exhausted) | 473 s (attribution 371 s) |
| performance-2x / 8 GB | 90.8 s |
| performance-4x / 8 GB, warm, after cold boot and after resume | 58.4 / 59.2 / 60.7 s |

A scaled-to-zero shared vCPU starts with a 5 s burst balance and then throttles to
6.25%. CPU steal was 3.5x user time during the slow job. Performance vCPUs avoid this.

## Cost

| Item | Before | After |
| --- | --- | --- |
| Idle: volume plus rootfs | $4.50 (30 GB volume, always running) | $0.15 (1 GB volume) + ~$0.39 (2.6 GB rootfs at $0.15/GB) ≈ **$0.55/month** |
| Compute | $50.78/month, always on | ~$0.18 per awake hour (performance-4x at $33 per perf vCPU plus 2 GB) |

Each wake bills about 5 minutes (≈ $0.015) plus job time (≈ $0.003 per 20k-character
excerpt). Spend matches the old $55/month only at about 300 awake hours per month.

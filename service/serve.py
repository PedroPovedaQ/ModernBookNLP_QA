"""Run API and worker together on a single persistent-volume host."""

import os
import signal
import subprocess
import sys
import threading


def main():
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    processes = []
    failed = False
    try:
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "service.api:create_app",
                    "--factory",
                    "--host",
                    os.environ.get("BOOKNLP_BIND_HOST", "0.0.0.0"),
                    "--port",
                    os.environ.get("PORT", "8000"),
                    "--workers",
                    "1",
                    "--limit-concurrency",
                    "32",
                    "--timeout-keep-alive",
                    "5",
                ]
            )
        )
        processes.append(subprocess.Popen([sys.executable, "-m", "service.worker"]))
        while not stopped.wait(1):
            if any(process.poll() is not None for process in processes):
                failed = True
                break
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

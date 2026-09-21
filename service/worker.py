"""Single-host worker: durable queue, persistent model child, enforced deadlines."""

import fcntl
import logging
import multiprocessing
import os
import shutil
import signal
import threading
import time
from contextlib import contextmanager

from service.config import Settings
from service.store import Store

log = logging.getLogger(__name__)


@contextmanager
def worker_lock(path):
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def model_child(connection, settings, parent_pid):
    # Ensure a supervisor SIGKILL cannot leave an orphan doing inference indefinitely.
    def watch_parent():
        while os.getppid() == parent_pid:
            time.sleep(1)
        os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        from service.model import Analyzer

        analyzer = Analyzer(settings)
        connection.send(("ready", None))
        while True:
            text = connection.recv()
            try:
                result = analyzer.analyze(text)
                connection.send(("ok", result))
            except Exception as exc:
                log.error("Analysis failed (%s)", type(exc).__name__)
                connection.send(("error", None))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        log.error("Model startup failed (%s)", type(exc).__name__)
    finally:
        connection.close()


class ModelProcess:
    def __init__(self, settings, target=model_child):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=target, args=(child, settings, os.getpid()), daemon=True
        )
        self.process.start()
        child.close()

    def wait(self, seconds, store, stopped, ready=False):
        deadline = time.monotonic() + seconds
        while not stopped.is_set() and time.monotonic() < deadline:
            store.heartbeat(ready)
            if self.connection.poll(min(1, max(0, deadline - time.monotonic()))):
                try:
                    return self.connection.recv()
                except EOFError:
                    raise RuntimeError("Model process exited") from None
            if not self.process.is_alive():
                raise RuntimeError("Model process exited")
        raise TimeoutError("Model stopped or deadline exceeded")

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        if self.process.is_alive():
            raise RuntimeError("Unable to stop model process")
        self.connection.close()


def run(settings: Settings):
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    store = Store(settings)
    with worker_lock(settings.data_dir / "worker.lock"):
        store.recover()
        model = None
        try:
            while not stopped.is_set():
                # Only generated scratch directories; no caller-controlled paths.
                for directory in settings.data_dir.glob("analysis-*"):
                    if directory.is_dir() and not directory.is_symlink():
                        shutil.rmtree(directory)
                if model is None:
                    store.heartbeat(False)
                    model = ModelProcess(settings)
                    try:
                        kind, _ = model.wait(settings.startup_seconds, store, stopped)
                        if kind != "ready":
                            raise RuntimeError("Invalid model startup response")
                    except (TimeoutError, RuntimeError):
                        model.close()
                        model = None
                        log.error(
                            "Model unavailable; retrying initialization in 30 seconds"
                        )
                        stopped.wait(30)
                        continue
                store.heartbeat(True)
                store.cleanup()
                job = store.claim()
                if job is None:
                    stopped.wait(1)
                    continue
                log.info("Processing job %s attempt %s", job["id"], job["attempts"])
                try:
                    model.connection.send(job["text"])
                    kind, result = model.wait(
                        settings.timeout_seconds, store, stopped, ready=True
                    )
                    if kind != "ok":
                        raise RuntimeError("Analysis failed")
                    store.finish(job["id"], result)
                    log.info("Completed job %s", job["id"])
                except (TimeoutError, RuntimeError, BrokenPipeError, EOFError):
                    model.close()
                    model = None
                    store.fail(
                        job["id"],
                        "analysis_interrupted"
                        if stopped.is_set()
                        else "analysis_failed_or_timed_out",
                    )
        finally:
            if model is not None:
                model.close()
            store.heartbeat(False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(Settings.from_env())

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

MODEL_VERSION = "modern-joint-84f1f34-adapter-v1"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    keys: dict[str, str] = field(default_factory=dict)
    max_chars: int = 50_000
    max_pending: int = 20
    max_client_pending: int = 5
    max_attempts: int = 2
    timeout_seconds: int = 600
    startup_seconds: int = 1800
    retention_seconds: int = 86400
    threads: int = 4
    batch_size: int = 2

    def __post_init__(self):
        for name in (
            "max_chars",
            "max_pending",
            "max_client_pending",
            "max_attempts",
            "timeout_seconds",
            "startup_seconds",
            "retention_seconds",
            "threads",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", k)
            or not isinstance(v, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", v)
            for k, v in self.keys.items()
        ):
            raise ValueError("Keys must map lowercase SHA-256 hashes to client names")
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    @property
    def analysis_version(self) -> str:
        return f"{MODEL_VERSION}-threads{self.threads}-batch{self.batch_size}"

    @classmethod
    def from_env(cls):
        keys = json.loads(os.environ.get("BOOKNLP_API_KEY_HASHES", "{}"))
        if not isinstance(keys, dict):
            raise ValueError("BOOKNLP_API_KEY_HASHES must be an object")
        fields = {
            name: int(os.environ["BOOKNLP_" + name.upper()])
            for name in (
                "max_chars",
                "max_pending",
                "max_client_pending",
                "max_attempts",
                "timeout_seconds",
                "startup_seconds",
                "retention_seconds",
                "threads",
                "batch_size",
            )
            if "BOOKNLP_" + name.upper() in os.environ
        }
        return cls(
            data_dir=Path(os.environ.get("BOOKNLP_DATA_DIR", "./data")),
            keys=keys,
            **fields,
        )

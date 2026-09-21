"""Write a credential and its server-side hash to private files; never log it."""

import argparse
import hashlib
import json
import os
import re
import secrets
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="speedreadify")
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.client):
        parser.error("Invalid client name")
    args.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = secrets.token_urlsafe(40)
    for name, value in [
        ("client-key.txt", key + "\n"),
        (
            "server.env",
            "BOOKNLP_API_KEY_HASHES='"
            + json.dumps({hashlib.sha256(key.encode()).hexdigest(): args.client})
            + "'\n",
        ),
    ]:
        fd = os.open(args.directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
    print("Credential files created in", args.directory)


if __name__ == "__main__":
    main()

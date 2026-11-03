"""Content bindings and immutable JSON outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular, non-symlink file required: {path.name}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"execution", "budget"}}


def read_json(path: Path, expected_sha256: str | None = None) -> Any:
    if path.is_symlink():
        raise ValueError("symlink JSON input prohibited")
    content = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"JSON commitment mismatch: {path.name}")
    return json.loads(content)


def write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def require_private_output(path: Path) -> None:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".git").exists():
            raise ValueError("row-level outputs must be outside every Git worktree")

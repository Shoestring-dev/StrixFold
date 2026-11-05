"""CPU-only, bounded I/O shared by the DeepSeek HIP input tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


class InputError(ValueError):
    """An input is incomplete, incompatible, or failed verification."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json(data: bytes | str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_unique_object,
                          parse_constant=lambda s: (_ for _ in ()).throw(InputError(f"invalid JSON constant: {s}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"invalid JSON: {exc}") from exc


def read_json(path: Path, limit: int = 128 << 20) -> Any:
    if path.stat().st_size > limit:
        raise InputError(f"JSON exceeds {limit} bytes: {path}")
    return decode_json(path.read_bytes())


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        while block := src.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise InputError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def checked_relative(root: Path, name: str) -> Path:
    """Permit HF-style subdirectories but never escape a snapshot via an index."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise InputError(f"invalid relative path: {name!r}")
    result = root / name
    if not result.resolve().is_relative_to(root.resolve()):
        raise InputError(f"path escapes snapshot: {name}")
    return result


def checked_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}")
    return value


def cli_error(exc: Exception) -> int:
    print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
    return 1

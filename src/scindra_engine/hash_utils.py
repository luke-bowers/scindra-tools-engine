from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def _to_path(path: str | Path) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""

    file_path = _to_path(path)
    hasher = sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_bytes(path: str | Path) -> int:
    """Return the file size in bytes."""

    file_path = _to_path(path)
    return file_path.stat().st_size


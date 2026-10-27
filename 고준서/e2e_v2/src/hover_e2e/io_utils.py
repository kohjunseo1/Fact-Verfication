from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Broken JSONL at {path}:{line_number}; do not silently skip it") from exc
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record must be object at {path}:{line_number}")
            yield value


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def index_by_uid(rows: Iterable[dict[str, Any]], *, expected_uids: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row.get("uid", ""))
        if not uid:
            raise ValueError("Every record must have a non-empty uid")
        if uid in output:
            raise ValueError(f"Duplicate uid: {uid}")
        output[uid] = row
    if expected_uids is not None:
        expected = [str(uid) for uid in expected_uids]
        if set(output) != set(expected):
            missing = sorted(set(expected) - set(output))
            extra = sorted(set(output) - set(expected))
            raise ValueError(f"UID mismatch: missing={missing[:5]}, extra={extra[:5]}")
    return output


class JsonlCheckpoint:
    """Append-only, config-bound checkpoint suitable for Google Drive.

    It never rewrites the growing predictions file. A tiny metadata JSON is
    atomically updated after flushes. Duplicate UIDs or config mixing fail fast.
    """

    def __init__(self, directory: str | Path, config: dict[str, Any], *, flush_every: int = 25):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records_path = self.directory / "predictions.jsonl"
        self.state_path = self.directory / "checkpoint.json"
        self.config_hash = fingerprint(config)
        self.flush_every = max(1, int(flush_every))
        self._completed_uids: set[str] = set()
        self._pending = 0
        if self.state_path.is_file():
            state = read_json(self.state_path)
            if state.get("config_sha256") != self.config_hash:
                raise RuntimeError("Checkpoint config mismatch; use a new run name")
        if self.records_path.is_file():
            for record in iter_jsonl(self.records_path):
                uid = str(record.get("uid", ""))
                if not uid or uid in self._completed_uids:
                    raise ValueError(f"Invalid or duplicate checkpoint uid: {uid!r}")
                self._completed_uids.add(uid)
        self._stream = self.records_path.open("a", encoding="utf-8")

    def is_completed(self, uid: str) -> bool:
        """O(1) lookup without copying the full UID set on every prediction."""
        return str(uid) in self._completed_uids

    @property
    def completed_count(self) -> int:
        return len(self._completed_uids)

    def append(self, record: dict[str, Any]) -> None:
        uid = str(record.get("uid", ""))
        if not uid or uid in self._completed_uids:
            raise ValueError(f"Invalid or duplicate checkpoint uid: {uid!r}")
        self._stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._completed_uids.add(uid)
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())
        atomic_write_json(self.state_path, {
            "schema_version": 1,
            "config_sha256": self.config_hash,
            "completed": len(self._completed_uids),
        })
        self._pending = 0

    def close(self) -> None:
        if not self._stream.closed:
            self.flush()
            self._stream.close()

    def __enter__(self) -> "JsonlCheckpoint":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

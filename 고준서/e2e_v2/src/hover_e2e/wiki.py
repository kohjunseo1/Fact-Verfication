from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import sqlite3
from typing import Iterable

from .normalization import db_title, nfc_title, stable_unique, title_key


class WikiStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        with self.path.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                raise ValueError(f"Not a SQLite database: {self.path}")
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=60)
        self.connection.execute("PRAGMA query_only=ON")
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(documents)")}
        if not {"id", "text"} <= columns:
            raise ValueError(f"Unexpected documents schema: {columns}")

    @lru_cache(maxsize=50_000)
    def get(self, title: str) -> tuple[str, str] | None:
        variants = stable_unique([db_title(title), str(title), nfc_title(title)])
        for variant in variants:
            row = self.connection.execute(
                "SELECT id, text FROM documents WHERE id=?", (variant,)
            ).fetchone()
            if row is not None:
                return nfc_title(row[0]), str(row[1])
        return None

    def text(self, title: str) -> str | None:
        row = self.get(title)
        return None if row is None else row[1]

    def iter_titles(self) -> Iterable[str]:
        for (title,) in self.connection.execute("SELECT id FROM documents"):
            yield nfc_title(title)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "WikiStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")
_TOKEN = re.compile(r"[\w][\w'.\-&$]*|[^\w\s]", re.UNICODE)
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "for",
    "by", "with", "from", "as", "is", "was", "are", "were", "be", "it",
}


def build_title_index(wiki: WikiStore, destination: str | Path, *, batch_size: int = 50_000) -> Path:
    """Build a reusable normalized title lookup without loading 5.2M titles into RAM."""
    path = Path(destination).resolve()
    if path.is_file():
        with sqlite3.connect(path) as connection:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            if (
                metadata.get("complete") == "1"
                and metadata.get("source_size") == str(wiki.path.stat().st_size)
            ):
                print(f"[title-index] reuse: {path}", flush=True)
                return path
        raise RuntimeError(f"Incomplete/stale title index; move it aside before rebuilding: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[title-index] building from {wiki.path} -> {path}", flush=True)
    temporary = path.with_name(path.name + ".tmp")
    with sqlite3.connect(temporary) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS titles(surface_key TEXT NOT NULL,title TEXT NOT NULL,PRIMARY KEY(surface_key,title)) WITHOUT ROWID")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        source_size = str(wiki.path.stat().st_size)
        if metadata and metadata.get("source_size") != source_size:
            raise RuntimeError(f"Partial title index belongs to another wiki DB: {temporary}")
        if not metadata:
            connection.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("complete", "0"), ("source_size", source_size), ("last_rowid", "0"),
                ("normalization", "NFC+spaces+casefold,parenthetical-alias-v1"),
            ])
            connection.commit()
        last_rowid = int(dict(connection.execute("SELECT key,value FROM metadata")).get("last_rowid", "0"))
        batch: list[tuple[str, str]] = []
        cursor = wiki.connection.execute(
            "SELECT rowid,id FROM documents WHERE rowid>? ORDER BY rowid", (last_rowid,)
        )
        for index, (rowid, raw_title) in enumerate(cursor, 1):
            title = nfc_title(raw_title)
            surfaces = {title_key(title)}
            base = _PARENTHETICAL.sub("", title)
            if base != title:
                surfaces.add(title_key(base))
            batch.extend((surface, title) for surface in surfaces if surface)
            if len(batch) >= batch_size:
                connection.executemany("INSERT OR IGNORE INTO titles VALUES (?,?)", batch)
                connection.execute(
                    "INSERT OR REPLACE INTO metadata VALUES ('last_rowid',?)", (str(rowid),)
                )
                connection.commit()
                batch.clear()
                if index % 500_000 < batch_size:
                    print(f"title index: {index:,}", flush=True)
        if batch:
            connection.executemany("INSERT OR IGNORE INTO titles VALUES (?,?)", batch)
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('complete','1')")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    temporary.replace(path)
    print(f"[title-index] complete: {path}", flush=True)
    return path


class TitleMatcher:
    def __init__(
        self,
        index_path: str | Path,
        *,
        max_ngram: int = 8,
        max_titles_per_surface: int = 8,
        max_text_chars: int = 12_000,
    ):
        self.path = Path(index_path).resolve()
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=60)
        self.connection.execute("PRAGMA query_only=ON")
        self.max_ngram = int(max_ngram)
        self.max_titles_per_surface = int(max_titles_per_surface)
        self.max_text_chars = int(max_text_chars)

    def _surface_keys(self, text: str) -> list[str]:
        matches = list(_TOKEN.finditer(text))
        surfaces: list[str] = []
        for start in range(len(matches)):
            for width in range(1, min(self.max_ngram, len(matches) - start) + 1):
                end = start + width - 1
                value = text[matches[start].start():matches[end].end()].strip(" \t\n,;:!?\"'")
                if not value:
                    continue
                if width == 1 and (len(value) < 3 or value.casefold() in _STOP):
                    continue
                surfaces.append(title_key(value))
        return stable_unique(surfaces)

    def match(self, text: str) -> tuple[str, ...]:
        # Do not LRU-cache raw Wikipedia bodies: retaining thousands of large
        # strings can exhaust Colab RAM. HoVer's lead-text DB is still capped
        # so pathological rows cannot dominate expansion time.
        keys = self._surface_keys(text[:self.max_text_chars])
        hits: dict[str, list[str]] = {}
        for offset in range(0, len(keys), 800):
            chunk = keys[offset:offset + 800]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT surface_key,title FROM titles WHERE surface_key IN ({placeholders}) ORDER BY surface_key,title"
            for surface, title in self.connection.execute(query, chunk):
                values = hits.setdefault(surface, [])
                if len(values) < self.max_titles_per_surface:
                    values.append(nfc_title(title))
        output = []
        for key in keys:
            output.extend(hits.get(key, ()))
        return tuple(stable_unique(output, key=title_key))

    def close(self) -> None:
        self.connection.close()

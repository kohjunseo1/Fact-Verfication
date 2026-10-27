from __future__ import annotations

import re
import unicodedata
from typing import Iterable, TypeVar


_SPACE = re.compile(r"\s+")
T = TypeVar("T")


def nfc_text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value))


def nfc_title(value: object) -> str:
    """Canonical comparison/display form; never changes a sentence index."""
    return _SPACE.sub(" ", nfc_text(value).replace("_", " ").strip())


def title_key(value: object) -> str:
    return nfc_title(value).casefold()


def db_title(value: object) -> str:
    """HoVer Wikipedia SQLite stores most titles in NFD."""
    return unicodedata.normalize("NFD", nfc_title(value))


def stable_unique(values: Iterable[T], key=lambda value: value) -> list[T]:
    output: list[T] = []
    seen: set[object] = set()
    for value in values:
        marker = key(value)
        if marker not in seen:
            seen.add(marker)
            output.append(value)
    return output


def unique_titles(values: Iterable[object]) -> list[str]:
    return stable_unique((nfc_title(value) for value in values), key=title_key)


def evidence_key(title: object, sentence_id: object) -> tuple[str, int]:
    if isinstance(sentence_id, bool) or not isinstance(sentence_id, int):
        raise TypeError(f"sentence_id must be int, got {sentence_id!r}")
    return title_key(title), sentence_id

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable

from .normalization import db_title, nfc_title, title_key
from .wiki import WikiStore


def _normalized_gold_text(value: str) -> str:
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def load_gold_text_references(
    path: str | Path,
) -> dict[tuple[str, int], set[str]]:
    """Load canonical Hotpot/HoVer Gold texts; absence is never treated as verified."""
    source = Path(path)
    if not source.is_file():
        return {}
    output: dict[tuple[str, int], set[str]] = {}
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Broken Gold text reference: {source}:{line_number}") from exc
            for fact in record.get("gold_facts", []):
                sentence_id = fact.get("sentence_id")
                text = fact.get("text")
                if (
                    fact.get("status") == "ok"
                    and isinstance(sentence_id, int)
                    and not isinstance(sentence_id, bool)
                    and isinstance(text, str)
                    and text.strip()
                ):
                    key = (title_key(fact.get("title", "")), sentence_id)
                    output.setdefault(key, set()).add(_normalized_gold_text(text))
    return output


class SentenceStore:
    """Read exact indexed sentence arrays; never invent or shift sentence IDs."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        overrides_path: str | Path | None = None,
        writable: bool = False,
    ):
        self.path = Path(cache_path).resolve()
        if not self.path.is_file() and not writable:
            raise FileNotFoundError(f"Sentence cache not found: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60)
        if not writable:
            self.connection.execute("PRAGMA query_only=ON")
        else:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS documents(
                    title_key TEXT PRIMARY KEY,
                    requested_title TEXT NOT NULL,
                    db_title TEXT NOT NULL,
                    db_text_sha256 TEXT NOT NULL,
                    sentences_json TEXT NOT NULL,
                    offsets_json TEXT NOT NULL,
                    updated_utc TEXT NOT NULL
                )
            """)
            self.connection.commit()
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(documents)")}
        if not {"title_key", "sentences_json"} <= columns:
            raise ValueError(f"Unexpected sentence cache schema: {columns}")
        self.overrides = self._load_overrides(overrides_path)

    @staticmethod
    def _load_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
        if path is None or not Path(path).is_file():
            return {}
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and "documents" in payload:
            payload = payload["documents"]
        if isinstance(payload, dict):
            records = []
            for mapping_title, value in payload.items():
                if isinstance(value, dict):
                    records.append({"title": mapping_title, **value})
        else:
            records = payload
        output = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("sentences"), list):
                continue
            title = record.get("title") or record.get("requested_title") or record.get("db_title")
            if title:
                output[title_key(title)] = record
        return output

    def get(self, title: str) -> dict[str, Any] | None:
        key = title_key(title)
        override = self.overrides.get(key)
        if override is not None:
            sentences = list(override["sentences"])
            return {
                "title": nfc_title(title), "sentences": sentences,
                "offsets": override.get("offsets", [None] * len(sentences)),
                "source": "verified_original_sentence_array",
                "sentence_array_sha256": hashlib.sha256(
                    json.dumps(sentences, ensure_ascii=False, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        variants = [db_title(title), nfc_title(title), key]
        row = None
        for variant in dict.fromkeys(variants):
            row = self.connection.execute(
                "SELECT requested_title,db_title,db_text_sha256,sentences_json,offsets_json "
                "FROM documents WHERE title_key=?", (variant,)
            ).fetchone()
            if row is not None:
                break
        if row is None:
            # Some caches used NFD requested/db title but a different title_key policy.
            row = self.connection.execute(
                "SELECT requested_title,db_title,db_text_sha256,sentences_json,offsets_json "
                "FROM documents WHERE requested_title=? OR db_title=? LIMIT 1",
                (db_title(title), db_title(title)),
            ).fetchone()
        if row is None:
            return None
        sentences = json.loads(row[3])
        offsets = json.loads(row[4])
        if len(offsets) != len(sentences):
            raise ValueError(f"Sentence/offset length mismatch: {title}")
        return {
            "title": nfc_title(row[1]), "sentences": sentences, "offsets": offsets,
            "db_text_sha256": row[2], "source": "corenlp_cache",
            "sentence_array_sha256": hashlib.sha256(
                json.dumps(sentences, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    def put(
        self, *, requested_title: str, database_title: str, raw_text: str,
        sentences: list[str], offsets: list[list[int]], updated_utc: str,
    ) -> None:
        if self.connection.execute("PRAGMA query_only").fetchone()[0]:
            raise RuntimeError("SentenceStore is read-only")
        if len(sentences) != len(offsets):
            raise ValueError("Sentence/offset length mismatch")
        for sentence, offset in zip(sentences, offsets):
            if not (
                isinstance(offset, list) and len(offset) == 2
                and all(isinstance(value, int) for value in offset)
                and 0 <= offset[0] < offset[1] <= len(raw_text)
                and raw_text[offset[0]:offset[1]] == sentence
            ):
                raise ValueError("CoreNLP offsets do not reproduce DB text exactly")
        self.connection.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?)",
            (
                db_title(requested_title), requested_title, database_title,
                hashlib.sha256(raw_text.encode()).hexdigest(),
                json.dumps(sentences, ensure_ascii=False),
                json.dumps(offsets), updated_utc,
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def backup_to(self, destination: str | Path) -> None:
        """Consistent SQLite backup, safe while the writable cache is open."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        with sqlite3.connect(temporary) as backup:
            self.connection.backup(backup)
        temporary.replace(path)


def audit_gold_alignment(
    claims: Iterable[dict[str, Any]], store: SentenceStore, *, wiki: WikiStore | None = None,
    gold_texts: dict[tuple[str, int], set[str]] | None = None,
) -> dict[str, Any]:
    claims = list(claims)
    references = gold_texts or {}
    errors = []
    unverified = []
    sources = Counter()
    checked_facts = 0
    offsets_checked = 0
    offsets_verified = 0
    gold_text_verified_facts = 0
    facts_by_uid = Counter()
    verified_facts_by_uid = Counter()
    for row in claims:
        uid = str(row["uid"])
        for title, sentence_id in row["supporting_facts"]:
            checked_facts += 1
            facts_by_uid[uid] += 1
            document = store.get(title)
            if document is None:
                errors.append({"uid": uid, "title": nfc_title(title), "sentence_id": sentence_id, "reason": "missing_sentence_array"})
                continue
            sources[document["source"]] += 1
            sentences = document["sentences"]
            if (
                isinstance(sentence_id, bool) or not isinstance(sentence_id, int)
                or not 0 <= sentence_id < len(sentences)
            ):
                errors.append({
                    "uid": uid, "title": nfc_title(title), "sentence_id": sentence_id,
                    "available_sentences": len(sentences), "reason": "gold_id_out_of_range",
                    "source": document["source"],
                })
                continue
            elif not isinstance(sentences[sentence_id], str) or not sentences[sentence_id].strip():
                errors.append({
                    "uid": uid, "title": nfc_title(title), "sentence_id": sentence_id,
                    "reason": "gold_sentence_empty", "source": document["source"],
                })
                continue
            if document["source"] == "corenlp_cache" and wiki is not None:
                raw_document = wiki.get(title)
                offset = document["offsets"][sentence_id]
                valid = False
                if raw_document is not None:
                    _, raw_text = raw_document
                    valid = (
                        hashlib.sha256(raw_text.encode()).hexdigest()
                        == document.get("db_text_sha256")
                        and isinstance(offset, list)
                        and len(offset) == 2
                        and all(isinstance(value, int) for value in offset)
                        and 0 <= offset[0] < offset[1] <= len(raw_text)
                        and raw_text[offset[0]:offset[1]] == sentences[sentence_id]
                    )
                offsets_checked += 1
                if not valid:
                    errors.append({
                        "uid": uid, "title": nfc_title(title),
                        "sentence_id": sentence_id,
                        "reason": "cached_offset_or_db_text_mismatch",
                        "source": document["source"], "offset": offset,
                    })
                    continue
                offsets_verified += 1
            reference_key = (title_key(title), int(sentence_id))
            expected_texts = references.get(reference_key)
            if not expected_texts:
                unverified.append({
                    "uid": uid, "title": nfc_title(title),
                    "sentence_id": sentence_id,
                    "reason": "gold_text_reference_missing",
                    "source": document["source"],
                })
                continue
            actual_text = _normalized_gold_text(sentences[sentence_id])
            if actual_text not in expected_texts:
                errors.append({
                    "uid": uid, "title": nfc_title(title),
                    "sentence_id": sentence_id,
                    "reason": "gold_text_mismatch",
                    "source": document["source"],
                })
                continue
            gold_text_verified_facts += 1
            verified_facts_by_uid[uid] += 1
    gold_text_mismatch_uids = sorted({
        error["uid"] for error in errors if error["reason"] == "gold_text_mismatch"
    })
    integrity_error_uids = sorted({
        error["uid"] for error in errors if error["reason"] != "gold_text_mismatch"
    })
    gold_text_unverified_uids = sorted({item["uid"] for item in unverified})
    verified_uids = sorted(
        uid for uid, total in facts_by_uid.items()
        if verified_facts_by_uid[uid] == total
    )
    excluded_uids = sorted(
        set(integrity_error_uids)
        | set(gold_text_mismatch_uids)
        | set(gold_text_unverified_uids)
    )
    return {
        "checked_facts": checked_facts,
        "offset_integrity": {
            "cached_offsets_checked_against_db": offsets_checked,
            "cached_offsets_verified_against_db": offsets_verified,
            "raw_text_validation_performed": wiki is not None,
            "does_not_verify_gold_text": True,
        },
        "gold_text_alignment": {
            "reference_entries": len(references),
            "verified_facts": gold_text_verified_facts,
            "unverified_facts": len(unverified),
            "verified_claims": len(verified_uids),
            "unverified_claims": len(excluded_uids),
        },
        "error_count": len(errors),
        "affected_claims": len(excluded_uids),
        "affected_uids": excluded_uids,
        "integrity_error_uids": integrity_error_uids,
        "gold_text_mismatch_uids": gold_text_mismatch_uids,
        "gold_text_unverified_uids": gold_text_unverified_uids,
        "gold_text_verified_uids": verified_uids,
        "metric_excluded_uids": excluded_uids,
        "source_counts": dict(sources),
        "errors": errors,
        "unverified_gold_facts": unverified,
        "sentence_ids_modified": False,
        "empty_sentences_inserted": False,
        "validation_bypassed": False,
    }


class CoreNLPHTTP:
    def __init__(self, url: str = "http://127.0.0.1:9000", timeout: int = 120):
        self.url = url
        self.timeout = int(timeout)

    def split(self, text: str) -> tuple[list[str], list[list[int]]]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is required for CoreNLP HTTP access") from exc
        properties = {
            "annotators": "tokenize,ssplit",
            "pipelineLanguage": "en",
            "outputFormat": "json",
        }
        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    self.url, params={"properties": json.dumps(properties)},
                    data=text.encode("utf-8"),
                    headers={"Content-Type": "text/plain; charset=utf-8"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        if response is None or last_error is not None:
            raise RuntimeError(f"CoreNLP request failed three times: {last_error}") from last_error
        # CoreNLP is Java and reports UTF-16 code-unit offsets. Python strings
        # use Unicode code-point offsets; supplementary characters need conversion.
        boundaries = {0: 0}
        utf16_offset = 0
        for python_index, character in enumerate(text, 1):
            utf16_offset += 2 if ord(character) > 0xFFFF else 1
            boundaries[utf16_offset] = python_index
        sentences, offsets = [], []
        for item in response.json().get("sentences", []):
            tokens = item.get("tokens", [])
            if not tokens:
                continue
            raw_start = int(tokens[0]["characterOffsetBegin"])
            raw_end = int(tokens[-1]["characterOffsetEnd"])
            if raw_start not in boundaries or raw_end not in boundaries:
                raise ValueError(f"CoreNLP offset is not a Unicode boundary: {(raw_start, raw_end)}")
            start = boundaries[raw_start]
            end = boundaries[raw_end]
            sentence = text[start:end]
            if not sentence:
                raise ValueError("CoreNLP returned an empty sentence span")
            sentences.append(sentence)
            offsets.append([start, end])
        if not sentences and text.strip():
            raise ValueError("CoreNLP returned no sentences for non-empty text")
        return sentences, offsets

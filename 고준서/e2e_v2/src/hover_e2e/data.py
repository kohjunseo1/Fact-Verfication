from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_json
from .normalization import title_key, unique_titles


@dataclass(frozen=True)
class InferenceClaim:
    uid: str
    claim: str
    num_hops: int


def inference_view(row: dict[str, Any]) -> InferenceClaim:
    """The only object accepted by Dev candidate-generation functions."""
    return InferenceClaim(str(row["uid"]), str(row["claim"]), int(row["num_hops"]))


def load_claims(path: str | Path, *, expected_count: int | None = None) -> list[dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise TypeError(f"Claims must be a JSON list: {path}")
    seen = set()
    for row in rows:
        uid = str(row.get("uid", ""))
        if not uid or uid in seen:
            raise ValueError(f"Missing/duplicate claim uid: {uid!r}")
        seen.add(uid)
        if int(row.get("num_hops", 0)) not in (2, 3, 4):
            raise ValueError(f"Invalid num_hops for {uid}: {row.get('num_hops')}")
        facts = row.get("supporting_facts", [])
        if not isinstance(facts, list):
            raise TypeError(f"supporting_facts must be list: {uid}")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} claims, got {len(rows)} in {path}")
    return rows


def gold_titles(row: dict[str, Any]) -> list[str]:
    return unique_titles(title for title, _ in row["supporting_facts"])


def gold_fact_keys(row: dict[str, Any]) -> set[tuple[str, int]]:
    return {(title_key(title), int(sentence_id)) for title, sentence_id in row["supporting_facts"]}


def load_tfidf(path: str | Path, *, expected_uids: Iterable[str] | None = None) -> dict[str, list[str]]:
    rows = read_json(path)
    output: dict[str, list[str]] = {}
    for row in rows:
        uid = str(row["id"])
        if uid in output:
            raise ValueError(f"Duplicate TF-IDF uid: {uid}")
        result = row.get("doc_retrieval_results")
        if not result or not result[0] or not isinstance(result[0][0], list):
            raise ValueError(f"Unexpected TF-IDF format: {uid}")
        output[uid] = unique_titles(result[0][0])
    if expected_uids is not None:
        expected = {str(uid) for uid in expected_uids}
        if set(output) != expected:
            raise ValueError(
                f"TF-IDF UID mismatch: missing={len(expected-set(output))}, extra={len(set(output)-expected)}"
            )
    return output


def _group_hash(group: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16)


def group_id(row: dict[str, Any]) -> str:
    return str(row.get("hpqa_id") or row["uid"])


def exclude_evaluation_group_overlap(
    train_rows: list[dict[str, Any]], evaluation_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Quarantine Train variants derived from any evaluation hpqa_id."""
    evaluation_groups = {group_id(row) for row in evaluation_rows}
    eligible = [row for row in train_rows if group_id(row) not in evaluation_groups]
    quarantined = [row for row in train_rows if group_id(row) in evaluation_groups]
    if {group_id(row) for row in eligible} & evaluation_groups:
        raise AssertionError("Train/Dev group leakage remains after quarantine")
    return eligible, quarantined


def group_train_validation_split(
    rows: list[dict[str, Any]], *, fraction: float = 0.1, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = group_id(row)
        groups[group].append(row)
    ordered = sorted(groups, key=lambda group: (_group_hash(group, seed), group))
    target = round(len(rows) * fraction)
    validation_groups: set[str] = set()
    count = 0
    for group in ordered:
        if count >= target:
            break
        validation_groups.add(group)
        count += len(groups[group])
    train = [row for row in rows if group_id(row) not in validation_groups]
    validation = [row for row in rows if group_id(row) in validation_groups]
    train_groups = {group_id(row) for row in train}
    val_groups = {group_id(row) for row in validation}
    if train_groups & val_groups:
        raise AssertionError("hpqa_id leakage between train and validation")
    return train, validation


def dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "claims": len(rows),
        "by_hop": dict(sorted(Counter(int(row["num_hops"]) for row in rows).items())),
        "by_label": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "unique_hpqa_ids": len({str(row.get("hpqa_id") or row["uid"]) for row in rows}),
    }

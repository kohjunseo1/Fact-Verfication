from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .data import InferenceClaim
from .retrieval import PairScorer
from .sentences import SentenceStore


def label_id(label: str) -> int:
    normalized = str(label).upper()
    if normalized == "SUPPORTED":
        return 1
    if normalized == "NOT_SUPPORTED":
        return 0
    raise ValueError(f"Unexpected HoVer label: {label!r}")


def evidence_context(items: Iterable[dict[str, Any]], k: int = 5) -> str:
    return " </s> ".join(
        f"{item['title']}. {item.get('sentence', item.get('text', ''))}"
        for item in list(items)[:k]
    )


def gold_context(row: dict[str, Any], store: SentenceStore) -> str:
    pieces = []
    for title, sentence_id in row["supporting_facts"]:
        document = store.get(title)
        if document is None or not 0 <= int(sentence_id) < len(document["sentences"]):
            raise ValueError(f"Unresolved gold sentence: {row['uid']} {title} {sentence_id}")
        pieces.append(f"{title}. {document['sentences'][int(sentence_id)]}")
    return " </s> ".join(pieces)


def _mix_gold(uid: str, ratio: float, seed: int) -> bool:
    value = int(hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()[:16], 16) / 16**16
    return value < ratio


def build_verifier_training_rows(
    claims: Iterable[dict[str, Any]], evidence_predictions: dict[str, dict[str, Any]], *,
    store: SentenceStore, retrieved_k: int = 5, gold_mix_ratio: float = 0.0, seed: int = 42,
    gold_text_verified_uids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if gold_mix_ratio > 0 and gold_text_verified_uids is None:
        raise ValueError("Gold mixing requires explicitly verified Gold-text UIDs")
    claims = list(claims)
    rows, failures = [], []
    print(f"[verifier:pairs] start claims={len(claims):,}", flush=True)
    for claim_index, row in enumerate(claims, 1):
        uid = str(row["uid"])
        use_gold = _mix_gold(uid, gold_mix_ratio, seed)
        source = "retrieved"
        try:
            context = evidence_context(evidence_predictions[uid]["predicted_evidence"], retrieved_k)
            if use_gold and uid in (gold_text_verified_uids or set()):
                context = gold_context(row, store)
                source = "train_gold_mix"
            elif use_gold:
                failures.append({
                    "uid": uid,
                    "reason": "gold_mix_skipped_unverified_gold_text",
                })
        except (KeyError, ValueError) as exc:
            failures.append({"uid": uid, "reason": type(exc).__name__, "detail": str(exc)})
            continue
        rows.append({
            "uid": uid, "text_a": str(row["claim"]), "text_b": context,
            "label": label_id(row["label"]), "source": source,
        })
        if claim_index == 1 or claim_index == len(claims) or claim_index % 500 == 0:
            print(
                f"[verifier:pairs] claims={claim_index:,}/{len(claims):,} "
                f"rows={len(rows):,} failures={len(failures):,}",
                flush=True,
            )
    return rows, failures


def verify_claim(
    claim: InferenceClaim, predicted_evidence: Iterable[dict[str, Any]], *,
    scorer: PairScorer, retrieved_k: int = 5,
) -> dict[str, Any]:
    context = evidence_context(predicted_evidence, retrieved_k)
    supported_probability = float(scorer.score([claim.claim], [context], batch_size=1)[0])
    return {
        "uid": claim.uid, "num_hops": claim.num_hops,
        "predicted_label": int(supported_probability >= 0.5),
        "supported_probability": supported_probability,
        "context": context, "uses_dev_gold": False,
    }

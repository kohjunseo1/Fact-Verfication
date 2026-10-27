from __future__ import annotations

import math
import re
from typing import Any, Iterable

from .data import InferenceClaim, gold_fact_keys, gold_titles
from .normalization import evidence_key, nfc_title, title_key
from .retrieval import PairScorer
from .sentences import SentenceStore


_WORD = re.compile(r"\w+", re.UNICODE)


def _lexical_score(claim: str, sentence: str) -> float:
    a = {token.casefold() for token in _WORD.findall(claim)}
    b = {token.casefold() for token in _WORD.findall(sentence)}
    return len(a & b) / max(len(a), 1)


def sentence_candidates(
    claim: InferenceClaim, document_titles: Iterable[str], *, store: SentenceStore,
    document_limit: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, failures = [], []
    for document_rank, title in enumerate(list(document_titles)[:document_limit], 1):
        document = store.get(title)
        if document is None:
            failures.append({"title": nfc_title(title), "reason": "missing_sentence_array"})
            continue
        for sentence_id, sentence in enumerate(document["sentences"]):
            candidates.append({
                "title": nfc_title(title), "sentence_id": sentence_id,
                "sentence": sentence, "document_rank": document_rank,
                "sentence_source": document["source"],
                "sentence_array_sha256": document["sentence_array_sha256"],
            })
    return candidates, failures


def build_sentence_training_rows(
    claims: Iterable[dict[str, Any]], retrieval_predictions: dict[str, dict[str, Any]], *,
    store: SentenceStore, document_limit: int = 5, negatives_per_positive: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims = list(claims)
    rows, failures = [], []
    print(f"[evidence:pairs] start claims={len(claims):,}", flush=True)
    for claim_index, row in enumerate(claims, 1):
        claim = InferenceClaim(str(row["uid"]), str(row["claim"]), int(row["num_hops"]))
        gold_docs = gold_titles(row)
        retrieved = retrieval_predictions[claim.uid]["ranked_documents"]
        # Gold insertion is Train-only and tagged; inference never calls this builder.
        documents = gold_docs + [title for title in retrieved if title_key(title) not in {title_key(g) for g in gold_docs}]
        candidates, missing = sentence_candidates(claim, documents, store=store, document_limit=document_limit)
        if missing:
            failures.append({"uid": claim.uid, "reason": "missing_sentence_array", "details": missing})
        gold = gold_fact_keys(row)
        positives = [item for item in candidates if evidence_key(item["title"], item["sentence_id"]) in gold]
        if len({evidence_key(item["title"], item["sentence_id"]) for item in positives}) != len(gold):
            failures.append({"uid": claim.uid, "reason": "gold_alignment_unresolved"})
            continue
        negatives = [item for item in candidates if evidence_key(item["title"], item["sentence_id"]) not in gold]
        negatives.sort(key=lambda item: (-_lexical_score(claim.claim, item["sentence"]), item["document_rank"], item["sentence_id"]))
        selected_negatives = negatives[:max(1, len(positives) * negatives_per_positive)]
        for item, label, source in [
            *((item, 1, "train_gold_positive") for item in positives),
            *((item, 0, "retrieved_hard_negative") for item in selected_negatives),
        ]:
            rows.append({
                "uid": claim.uid, "title": item["title"], "sentence_id": item["sentence_id"],
                "text_a": claim.claim, "text_b": f"{item['title']}. {item['sentence']}",
                "label": label, "source": source,
            })
        if claim_index == 1 or claim_index == len(claims) or claim_index % 250 == 0:
            print(
                f"[evidence:pairs] claims={claim_index:,}/{len(claims):,} "
                f"pairs={len(rows):,} failures={len(failures):,}",
                flush=True,
            )
    return rows, failures


def select_evidence(
    claim: InferenceClaim, ranked_documents: Iterable[str], *,
    store: SentenceStore, scorer: PairScorer, document_limit: int = 5,
    sentences_per_document: int = 2, save_top_k: int = 12, batch_size: int = 128,
) -> dict[str, Any]:
    candidates, failures = sentence_candidates(
        claim, ranked_documents, store=store, document_limit=document_limit
    )
    left = [claim.claim] * len(candidates)
    right = [f"{item['title']}. {item['sentence']}" for item in candidates]
    scores = scorer.score(left, right, batch_size=batch_size) if candidates else []
    if len(scores) != len(candidates) or not all(math.isfinite(float(score)) for score in scores):
        raise RuntimeError(f"Invalid sentence scorer output: {claim.uid}")
    ranked = sorted(
        ({**item, "score": float(score)} for item, score in zip(candidates, scores)),
        key=lambda item: (-item["score"], item["document_rank"], item["sentence_id"]),
    )
    per_document: dict[str, int] = {}
    constrained = []
    for item in ranked:
        key = title_key(item["title"])
        if per_document.get(key, 0) >= int(sentences_per_document):
            continue
        per_document[key] = per_document.get(key, 0) + 1
        constrained.append(item)
    return {
        "uid": claim.uid, "num_hops": claim.num_hops,
        "predicted_evidence": constrained[:save_top_k],
        "candidate_sentence_count": len(candidates), "failures": failures,
        "sentence_ids_modified": False, "uses_dev_gold": False,
    }


def oracle_document_selection(
    row: dict[str, Any], *, store: SentenceStore, scorer: PairScorer,
    sentences_per_document: int = 2, save_top_k: int = 12, batch_size: int = 128,
) -> dict[str, Any]:
    """Evaluation-only oracle; never feed this output into Dev production inference."""
    claim = InferenceClaim(str(row["uid"]), str(row["claim"]), int(row["num_hops"]))
    result = select_evidence(
        claim, gold_titles(row), store=store, scorer=scorer,
        document_limit=claim.num_hops,
        sentences_per_document=sentences_per_document,
        save_top_k=save_top_k,
        batch_size=batch_size,
    )
    result["oracle_document_evaluation_only"] = True
    return result

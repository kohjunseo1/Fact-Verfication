from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Iterable

from .data import gold_fact_keys, gold_titles
from .normalization import evidence_key, title_key


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "count": int(numerator),
        "total": int(denominator),
        "fraction": numerator / denominator if denominator else None,
    }


def all_document_hit(predicted: Iterable[str], gold: Iterable[str], k: int | None = None) -> bool:
    values = list(predicted)
    if k is not None:
        values = values[:k]
    predicted_keys = {title_key(title) for title in values}
    return {title_key(title) for title in gold} <= predicted_keys


def document_metrics(
    claims: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    ranked_field: str = "ranked_documents",
    candidate_field: str = "candidate_documents",
    ks: Iterable[int] = (3, 4, 5, 10, 20, 100),
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    scopes = {"all": claims}
    scopes.update({str(hop): [row for row in claims if int(row["num_hops"]) == hop] for hop in (2, 3, 4)})
    for scope, rows in scopes.items():
        hits = Counter()
        candidate_hits = 0
        first_gold = 0
        for row in rows:
            prediction = predictions[str(row["uid"])]
            gold = gold_titles(row)
            ranked = prediction[ranked_field]
            candidates = prediction.get(candidate_field, ranked)
            candidate_hits += int(all_document_hit(candidates, gold))
            first_gold += int(bool({title_key(t) for t in ranked[:1]} & {title_key(t) for t in gold}))
            for k in ks:
                hits[int(k)] += int(all_document_hit(ranked, gold, int(k)))
        output[scope] = {
            "claims": len(rows),
            "candidate_recall": _rate(candidate_hits, len(rows)),
            "first_rank_contains_gold": _rate(first_gold, len(rows)),
            "all_document_hit": {f"hit_at_{k}": _rate(hits[int(k)], len(rows)) for k in ks},
        }
    return output


def evidence_prf(gold: set[tuple[str, int]], predicted: set[tuple[str, int]]) -> dict[str, float | int]:
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else float(not gold)
    recall = true_positive / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "gold": len(gold),
        "predicted": len(predicted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": int(gold == predicted),
    }


def gold_document_fact_coverage(
    gold: set[tuple[str, int]], predicted: set[tuple[str, int]],
) -> bool:
    """Every Gold document needs at least one exact Gold fact, not just its title."""
    gold_documents = {title for title, _ in gold}
    matched = gold & predicted
    return all(any(title == document for title, _ in matched) for document in gold_documents)


def _facts(items: Iterable[Any], limit: int | None = None) -> list[tuple[str, int]]:
    output = []
    for item in items:
        if isinstance(item, dict):
            pair = evidence_key(item["title"], int(item["sentence_id"]))
        else:
            pair = evidence_key(item[0], int(item[1]))
        if pair not in output:
            output.append(pair)
        if limit is not None and len(output) >= limit:
            break
    return output


def official_filtered_evidence(items: Iterable[Any], num_hops: int, submit_k: int | None = None) -> set[tuple[str, int]]:
    """HoVer homepage rule: first k+1 docs, first two sentences per doc."""
    per_doc: dict[str, int] = {}
    kept: set[tuple[str, int]] = set()
    source = list(items)
    if submit_k is not None:
        source = source[:submit_k]
    for title, sentence_id in _facts(source):
        if title not in per_doc:
            if len(per_doc) >= int(num_hops) + 1:
                continue
            per_doc[title] = 0
        if per_doc[title] < 2:
            per_doc[title] += 1
            kept.add((title, sentence_id))
    return kept


def evidence_metrics(
    claims: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], *,
    evidence_field: str = "predicted_evidence", submit_k: int | None = None,
    alignment_bad_uids: set[str] | None = None,
) -> dict[str, Any]:
    bad = alignment_bad_uids or set()
    output: dict[str, Any] = {}
    scopes = {"all": claims}
    scopes.update({str(hop): [row for row in claims if int(row["num_hops"]) == hop] for hop in (2, 3, 4)})
    for scope, rows in scopes.items():
        metrics = []
        coverage = 0
        verified = 0
        for row in rows:
            uid = str(row["uid"])
            gold = gold_fact_keys(row)
            kept = official_filtered_evidence(
                predictions[uid][evidence_field], int(row["num_hops"]), submit_k
            )
            coverage += int(gold_document_fact_coverage(gold, kept))
            if uid not in bad:
                metrics.append(evidence_prf(gold, kept))
                verified += 1
        output[scope] = {
            "claims": len(rows),
            "alignment_verified_claims": verified,
            "alignment_issue_claims": len(rows) - verified,
            "coverage": _rate(coverage, len(rows)),
            "macro_f1_verified": mean(item["f1"] for item in metrics) if metrics else None,
            "exact_verified": mean(item["exact"] for item in metrics) if metrics else None,
        }
    return output


def binary_classification_metrics(gold: list[int], predicted: list[int]) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must be non-empty and aligned")
    per_class = {}
    f1s = []
    for label in (0, 1):
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(g == label for g in gold)}
        f1s.append(f1)
    return {
        "accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(gold),
        "macro_f1": mean(f1s),
        "per_class": per_class,
    }


def hover_scores(
    claims: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], *,
    label_field: str = "predicted_label", evidence_field: str = "predicted_evidence",
    submit_k: int | None = None,
) -> dict[str, Any]:
    label_gold, label_pred = [], []
    hover, evidence_rows = [], []
    for row in claims:
        uid = str(row["uid"])
        prediction = predictions[uid]
        gold_label = 1 if str(row["label"]).upper() == "SUPPORTED" else 0
        pred_label = int(prediction[label_field])
        kept = official_filtered_evidence(prediction[evidence_field], int(row["num_hops"]), submit_k)
        gold = gold_fact_keys(row)
        covered = gold_document_fact_coverage(gold, kept)
        label_gold.append(gold_label)
        label_pred.append(pred_label)
        hover.append(int(covered and pred_label == gold_label))
        evidence_rows.append(evidence_prf(gold, kept))
    classification = binary_classification_metrics(label_gold, label_pred)
    return {
        **classification,
        "hover_score": mean(hover),
        "evidence_f1": mean(item["f1"] for item in evidence_rows),
        "evidence_exact": mean(item["exact"] for item in evidence_rows),
        "claims": len(claims),
    }

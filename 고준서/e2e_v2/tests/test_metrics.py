import pytest

from hover_e2e.metrics import (
    binary_classification_metrics, document_metrics, evidence_metrics, hover_scores,
    official_filtered_evidence,
)


CLAIM = {
    "uid": "u1",
    "claim": "A and B",
    "num_hops": 2,
    "label": "SUPPORTED",
    "supporting_facts": [["A", 0], ["B", 1]],
}


def test_document_all_gold_hit_and_candidate_recall():
    predictions = {
        "u1": {
            "ranked_documents": ["A", "Noise", "B"],
            "candidate_documents": ["A", "Noise", "B", "Other"],
        }
    }
    metrics = document_metrics([CLAIM], predictions, ks=[2, 3])
    assert metrics["all"]["all_document_hit"]["hit_at_2"]["count"] == 0
    assert metrics["all"]["all_document_hit"]["hit_at_3"]["count"] == 1
    assert metrics["all"]["candidate_recall"]["fraction"] == 1


def test_official_filter_first_hop_plus_one_docs_and_two_sentences():
    evidence = [
        {"title": "A", "sentence_id": 0},
        {"title": "A", "sentence_id": 2},
        {"title": "A", "sentence_id": 3},
        {"title": "B", "sentence_id": 1},
        {"title": "C", "sentence_id": 0},
        {"title": "D", "sentence_id": 0},
    ]
    kept = official_filtered_evidence(evidence, num_hops=2)
    assert kept == {("a", 0), ("a", 2), ("b", 1), ("c", 0)}
    scores = hover_scores([CLAIM], {"u1": {
        "predicted_label": 1, "predicted_evidence": evidence,
    }})
    assert scores["hover_score"] == 1
    assert scores["evidence_f1"] == pytest.approx(2 * 2 / (4 + 2))


def test_macro_f1_is_independent_per_class():
    metrics = binary_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_wrong_sentence_ids_do_not_satisfy_hover_or_document_coverage():
    wrong = [
        {"title": "A", "sentence_id": 99},
        {"title": "B", "sentence_id": 98},
    ]
    predictions = {"u1": {"predicted_label": 1, "predicted_evidence": wrong}}
    assert hover_scores([CLAIM], predictions)["hover_score"] == 0
    assert evidence_metrics([CLAIM], predictions)["all"]["coverage"]["fraction"] == 0

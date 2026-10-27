import unicodedata

from hover_e2e.data import (
    exclude_evaluation_group_overlap, group_train_validation_split, inference_view,
)
from hover_e2e.normalization import nfc_title, title_key, unique_titles


def test_nfc_title_deduplicates_nfd_and_nfc():
    nfc = "Ossian Elgström"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc_title(nfd) == nfc
    assert unique_titles([nfd, nfc, "Ossian_Elgström"]) == [nfc]
    assert title_key(nfd) == title_key(nfc)


def test_group_split_has_no_hpqa_leakage_and_is_deterministic():
    rows = [
        {"uid": f"u{i}", "hpqa_id": f"g{i // 2}", "num_hops": 2,
         "claim": "x", "label": "SUPPORTED", "supporting_facts": []}
        for i in range(20)
    ]
    train_a, val_a = group_train_validation_split(rows, fraction=0.2, seed=42)
    train_b, val_b = group_train_validation_split(rows, fraction=0.2, seed=42)
    assert [row["uid"] for row in train_a] == [row["uid"] for row in train_b]
    assert [row["uid"] for row in val_a] == [row["uid"] for row in val_b]
    assert {row["hpqa_id"] for row in train_a}.isdisjoint({row["hpqa_id"] for row in val_a})


def test_inference_view_redacts_all_gold_fields():
    view = inference_view({
        "uid": "u", "claim": "c", "num_hops": 2,
        "label": "SUPPORTED", "supporting_facts": [["Secret", 0]],
    })
    assert view.uid == "u"
    assert not hasattr(view, "label")
    assert not hasattr(view, "supporting_facts")


def test_official_train_variants_overlapping_dev_are_quarantined():
    train = [
        {"uid": "t1", "hpqa_id": "shared"},
        {"uid": "t2", "hpqa_id": "train-only"},
    ]
    dev = [{"uid": "d1", "hpqa_id": "shared"}]
    eligible, quarantined = exclude_evaluation_group_overlap(train, dev)
    assert [row["uid"] for row in eligible] == ["t2"]
    assert [row["uid"] for row in quarantined] == ["t1"]

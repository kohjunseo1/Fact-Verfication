import sqlite3
import pytest

from hover_e2e.data import InferenceClaim
from hover_e2e.evidence import select_evidence
from hover_e2e.metrics import document_metrics, hover_scores
from hover_e2e.multihop import expand_and_rerank
from hover_e2e.retrieval import LexicalPairScorer, rerank_claim
from hover_e2e.sentences import SentenceStore, audit_gold_alignment
from hover_e2e.verifier import build_verifier_training_rows, verify_claim
from hover_e2e.wiki import TitleMatcher, WikiStore, build_title_index


class SupportedScorer:
    def score(self, left, right, *, batch_size=64):
        return [0.9 for _ in left]


class FixedScorer:
    def score(self, left, right, *, batch_size=64):
        return [0.9, 0.8, 0.7, 0.6][:len(left)]


class MemorySentenceStore:
    def get(self, title):
        values = {"A": ["a0", "a1", "a2"], "B": ["b0"]}.get(title)
        if values is None:
            return None
        return {
            "sentences": values,
            "source": "fixture",
            "sentence_array_sha256": title,
        }


def test_sentence_selector_enforces_top_two_per_document():
    prediction = select_evidence(
        InferenceClaim("u", "claim", 2), ["A", "B"],
        store=MemorySentenceStore(), scorer=FixedScorer(),
        sentences_per_document=2, save_top_k=4,
    )
    assert [(item["title"], item["sentence_id"]) for item in prediction["predicted_evidence"]] == [
        ("A", 0), ("A", 1), ("B", 0),
    ]


def test_gold_mix_requires_explicit_gold_text_verification():
    with pytest.raises(ValueError, match="verified Gold-text UIDs"):
        build_verifier_training_rows(
            [], {}, store=MemorySentenceStore(), gold_mix_ratio=0.25,
        )


def test_cpu_mock_pipeline_preserves_ids_and_recovers_second_hop(tmp_path):
    wiki_path = tmp_path / "wiki.db"
    with sqlite3.connect(wiki_path) as connection:
        connection.execute("CREATE TABLE documents(id TEXT PRIMARY KEY,text TEXT NOT NULL)")
        connection.executemany("INSERT INTO documents VALUES (?,?)", [
            ("Alpha", "Alpha connects to Beta."),
            ("Beta", "Beta confirms the relation."),
            ("Noise", "Nothing relevant is here."),
        ])
    cache_path = tmp_path / "sentences.sqlite"
    store = SentenceStore(cache_path, writable=True)
    store.put(
        requested_title="Alpha", database_title="Alpha", raw_text="Alpha connects to Beta.",
        sentences=["Alpha connects to Beta."], offsets=[[0, 23]], updated_utc="now",
    )
    store.put(
        requested_title="Beta", database_title="Beta", raw_text="Beta confirms the relation.",
        sentences=["Beta confirms the relation."], offsets=[[0, 27]], updated_utc="now",
    )
    store.put(
        requested_title="Noise", database_title="Noise", raw_text="Nothing relevant is here.",
        sentences=["Nothing relevant is here."], offsets=[[0, 25]], updated_utc="now",
    )
    store.close()

    claim = InferenceClaim("u1", "Alpha connects to Beta", 2)
    with WikiStore(wiki_path) as wiki:
        index = build_title_index(wiki, tmp_path / "titles.sqlite")
        matcher = TitleMatcher(index)
        base = rerank_claim(
            claim, ["Alpha", "Noise"], wiki=wiki, scorer=LexicalPairScorer()
        )
        assert set(base) >= {"uid", "ranked_documents", "candidate_documents", "scores"}
        expanded = expand_and_rerank(
            claim, base, wiki=wiki, matcher=matcher,
            scorer=SupportedScorer(), seed_docs=1, beam_width=2,
            candidate_limit=1, max_scored_candidates=1, max_scored_per_depth=1,
        )
        matcher.close()
    assert "Beta" in expanded["candidate_documents"]
    candidate_metric = document_metrics([{
        "uid": "u1", "claim": claim.claim, "num_hops": 2,
        "label": "SUPPORTED", "supporting_facts": [["Alpha", 0], ["Beta", 0]],
    }], {"u1": expanded}, ks=[5])
    assert candidate_metric["all"]["candidate_recall"]["fraction"] == 1
    scored = {
        title for path in expanded["paths"] for title in path["documents"]
    }
    assert scored <= set(expanded["candidate_documents"])
    assert expanded["uses_gold_for_generation"] is False
    assert expanded["uses_evidence_for_generation"] is False

    store = SentenceStore(cache_path)
    evidence = select_evidence(
        claim, expanded["ranked_documents"], store=store,
        scorer=SupportedScorer(), document_limit=2, save_top_k=4,
    )
    gold_row = {
        "uid": "u1", "claim": claim.claim, "num_hops": 2,
        "label": "SUPPORTED", "supporting_facts": [["Alpha", 0], ["Beta", 0]],
    }
    with WikiStore(wiki_path) as wiki:
        unverified_audit = audit_gold_alignment([gold_row], store, wiki=wiki)
        audit = audit_gold_alignment(
            [gold_row], store, wiki=wiki,
            gold_texts={
                ("alpha", 0): {"alpha connects to beta."},
                ("beta", 0): {"beta confirms the relation."},
            },
        )
    store.close()
    assert unverified_audit["offset_integrity"]["cached_offsets_verified_against_db"] == 2
    assert unverified_audit["gold_text_alignment"]["verified_facts"] == 0
    assert unverified_audit["gold_text_alignment"]["unverified_facts"] == 2
    assert audit["error_count"] == 0
    assert audit["gold_text_alignment"]["verified_claims"] == 1
    assert {(item["title"], item["sentence_id"]) for item in evidence["predicted_evidence"]} >= {
        ("Alpha", 0), ("Beta", 0),
    }
    assert evidence["sentence_ids_modified"] is False
    assert evidence["uses_dev_gold"] is False

    verification = verify_claim(
        claim, evidence["predicted_evidence"], scorer=SupportedScorer(), retrieved_k=4
    )
    assert verification["uses_dev_gold"] is False
    combined = {"u1": {
        "predicted_label": verification["predicted_label"],
        "predicted_evidence": evidence["predicted_evidence"],
    }}
    assert hover_scores([gold_row], combined)["hover_score"] == 1

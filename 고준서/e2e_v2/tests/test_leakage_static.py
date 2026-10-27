import inspect

from hover_e2e.evidence import select_evidence
from hover_e2e.multihop import expand_and_rerank, expansion_candidates
from hover_e2e.retrieval import rerank_claim
from hover_e2e.verifier import verify_claim


def test_dev_inference_entrypoints_have_no_gold_arguments():
    forbidden = {"gold", "supporting_facts", "oracle", "label"}
    for function in (rerank_claim, expansion_candidates, expand_and_rerank, select_evidence, verify_claim):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & forbidden, (function.__name__, parameters & forbidden)
        source = inspect.getsource(function)
        assert "supporting_facts" not in source

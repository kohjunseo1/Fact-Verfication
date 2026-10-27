import pytest

from hover_e2e.io_utils import JsonlCheckpoint, index_by_uid, iter_jsonl
from hover_e2e.stages import _prepare_model_contract


def test_append_only_checkpoint_resumes_and_rejects_config_mix(tmp_path):
    with JsonlCheckpoint(tmp_path / "stage", {"a": 1}, flush_every=1) as checkpoint:
        checkpoint.append({"uid": "u1", "value": 1})
    with JsonlCheckpoint(tmp_path / "stage", {"a": 1}) as checkpoint:
        assert checkpoint.is_completed("u1")
        assert not checkpoint.is_completed("u2")
        assert checkpoint.completed_count == 1
        checkpoint.append({"uid": "u2", "value": 2})
    assert list(index_by_uid(iter_jsonl(tmp_path / "stage/predictions.jsonl"))) == ["u1", "u2"]
    with pytest.raises(RuntimeError, match="config mismatch"):
        JsonlCheckpoint(tmp_path / "stage", {"a": 2})


def test_uid_contract_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate uid"):
        index_by_uid([{"uid": "x"}, {"uid": "x"}])


def test_model_contract_rejects_changed_training_cohort(tmp_path):
    stage = tmp_path / "retrieval"
    _prepare_model_contract(stage, {"model_name": "x"}, 42, [{"uid": "t1"}], [{"uid": "v1"}])
    _prepare_model_contract(stage, {"model_name": "x"}, 42, [{"uid": "t1"}], [{"uid": "v1"}])
    with pytest.raises(RuntimeError, match="Model contract mismatch"):
        _prepare_model_contract(
            stage, {"model_name": "x"}, 42, [{"uid": "different"}], [{"uid": "v1"}]
        )

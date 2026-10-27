import json
import base64
import io
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import zipfile

from hover_e2e.sentences import CoreNLPHTTP
from hover_e2e.colab_runtime import BASE_FILES, discover_e2e_root, prepare_runtime


def test_discover_e2e_root_from_notebook_directory(tmp_path, monkeypatch):
    root = tmp_path / "고준서/e2e_v2"
    (root / "src/hover_e2e").mkdir(parents=True)
    (root / "notebooks").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    monkeypatch.delenv("HOVER_E2E_ROOT", raising=False)
    assert discover_e2e_root(root / "notebooks") == root.resolve()


def test_prepare_runtime_reuses_repository_local_data(tmp_path, monkeypatch):
    e2e_root = tmp_path / "고준서/e2e_v2"
    e2e_root.mkdir(parents=True)
    data_root = e2e_root.parent / "data"
    for relative in BASE_FILES:
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    monkeypatch.delenv("HOVER_DATA_ROOT", raising=False)
    monkeypatch.delenv("HOVER_E2E_OUTPUT_ROOT", raising=False)
    context = prepare_runtime(e2e_root, run_name="test")
    assert context.data_root == data_root.resolve()
    assert context.output_root == (e2e_root / "runs").resolve()


def test_all_notebooks_are_clean_and_code_cells_compile():
    notebook_root = Path(__file__).resolve().parents[1] / "notebooks"
    expected = [
        "00_preflight.ipynb", "01_retrieval.ipynb", "02_multihop.ipynb",
        "03_evidence.ipynb", "04_verifier.ipynb", "05_e2e.ipynb", "run_all.ipynb",
    ]
    assert [path.name for path in sorted(notebook_root.glob("*.ipynb"))] == expected
    for path in sorted(notebook_root.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        ids = set()
        for cell in notebook["cells"]:
            assert cell["id"] not in ids
            ids.add(cell["id"])
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                compile("".join(cell["source"]), f"{path}:{cell['id']}", "exec")
        bootstrap = "".join(notebook["cells"][1]["source"])
        assert "HOVER_E2E_ROOT" in bootstrap
        assert "prepare_runtime" in bootstrap
        assert bootstrap.index("if E2E_ROOT is None") < bootstrap.index("drive.mount")


def test_run_all_stage_order_is_explicit():
    path = Path(__file__).resolve().parents[1] / "notebooks/run_all.ipynb"
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in json.loads(path.read_text())["cells"]
    )
    first_retrieval = source.index("for stage in ('retrieval','multihop')")
    corenlp_start = source.index("CORENLP_URL,CORENLP_PROCESS=start_corenlp", first_retrieval)
    evidence = source.index("run_stage('evidence'", corenlp_start)
    corenlp_stop = source.index("CORENLP_PROCESS.terminate()", evidence)
    verifier = source.index("for stage in ('verifier','evaluate')", corenlp_stop)
    assert first_retrieval < corenlp_start < evidence < corenlp_stop < verifier
    assert source.index("'preflight'") < source.index("'retrieval'")
    assert source.index("run_stage(stage,'2')") < source.index("run_stage(stage,'2,3,4')")
    assert "RUN_3_4_HOP=False" in source
    assert "if not RUN_3_4_HOP" in source


def test_colab_entry_notebooks_embed_runnable_source_bundle():
    notebook_root = Path(__file__).resolve().parents[1] / "notebooks"
    for name in ("00_preflight.ipynb", "run_all.ipynb"):
        notebook = json.loads((notebook_root / name).read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][1]["source"])
        match = re.search(r"BUNDLE_B64\s*=\s*'([^']+)'", source)
        assert match, name
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(match.group(1)))) as archive:
            names = set(archive.namelist())
        assert "e2e_v2/pyproject.toml" in names
        assert "e2e_v2/configs/base.json" in names
        assert "e2e_v2/src/hover_e2e/cli.py" in names


def test_corenlp_utf16_offsets_are_converted_to_python(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"sentences": [
                {"tokens": [
                    {"characterOffsetBegin": 0, "characterOffsetEnd": 1},
                    {"characterOffsetBegin": 1, "characterOffsetEnd": 4},
                ]},
                {"tokens": [
                    {"characterOffsetBegin": 5, "characterOffsetEnd": 9},
                    {"characterOffsetBegin": 9, "characterOffsetEnd": 10},
                ]},
            ]}

    fake_requests = SimpleNamespace(post=lambda *args, **kwargs: Response())
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    sentences, offsets = CoreNLPHTTP().split("A😀. Next.")
    assert sentences == ["A😀.", "Next."]
    assert offsets == [[0, 3], [4, 9]]

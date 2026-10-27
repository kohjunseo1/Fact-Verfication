from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


@dataclass(frozen=True)
class ColabContext:
    e2e_root: Path
    drive_project: Path
    data_root: Path
    output_root: Path
    run_name: str


BASE_FILES = {
    "raw/hover_train_release_v1.1.json": "data/raw/hover_train_release_v1.1.json",
    "raw/hover_dev_release_v1.1.json": "data/raw/hover_dev_release_v1.1.json",
    "raw/wiki_wo_links.db": "data/raw/wiki_wo_links.db",
    "retrieval/train_tfidf_doc_retrieval_results.json": "data/raw/train_tfidf_doc_retrieval_results.json",
    "retrieval/dev_tfidf_doc_retrieval_results.json": "data/raw/dev_tfidf_doc_retrieval_results.json",
}

BASE_MANIFEST_KEYS = {
    "raw/hover_train_release_v1.1.json": "train_claims",
    "raw/hover_dev_release_v1.1.json": "dev_claims",
    "raw/wiki_wo_links.db": "wiki_db",
    "retrieval/train_tfidf_doc_retrieval_results.json": "train_tfidf",
    "retrieval/dev_tfidf_doc_retrieval_results.json": "dev_tfidf",
}


def _base_data_complete(data_root: Path) -> bool:
    return all(
        (data_root / relative).is_file()
        and (data_root / relative).stat().st_size > 0
        for relative in BASE_FILES
    )


def discover_e2e_root(start: str | Path | None = None) -> Path:
    """Find e2e_v2 across repository, uploaded Colab, and explicit layouts."""
    bases: list[Path] = []
    if os.environ.get("HOVER_E2E_ROOT"):
        bases.append(Path(os.environ["HOVER_E2E_ROOT"]).expanduser())
    current = Path(start or Path.cwd()).expanduser()
    bases.extend([current, *current.parents])
    bases.extend([
        Path("/content/e2e_v2"),
        Path("/content/고준서/e2e_v2"),
        Path("/content/Fact-Verfication/고준서/e2e_v2"),
        Path("/content/drive/MyDrive/hover-project/고준서/e2e_v2"),
    ])
    candidates: list[Path] = []
    for base in bases:
        candidates.extend([base, base / "e2e_v2", base / "고준서/e2e_v2"])
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/hover_e2e").is_dir():
            return candidate
    checked = "\n".join(f"  - {path}" for path in dict.fromkeys(candidates))
    raise FileNotFoundError(
        "e2e_v2 source directory is not visible in this kernel. Checked:\n"
        f"{checked}\n"
        "A remote Colab kernel cannot read /Users/... on the Mac. Upload the repository "
        "under /content, set HOVER_E2E_ROOT, or mount the location containing the code."
    )


def prepare_runtime(
    e2e_root: str | Path,
    *,
    drive_project: str | Path | None = None,
    run_name: str = "roberta_e2e_v2",
) -> ColabContext:
    """Use 고준서/data directly locally; copy from Drive only inside Colab."""
    root = Path(e2e_root).resolve()
    data_candidates = []
    if os.environ.get("HOVER_DATA_ROOT"):
        data_candidates.append(Path(os.environ["HOVER_DATA_ROOT"]).expanduser())
    data_candidates.extend([
        root.parent / "data",
        Path("/content/hover_data"),
        Path("/content/고준서/data"),
        Path("/content/Fact-Verfication/고준서/data"),
    ])
    repository_data = next(
        (path.resolve() for path in data_candidates if _base_data_complete(path.resolve())),
        (root.parent / "data").resolve(),
    )
    drive = Path(drive_project).resolve() if drive_project else None
    repository_data_is_drive = str(repository_data).startswith(
        ("/content/drive/", "/content/gdrive/")
    )
    if _base_data_complete(repository_data) and not repository_data_is_drive:
        output = Path(
            os.environ.get(
                "HOVER_E2E_OUTPUT_ROOT",
                str(drive / "고준서/e2e_v2_runs") if drive and drive.is_dir()
                else str(root / "runs"),
            )
        ).resolve()
        output.mkdir(parents=True, exist_ok=True)
        context_root = drive if drive and drive.is_dir() else root.parents[1]
        os.environ["HOVER_DATA_ROOT"] = str(repository_data)
        os.environ["HOVER_E2E_OUTPUT_ROOT"] = str(output)
        print("[runtime] reuse repository-local HoVer data:", repository_data, flush=True)
        print("[runtime] outputs:", output, flush=True)
        return ColabContext(root, context_root, repository_data, output, run_name)
    if drive is not None and drive.is_dir():
        return prepare_colab(root, drive_project=drive, run_name=run_name)
    raise FileNotFoundError(
        f"Local HoVer data missing under {repository_data}. "
        "Expected raw/ and retrieval/ inputs; Drive fallback is unavailable."
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        print("[runtime] reuse:", destination, flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()
    print(f"[runtime] copy: {source} -> {destination} ({source.stat().st_size / 1e9:.2f}GB)", flush=True)
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != source.stat().st_size:
        raise IOError(f"Copy size mismatch: {source} -> {temporary}")
    temporary.replace(destination)


def prepare_colab(
    e2e_root: str | Path,
    *,
    drive_project: str | Path = "/content/drive/MyDrive/hover-project",
    data_root: str | Path = "/content/hover_data",
    run_name: str = "roberta_e2e_v2",
) -> ColabContext:
    root = Path(e2e_root).resolve()
    drive = Path(drive_project).resolve()
    local = Path(data_root).resolve()
    output = drive / "고준서" / "e2e_v2_runs"
    if not drive.is_dir():
        raise FileNotFoundError(f"Drive project not mounted: {drive}")
    manifest_path = drive / "baseline_v2/config/data_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else {}
    )
    for relative, source_relative in BASE_FILES.items():
        manifest_item = manifest.get("files", {}).get(BASE_MANIFEST_KEYS[relative], {})
        manifest_relative = manifest_item.get("relative_path")
        source_candidates = [
            root.parent / "data" / relative,
            drive / "고준서/data" / relative,
            drive / source_relative,
            drive / "data" / Path(source_relative).name,
            drive / "data/hover/tfidf_retrieved" / Path(source_relative).name,
            drive / "data/processed" / Path(source_relative).name,
            drive / "dataset" / Path(source_relative).name,
            drive / "baseline_v2/artifacts" / Path(source_relative).name,
        ]
        if manifest_relative:
            source_candidates.insert(0, drive / manifest_relative)
        source = next((path for path in source_candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"Drive base input missing: {Path(source_relative).name}. "
                "Run 고준서/notebooks/00a_local_data_setup.ipynb first."
            )
        _copy_atomic(source, local / relative)

    shared = output / run_name / "shared"
    artifact_candidates = {
        "artifacts/02_corenlp_sentence_cache_v2.sqlite": [
            shared / "02_corenlp_sentence_cache_v2.sqlite",
            drive / "고준서/artifacts/02_corenlp_sentence_cache_v2.sqlite",
        ],
        "artifacts/02_verified_reference_overrides_offline.json": [
            shared / "02_verified_reference_overrides_offline.json",
            drive / "고준서/artifacts/02_verified_reference_overrides_offline.json",
        ],
        "artifacts/02_sentence_alignment_diagnostics.json": [
            drive / "고준서/results/02_sentence_alignment_diagnostics.json",
        ],
        "artifacts/02_reference_candidate_audit.json": [
            drive / "고준서/results/02_reference_candidate_audit.json",
        ],
        "artifacts/02_gold_sentence_alignment.jsonl": [
            drive / "baseline_v2/artifacts/02_gold_sentence_alignment.jsonl",
        ],
        "artifacts/wiki_title_index.sqlite": [
            shared / "wiki_title_index.sqlite",
            drive / "고준서/data/artifacts/wiki_title_index.sqlite",
        ],
    }
    for relative, candidates in artifact_candidates.items():
        source = next((path for path in candidates if path.is_file()), None)
        if source is not None:
            _copy_atomic(source, local / relative)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HOVER_DATA_ROOT"] = str(local)
    os.environ["HOVER_E2E_OUTPUT_ROOT"] = str(output)
    print(f"[runtime] data ready: {local}", flush=True)
    print(f"[runtime] outputs: {output}", flush=True)
    return ColabContext(root, drive, local, output, run_name)


def backup_shared_artifacts(context: ColabContext) -> None:
    print("[runtime] backing up shared artifacts", flush=True)
    shared = context.output_root / context.run_name / "shared"
    for name in (
        "wiki_title_index.sqlite",
        "02_corenlp_sentence_cache_v2.sqlite",
        "02_verified_reference_overrides_offline.json",
    ):
        source = context.data_root / "artifacts" / name
        if source.is_file():
            _copy_atomic(source, shared / name)
    print("[runtime] shared artifact backup complete", flush=True)


def _corenlp_ready(url: str) -> bool:
    try:
        import requests
        response = requests.post(
            url,
            params={"properties": json.dumps({
                "annotators": "tokenize,ssplit", "pipelineLanguage": "en", "outputFormat": "json"
            })},
            data=b"CoreNLP readiness check.",
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=8,
        )
        response.raise_for_status()
        return bool(response.json().get("sentences"))
    except Exception:
        return False


def start_corenlp(context: ColabContext) -> tuple[str, subprocess.Popen | None]:
    existing_url = os.environ.get("CORENLP_URL", "http://127.0.0.1:9000").rstrip("/")
    if _corenlp_ready(existing_url):
        print("[corenlp] reuse ready server:", existing_url, flush=True)
        return existing_url, None
    candidates = []
    if os.environ.get("CORENLP_HOME"):
        candidates.append(Path(os.environ["CORENLP_HOME"]))
    candidates += [
        context.drive_project / "tools/stanford-corenlp",
        context.drive_project / "data/tools/stanford-corenlp",
        Path("/content/stanford-corenlp"),
    ]
    for parent in (context.drive_project / "tools", context.drive_project / "data/tools", Path("/content")):
        if parent.is_dir():
            candidates.extend(sorted(parent.glob("stanford-corenlp*")))
    home = next((path for path in candidates if path.is_dir() and any(path.glob("stanford-corenlp-*.jar"))), None)
    if home is None:
        import stanza
        home = context.drive_project / "tools/stanford-corenlp"
        home.parent.mkdir(parents=True, exist_ok=True)
        print("CoreNLP not cached; one-time official download:", home, flush=True)
        stanza.install_corenlp(dir=str(home))
        if not any(home.glob("stanford-corenlp-*.jar")):
            raise RuntimeError(f"CoreNLP installation has no JAR: {home}")
    if shutil.which("java") is None:
        raise RuntimeError("Java is unavailable; cannot start CoreNLP")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_root = Path("/content") if Path("/content").is_dir() else Path("/tmp")
    log_path = log_root / f"hover_corenlp_{port}.log"
    log_stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            "java", "-mx4g", "-cp", "*",
            "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
            "-port", str(port), "-timeout", "30000",
        ],
        cwd=home,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        if process.poll() is not None:
            log_stream.close()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"CoreNLP exited early ({process.returncode}):\n{tail}")
        if _corenlp_ready(url):
            os.environ["CORENLP_URL"] = url
            print("[corenlp] ready:", url, flush=True)
            return url, process
        time.sleep(2)
    process.terminate()
    log_stream.close()
    raise TimeoutError(f"CoreNLP was not ready within 120 seconds; log={log_path}")

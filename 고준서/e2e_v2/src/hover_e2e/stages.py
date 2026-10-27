from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

from .config import Settings, require_paths
from .data import (
    dataset_summary, exclude_evaluation_group_overlap, gold_fact_keys, gold_titles, group_id,
    group_train_validation_split, inference_view, load_claims, load_tfidf,
)
from .evidence import (
    build_sentence_training_rows, oracle_document_selection, select_evidence,
)
from .io_utils import (
    JsonlCheckpoint, atomic_write_json, fingerprint, index_by_uid, iter_jsonl,
    read_json, write_jsonl_atomic,
)
from .metrics import (
    document_metrics, evidence_metrics, gold_document_fact_coverage, hover_scores,
    official_filtered_evidence,
)
from .modeling import HFPairScorer, require_cuda_runtime, train_pair_classifier
from .multihop import build_multihop_training_rows, expand_and_rerank
from .normalization import title_key
from .retrieval import build_reranker_training_rows, rerank_claim
from .sentences import (
    CoreNLPHTTP, SentenceStore, audit_gold_alignment, load_gold_text_references,
)
from .verifier import build_verifier_training_rows, label_id, verify_claim
from .wiki import TitleMatcher, WikiStore, build_title_index


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pending_rows(
    rows: Iterable[dict[str, Any]], checkpoint: JsonlCheckpoint, label: str,
) -> list[dict[str, Any]]:
    materialized = list(rows)
    pending = [row for row in materialized if not checkpoint.is_completed(str(row["uid"]))]
    print(
        f"[{label}] total={len(materialized):,} completed={len(materialized)-len(pending):,} "
        f"pending={len(pending):,}",
        flush=True,
    )
    return pending


def _log_progress(label: str, index: int, total: int, started: float) -> None:
    if not total:
        return
    interval = max(1, min(100, total // 50))
    if index != 1 and index != total and index % interval:
        return
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = index / elapsed
    eta = (total - index) / rate if rate else 0.0
    print(
        f"[{label}] {index:,}/{total:,} ({index/total:.1%}) "
        f"rate={rate:.2f}/s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
        flush=True,
    )


def _base_inputs(settings: Settings):
    started = time.perf_counter()
    print(f"[data] validating inputs under {settings.data_root}", flush=True)
    paths = require_paths(
        settings, ["train_claims", "dev_claims", "train_tfidf", "dev_tfidf", "wiki_db"]
    )
    train = load_claims(paths["train_claims"], expected_count=18_171)
    dev = load_claims(paths["dev_claims"], expected_count=4_000)
    train_tfidf = load_tfidf(paths["train_tfidf"], expected_uids=[row["uid"] for row in train])
    dev_tfidf = load_tfidf(paths["dev_tfidf"], expected_uids=[row["uid"] for row in dev])
    print(
        f"[data] ready train={len(train):,} dev={len(dev):,} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return paths, train, dev, train_tfidf, dev_tfidf


def _split_rows(
    settings: Settings,
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
):
    eligible, quarantined = exclude_evaluation_group_overlap(train, dev)
    subtrain, validation = group_train_validation_split(
        eligible,
        fraction=float(settings.raw["split"]["validation_fraction"]),
        seed=settings.seed,
    )
    return subtrain, validation, quarantined


def _filter_hops(rows: Iterable[dict[str, Any]], hops: set[int]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["num_hops"]) in hops]


def _stage_config(settings: Settings, stage: str) -> dict[str, Any]:
    return {"schema_version": 3, "stage": stage, "seed": settings.seed, "config": settings.raw[stage]}


def _prepare_model_contract(
    stage_dir: Path,
    config: dict[str, Any],
    seed: int,
    train_claims: list[dict[str, Any]],
    validation_claims: list[dict[str, Any]],
) -> None:
    """Bind a reusable model/checkpoint directory to config and exact claim cohorts."""
    payload = {
        "schema_version": 1,
        "pipeline_schema_version": 3,
        "seed": int(seed),
        "config": config,
        "train_uids_sha256": fingerprint([str(row["uid"]) for row in train_claims]),
        "validation_uids_sha256": fingerprint(
            [str(row["uid"]) for row in validation_claims]
        ),
    }
    payload["contract_sha256"] = fingerprint(payload)
    contract_path = stage_dir / "model_contract.json"
    if contract_path.is_file():
        existing = read_json(contract_path)
        if existing != payload:
            raise RuntimeError(
                f"Model contract mismatch in {stage_dir}; use a new run name"
            )
        return
    model_dir = stage_dir / "model"
    if model_dir.is_dir() and any(model_dir.iterdir()):
        raise RuntimeError(
            f"Existing model has no v2 contract in {stage_dir}; use a new run name"
        )
    atomic_write_json(contract_path, payload)


def _records(run_dir: Path, stage: str, split: str) -> dict[str, dict[str, Any]]:
    path = run_dir / stage / split / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Required previous-stage predictions missing: {path}")
    return index_by_uid(iter_jsonl(path))


def _historical_alignment_uids(settings: Settings) -> set[str]:
    path = settings.data_path("reference_alignment_audit")
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(uid) for uid in payload.get("unverified_uids", [])}


def _gold_text_references(settings: Settings) -> dict[tuple[str, int], set[str]]:
    return load_gold_text_references(settings.data_path("gold_sentence_reference"))


def preflight(settings: Settings, run_name: str, *, build_titles: bool = True) -> dict[str, Any]:
    paths, train, dev, train_tfidf, dev_tfidf = _base_inputs(settings)
    expected = {
        int(key): int(value)
        for key, value in settings.raw["evaluation"]["expected_dev_by_hop"].items()
    }
    actual = Counter(int(row["num_hops"]) for row in dev)
    if dict(actual) != expected:
        raise ValueError(f"Dev hop counts mismatch: {dict(actual)} != {expected}")
    subtrain, validation, quarantined = _split_rows(settings, train, dev)
    subtrain_groups = {group_id(row) for row in subtrain}
    validation_groups = {group_id(row) for row in validation}
    dev_groups = {group_id(row) for row in dev}
    if subtrain_groups & validation_groups:
        raise AssertionError("Train/Validation hpqa_id leakage")

    stage_dir = settings.stage_dir(run_name, "00_preflight")
    stage_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(stage_dir / "split_manifest.json", {
        "schema_version": 1,
        "seed": settings.seed,
        "group_key": settings.raw["split"]["group_key"],
        "subtrain_uids": [str(row["uid"]) for row in subtrain],
        "validation_uids": [str(row["uid"]) for row in validation],
        "quarantined_train_uids": [str(row["uid"]) for row in quarantined],
        "quarantined_hpqa_ids": sorted({group_id(row) for row in quarantined}),
        "quarantine_reason": "hpqa_id also present in official Dev",
    })

    title_index_path = settings.data_path("title_index")
    with WikiStore(paths["wiki_db"]) as wiki:
        if build_titles:
            build_title_index(wiki, title_index_path)

    sentence_cache = settings.data_path("sentence_cache")
    overrides = settings.data_path("sentence_overrides")
    alignment = None
    if sentence_cache.is_file():
        store = SentenceStore(sentence_cache, overrides_path=overrides)
        try:
            with WikiStore(paths["wiki_db"]) as audit_wiki:
                alignment = audit_gold_alignment(
                    dev, store, wiki=audit_wiki,
                    gold_texts=_gold_text_references(settings),
                )
        finally:
            store.close()
        atomic_write_json(stage_dir / "alignment_audit.json", alignment)
    historical_alignment_uids = _historical_alignment_uids(settings)

    report = {
        "status": "READY_FOR_COLAB" if sentence_cache.is_file() else "NEEDS_COLAB_RUN",
        "created_utc": utc_now(),
        "inputs": {
            key: {"path": str(path), "bytes": path.stat().st_size}
            for key, path in paths.items()
        },
        "train": dataset_summary(train),
        "eligible_train_after_dev_group_quarantine": dataset_summary(subtrain + validation),
        "quarantined_train_dev_overlap": dataset_summary(quarantined),
        "subtrain": dataset_summary(subtrain),
        "validation": dataset_summary(validation),
        "dev": dataset_summary(dev),
        "tfidf": {"train_uids": len(train_tfidf), "dev_uids": len(dev_tfidf)},
        "title_index": {"path": str(title_index_path), "exists": title_index_path.is_file()},
        "sentence_cache": {"path": str(sentence_cache), "exists": sentence_cache.is_file()},
        "alignment": None if alignment is None else {
            "affected_claims": alignment["affected_claims"],
            "historical_unverified_claims": len(historical_alignment_uids),
            "expected_known_issue_claims": settings.raw["evidence"]["known_alignment_issue_claims"],
            "note": "Reported separately; no sentence ID mutation or validation bypass.",
        },
        "leakage_checks": {
            "raw_train_dev_hpqa_group_overlap": len({group_id(row) for row in quarantined}),
            "quarantined_train_claims": len(quarantined),
            "effective_train_dev_hpqa_group_overlap": len(
                (subtrain_groups | validation_groups) & dev_groups
            ),
            "dev_used_for_model_selection": False,
            "dev_gold_used_for_candidate_generation": False,
        },
    }
    atomic_write_json(stage_dir / "report.json", report)
    return report


def _train_or_reuse(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    stage_dir: Path,
    config: dict[str, Any],
    seed: int,
    class_weights=None,
) -> Path:
    model_path = stage_dir / "model" / "best_model"
    if (model_path / "config.json").is_file():
        print(f"[train] reuse best model: {model_path}", flush=True)
        return model_path
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Pair text can be several GB. It is deterministic from source/config, so do
    # not push it through Drive. Persist only a contract summary and small sample.
    atomic_write_json(stage_dir / "training_pairs_summary.json", {
        "train_pairs": len(train_rows),
        "validation_pairs": len(validation_rows),
        "train_positive": sum(int(row["label"]) for row in train_rows),
        "validation_positive": sum(int(row["label"]) for row in validation_rows),
        "train_sample": train_rows[:3],
        "validation_sample": validation_rows[:3],
    })
    print(
        f"[train] start model={config['model_name']} train_pairs={len(train_rows):,} "
        f"validation_pairs={len(validation_rows):,} epochs={config['epochs']}",
        flush=True,
    )
    train_pair_classifier(
        train_rows,
        validation_rows,
        output_dir=stage_dir / "model",
        model_name=config["model_name"],
        max_length=int(config["max_length"]),
        epochs=float(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        train_batch_size=int(config["train_batch_size"]),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        eval_batch_size=int(config["eval_batch_size"]),
        seed=seed,
        class_weights=class_weights,
    )
    print(f"[train] complete best_model={model_path}", flush=True)
    return model_path


def retrieval_stage(settings: Settings, run_name: str, *, hops: set[int]) -> dict[str, Any]:
    print("CUDA:", require_cuda_runtime(), flush=True)
    paths, train, dev, train_tfidf, dev_tfidf = _base_inputs(settings)
    subtrain, validation, _ = _split_rows(settings, train, dev)
    eligible_train = subtrain + validation
    config = settings.raw["retrieval"]
    stage_dir = settings.stage_dir(run_name, "01_retrieval")
    _prepare_model_contract(stage_dir, config, settings.seed, subtrain, validation)
    with WikiStore(paths["wiki_db"]) as wiki:
        model_path = stage_dir / "model" / "best_model"
        if not (model_path / "config.json").is_file():
            train_pairs = build_reranker_training_rows(
                subtrain, train_tfidf, wiki=wiki,
                hard_negatives=int(config["train_hard_negatives"]),
            )
            validation_pairs = build_reranker_training_rows(
                validation, train_tfidf, wiki=wiki,
                hard_negatives=int(config["train_hard_negatives"]),
            )
            model_path = _train_or_reuse(
                train_pairs, validation_pairs, stage_dir=stage_dir,
                config=config, seed=settings.seed,
            )
        scorer = HFPairScorer(model_path, max_length=int(config["max_length"]))
        for split, rows, ranks in (
            ("train", train, train_tfidf),
            ("dev", dev, dev_tfidf),
        ):
            selected = eligible_train if split == "train" else _filter_hops(rows, hops)
            checkpoint_config = {**_stage_config(settings, "retrieval"), "split": split}
            with JsonlCheckpoint(stage_dir / split, checkpoint_config) as checkpoint:
                label = f"retrieval:{split}"
                pending = _pending_rows(selected, checkpoint, label)
                started = time.perf_counter()
                for index, row in enumerate(pending, 1):
                    uid = str(row["uid"])
                    prediction = rerank_claim(
                        inference_view(row),
                        ranks[uid][:int(config["tfidf_k"])],
                        wiki=wiki,
                        scorer=scorer,
                        batch_size=int(config["eval_batch_size"]),
                    )
                    if split == "train":
                        prediction.pop("tfidf_top100", None)
                        prediction.pop("candidate_documents", None)
                    checkpoint.append(prediction)
                    _log_progress(label, index, len(pending), started)
    dev_predictions = _records(settings.output_root / run_name, "01_retrieval", "dev")
    evaluated = _filter_hops(dev, hops)
    metrics = document_metrics(
        evaluated, dev_predictions, ks=settings.raw["evaluation"]["document_ks"]
    )
    atomic_write_json(stage_dir / "metrics.json", metrics)
    return metrics


def multihop_stage(settings: Settings, run_name: str, *, hops: set[int]) -> dict[str, Any]:
    print("CUDA:", require_cuda_runtime(), flush=True)
    paths, train, dev, _, _ = _base_inputs(settings)
    subtrain, validation, _ = _split_rows(settings, train, dev)
    eligible_train = subtrain + validation
    config = settings.raw["multihop"]
    run_dir = settings.output_root / run_name
    stage_dir = settings.stage_dir(run_name, "02_multihop")
    _prepare_model_contract(stage_dir, config, settings.seed, subtrain, validation)
    base_train = _records(run_dir, "01_retrieval", "train")
    base_dev = _records(run_dir, "01_retrieval", "dev")
    with WikiStore(paths["wiki_db"]) as wiki:
        title_index = build_title_index(wiki, settings.data_path("title_index"))
        matcher = TitleMatcher(
            title_index,
            max_ngram=int(config["max_title_ngram"]),
            max_titles_per_surface=int(config["max_titles_per_surface"]),
            max_text_chars=int(config["source_max_chars"]),
        )
        try:
            model_path = stage_dir / "model" / "best_model"
            if not (model_path / "config.json").is_file():
                train_pairs = build_multihop_training_rows(
                    subtrain, base_train, wiki=wiki, matcher=matcher,
                    hard_negatives=int(config["train_hard_negatives"]),
                    candidate_limit=int(config["candidate_limit"]),
                )
                validation_pairs = build_multihop_training_rows(
                    validation, base_train, wiki=wiki, matcher=matcher,
                    hard_negatives=int(config["train_hard_negatives"]),
                    candidate_limit=int(config["candidate_limit"]),
                )
                model_path = _train_or_reuse(
                    train_pairs, validation_pairs, stage_dir=stage_dir,
                    config=config, seed=settings.seed,
                )
            scorer = HFPairScorer(model_path, max_length=int(config["max_length"]))
            for split, rows, base in (
                ("train", train, base_train),
                ("dev", dev, base_dev),
            ):
                selected = eligible_train if split == "train" else _filter_hops(rows, hops)
                checkpoint_config = {**_stage_config(settings, "multihop"), "split": split}
                with JsonlCheckpoint(stage_dir / split, checkpoint_config) as checkpoint:
                    label = f"multihop:{split}"
                    pending = _pending_rows(selected, checkpoint, label)
                    started = time.perf_counter()
                    for index, row in enumerate(pending, 1):
                        uid = str(row["uid"])
                        prediction = expand_and_rerank(
                            inference_view(row), base[uid], wiki=wiki,
                            matcher=matcher, scorer=scorer,
                            seed_docs=int(config["seed_docs"]),
                            beam_width=int(config["beam_width"]),
                            candidate_limit=int(config["candidate_limit"]),
                            max_scored_candidates=int(config["max_scored_candidates"]),
                            max_scored_per_depth=int(config["max_scored_per_depth"]),
                            score_alpha=float(config["path_score_alpha"]),
                            batch_size=int(config["eval_batch_size"]),
                        )
                        if split == "train":
                            prediction.pop("candidate_documents", None)
                            prediction.pop("paths", None)
                            prediction.pop("trace", None)
                        checkpoint.append(prediction)
                        _log_progress(label, index, len(pending), started)
        finally:
            matcher.close()
    dev_predictions = _records(run_dir, "02_multihop", "dev")
    evaluated = _filter_hops(dev, hops)
    runtime_rows = [
        {"uid": str(row["uid"]), **dev_predictions[str(row["uid"])]["runtime"]}
        for row in evaluated
        if "runtime" in dev_predictions[str(row["uid"])]
    ]
    elapsed_total = sum(float(row["elapsed_seconds"]) for row in runtime_rows)
    scored_total = sum(int(row["scored_instances"]) for row in runtime_rows)
    sample_size = min(int(config["runtime_sample_size"]), len(runtime_rows))
    runtime_report = {
        "claims": len(runtime_rows),
        "elapsed_seconds_sum": elapsed_total,
        "claims_per_second": len(runtime_rows) / elapsed_total if elapsed_total else None,
        "scored_instances": scored_total,
        "scored_instances_per_second": scored_total / elapsed_total if elapsed_total else None,
        "mean_unique_candidate_documents": (
            sum(int(row["unique_candidate_documents"]) for row in runtime_rows) / len(runtime_rows)
            if runtime_rows else None
        ),
        "max_process_peak_rss_mib": (
            max(float(row["process_peak_rss_mib"]) for row in runtime_rows)
            if runtime_rows else None
        ),
        "budget_exhausted_claims": sum(bool(row["budget_exhausted"]) for row in runtime_rows),
        "first_runtime_sample": runtime_rows[:sample_size],
        "sample_size": sample_size,
        "note": "Per-claim elapsed sums exclude model load; RSS is process peak, not incremental allocation.",
    }
    atomic_write_json(stage_dir / "runtime_metrics.json", runtime_report)
    metrics = document_metrics(
        evaluated, dev_predictions, ks=settings.raw["evaluation"]["document_ks"]
    )
    metrics["runtime"] = runtime_report
    atomic_write_json(stage_dir / "metrics.json", metrics)
    return metrics


def _ensure_sentence_arrays(
    titles: Iterable[str],
    *,
    store: SentenceStore,
    wiki: WikiStore,
    corenlp: CoreNLPHTTP,
    mirror_path: Path | None = None,
    backup_every: int = 5_000,
) -> list[dict[str, Any]]:
    unique_titles = list(dict.fromkeys(titles))
    failures: list[dict[str, Any]] = []
    cached = 0
    created = 0
    started = time.perf_counter()
    print(f"[sentences:cache] checking titles={len(unique_titles):,}", flush=True)
    for index, title in enumerate(unique_titles, 1):
        if store.get(title) is not None:
            cached += 1
            _log_progress("sentences:cache", index, len(unique_titles), started)
            continue
        document = wiki.get(title)
        if document is None:
            failures.append({"title": title, "reason": "missing_wiki_document"})
            continue
        database_title, text = document
        try:
            sentences, offsets = corenlp.split(text)
            store.put(
                requested_title=title,
                database_title=database_title,
                raw_text=text,
                sentences=sentences,
                offsets=offsets,
                updated_utc=utc_now(),
            )
            created += 1
        except Exception as exc:
            failures.append({"title": title, "reason": type(exc).__name__, "detail": str(exc)})
        _log_progress("sentences:cache", index, len(unique_titles), started)
        if mirror_path is not None and index % backup_every == 0:
            store.backup_to(mirror_path)
    if mirror_path is not None:
        store.backup_to(mirror_path)
    print(
        f"[sentences:cache] complete cached={cached:,} created={created:,} "
        f"failures={len(failures):,} elapsed={(time.perf_counter() - started) / 60:.1f}m",
        flush=True,
    )
    return failures


def evidence_stage(
    settings: Settings,
    run_name: str,
    *,
    hops: set[int],
    corenlp_url: str = "http://127.0.0.1:9000",
) -> dict[str, Any]:
    print("CUDA:", require_cuda_runtime(), flush=True)
    paths, train, dev, _, _ = _base_inputs(settings)
    subtrain, validation, _ = _split_rows(settings, train, dev)
    eligible_train = subtrain + validation
    config = settings.raw["evidence"]
    run_dir = settings.output_root / run_name
    stage_dir = settings.stage_dir(run_name, "03_evidence")
    _prepare_model_contract(stage_dir, config, settings.seed, subtrain, validation)
    retrieval_train = _records(run_dir, "02_multihop", "train")
    retrieval_dev = _records(run_dir, "02_multihop", "dev")
    cache_path = settings.data_path("sentence_cache")
    overrides_path = settings.data_path("sentence_overrides")
    mirror_path = run_dir / "shared" / "02_corenlp_sentence_cache_v2.sqlite"
    if not cache_path.is_file() and mirror_path.is_file():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mirror_path, cache_path)

    needed = []
    eligible_training_rows = subtrain + validation
    for row in eligible_training_rows:
        needed.extend(gold_titles(row))
        needed.extend(
            retrieval_train[str(row["uid"])]["ranked_documents"][:int(config["documents"])]
        )
    for row in _filter_hops(dev, hops):
        needed.extend(
            retrieval_dev[str(row["uid"])]["ranked_documents"][:int(config["documents"])]
        )
    print(
        f"[evidence] sentence arrays requested={len(needed):,} "
        f"unique={len(dict.fromkeys(needed)):,}",
        flush=True,
    )
    with WikiStore(paths["wiki_db"]) as wiki:
        writable_store = SentenceStore(
            cache_path, overrides_path=overrides_path, writable=True
        )
        try:
            population_failures = _ensure_sentence_arrays(
                needed,
                store=writable_store,
                wiki=wiki,
                corenlp=CoreNLPHTTP(corenlp_url),
                mirror_path=mirror_path,
            )
        finally:
            writable_store.close()

    print("[evidence] auditing Train Gold sentence alignment", flush=True)
    gold_texts = _gold_text_references(settings)
    train_audit_store = SentenceStore(cache_path, overrides_path=overrides_path)
    try:
        with WikiStore(paths["wiki_db"]) as audit_wiki:
            train_alignment = audit_gold_alignment(
                eligible_train, train_audit_store, wiki=audit_wiki,
                gold_texts=gold_texts,
            )
    finally:
        train_audit_store.close()
    atomic_write_json(stage_dir / "train_alignment_audit.json", train_alignment)
    print(
        f"[evidence] Train alignment audit complete affected="
        f"{train_alignment.get('affected_claims', 0):,}",
        flush=True,
    )

    store = SentenceStore(cache_path, overrides_path=overrides_path)
    try:
        train_failures, validation_failures = [], []
        model_path = stage_dir / "model" / "best_model"
        if not (model_path / "config.json").is_file():
            train_pairs, train_failures = build_sentence_training_rows(
                subtrain,
                retrieval_train,
                store=store,
                document_limit=int(config["documents"]),
                negatives_per_positive=int(config["train_hard_negatives_per_positive"]),
            )
            validation_pairs, validation_failures = build_sentence_training_rows(
                validation,
                retrieval_train,
                store=store,
                document_limit=int(config["documents"]),
                negatives_per_positive=int(config["train_hard_negatives_per_positive"]),
            )
            model_path = _train_or_reuse(
                train_pairs, validation_pairs, stage_dir=stage_dir,
                config=config, seed=settings.seed,
            )
        scorer = HFPairScorer(model_path, max_length=int(config["max_length"]))
        for split, rows, retrieval in (
            ("train", train, retrieval_train),
            ("dev", dev, retrieval_dev),
        ):
            selected = eligible_train if split == "train" else _filter_hops(rows, hops)
            checkpoint_config = {**_stage_config(settings, "evidence"), "split": split}
            with JsonlCheckpoint(stage_dir / split, checkpoint_config) as checkpoint:
                label = f"evidence:{split}"
                pending = _pending_rows(selected, checkpoint, label)
                started = time.perf_counter()
                for index, row in enumerate(pending, 1):
                    uid = str(row["uid"])
                    checkpoint.append(select_evidence(
                        inference_view(row),
                        retrieval[uid]["ranked_documents"],
                        store=store,
                        scorer=scorer,
                        document_limit=int(config["documents"]),
                        sentences_per_document=int(config["sentences_per_document"]),
                        save_top_k=int(config["save_top_k"]),
                        batch_size=int(config["eval_batch_size"]),
                    ))
                    _log_progress(label, index, len(pending), started)
    finally:
        store.close()

    # Dev Gold enters only after production evidence predictions are complete.
    # This block prepares isolated oracle/audit data and cannot alter predictions.
    with WikiStore(paths["wiki_db"]) as wiki:
        oracle_store = SentenceStore(
            cache_path, overrides_path=overrides_path, writable=True
        )
        try:
            oracle_population_failures = _ensure_sentence_arrays(
                [title for row in dev for title in gold_titles(row)],
                store=oracle_store,
                wiki=wiki,
                corenlp=CoreNLPHTTP(corenlp_url),
                mirror_path=mirror_path,
            )
        finally:
            oracle_store.close()
    audit_store = SentenceStore(cache_path, overrides_path=overrides_path)
    try:
        with WikiStore(paths["wiki_db"]) as audit_wiki:
            alignment = audit_gold_alignment(
                dev, audit_store, wiki=audit_wiki, gold_texts=gold_texts,
            )
    finally:
        audit_store.close()
    historical_uids = _historical_alignment_uids(settings)
    alignment["historical_unverified_uids"] = sorted(historical_uids)
    alignment["historical_unverified_claims"] = len(historical_uids)
    alignment["metric_excluded_uids"] = sorted(
        set(alignment["metric_excluded_uids"]) | historical_uids
    )
    atomic_write_json(stage_dir / "alignment_audit.json", alignment)
    write_jsonl_atomic(
        stage_dir / "failures.jsonl",
        population_failures + train_failures + validation_failures + [
            {**failure, "scope": "oracle_evaluation_only"}
            for failure in oracle_population_failures
        ],
    )
    dev_predictions = _records(run_dir, "03_evidence", "dev")
    metrics = evidence_metrics(
        _filter_hops(dev, hops),
        dev_predictions,
        alignment_bad_uids=set(alignment["metric_excluded_uids"]),
    )
    atomic_write_json(stage_dir / "metrics.json", metrics)
    return metrics


def verifier_stage(settings: Settings, run_name: str, *, hops: set[int]) -> dict[str, Any]:
    print("CUDA:", require_cuda_runtime(), flush=True)
    _, train, dev, _, _ = _base_inputs(settings)
    subtrain, validation, _ = _split_rows(settings, train, dev)
    eligible_train = subtrain + validation
    config = settings.raw["verifier"]
    run_dir = settings.output_root / run_name
    stage_dir = settings.stage_dir(run_name, "04_verifier")
    _prepare_model_contract(stage_dir, config, settings.seed, subtrain, validation)
    evidence_train = _records(run_dir, "03_evidence", "train")
    evidence_dev = _records(run_dir, "03_evidence", "dev")
    print(
        f"[verifier] loaded evidence train={len(evidence_train):,} dev={len(evidence_dev):,}",
        flush=True,
    )
    train_failures, validation_failures = [], []
    train_alignment_path = run_dir / "03_evidence" / "train_alignment_audit.json"
    if not train_alignment_path.is_file():
        raise FileNotFoundError(f"Train Gold alignment audit missing: {train_alignment_path}")
    train_alignment = read_json(train_alignment_path)
    verified_gold_uids = set(train_alignment.get("gold_text_verified_uids", []))
    model_path = stage_dir / "model" / "best_model"
    if not (model_path / "config.json").is_file():
        store = SentenceStore(
            settings.data_path("sentence_cache"),
            overrides_path=settings.data_path("sentence_overrides"),
        )
        try:
            train_rows, train_failures = build_verifier_training_rows(
                subtrain,
                evidence_train,
                store=store,
                retrieved_k=int(config["retrieved_sentences"]),
                gold_mix_ratio=float(config["gold_mix_ratio"]),
                seed=settings.seed,
                gold_text_verified_uids=verified_gold_uids,
            )
            validation_rows, validation_failures = build_verifier_training_rows(
                validation,
                evidence_train,
                store=store,
                retrieved_k=int(config["retrieved_sentences"]),
                gold_mix_ratio=0.0,
                seed=settings.seed,
            )
        finally:
            store.close()
        counts = Counter(row["label"] for row in train_rows)
        class_weights = [len(train_rows) / (2 * counts[label]) for label in (0, 1)]
        model_path = _train_or_reuse(
            train_rows,
            validation_rows,
            stage_dir=stage_dir,
            config=config,
            seed=settings.seed,
            class_weights=class_weights,
        )
    scorer = HFPairScorer(model_path, max_length=int(config["max_length"]))
    for split, rows, evidence in (
        ("train", train, evidence_train),
        ("dev", dev, evidence_dev),
    ):
        selected = eligible_train if split == "train" else _filter_hops(rows, hops)
        checkpoint_config = {**_stage_config(settings, "verifier"), "split": split}
        with JsonlCheckpoint(stage_dir / split, checkpoint_config) as checkpoint:
            label = f"verifier:{split}"
            pending = _pending_rows(selected, checkpoint, label)
            started = time.perf_counter()
            for index, row in enumerate(pending, 1):
                uid = str(row["uid"])
                checkpoint.append(verify_claim(
                    inference_view(row),
                    evidence[uid]["predicted_evidence"],
                    scorer=scorer,
                    retrieved_k=int(config["retrieved_sentences"]),
                ))
                _log_progress(label, index, len(pending), started)
    write_jsonl_atomic(stage_dir / "failures.jsonl", train_failures + validation_failures)
    return {
        "train_failures": len(train_failures),
        "validation_failures": len(validation_failures),
    }


def evaluate_stage(settings: Settings, run_name: str, *, hops: set[int]) -> dict[str, Any]:
    print("CUDA:", require_cuda_runtime(), flush=True)
    _, train, dev, _, dev_tfidf = _base_inputs(settings)
    _, validation, _ = _split_rows(settings, train, dev)
    selected = _filter_hops(dev, hops)
    run_dir = settings.output_root / run_name
    stage_dir = settings.stage_dir(run_name, "05_e2e")
    reranker = _records(run_dir, "01_retrieval", "dev")
    retrieval = _records(run_dir, "02_multihop", "dev")
    evidence = _records(run_dir, "03_evidence", "dev")
    verifier = _records(run_dir, "04_verifier", "dev")
    evidence_train = _records(run_dir, "03_evidence", "train")
    verifier_train = _records(run_dir, "04_verifier", "train")
    print(
        f"[evaluate] loaded predictions selected_dev={len(selected):,}; "
        "building retrieved metrics and failure analysis",
        flush=True,
    )
    combined = {}
    failure_map: dict[str, dict[str, Any]] = {}
    for row in selected:
        uid = str(row["uid"])
        combined[uid] = {
            "uid": uid,
            "predicted_label": verifier[uid]["predicted_label"],
            "supported_probability": verifier[uid]["supported_probability"],
            "predicted_evidence": evidence[uid]["predicted_evidence"],
        }
        gold_doc_keys = {title_key(title) for title in gold_titles(row)}
        candidate_keys = {title_key(title) for title in retrieval[uid]["candidate_documents"]}
        ranked5_keys = {title_key(title) for title in retrieval[uid]["ranked_documents"][:5]}
        reasons = []
        if not gold_doc_keys <= candidate_keys:
            reasons.append("gold_document_missing_from_candidates")
        elif not gold_doc_keys <= ranked5_keys:
            reasons.append("gold_document_below_top5")
        if int(combined[uid]["predicted_label"]) != label_id(row["label"]):
            reasons.append("label_incorrect")
        if evidence[uid].get("failures"):
            reasons.append("sentence_array_missing")
        if reasons:
            failure_map[uid] = {
                "uid": uid,
                "num_hops": int(row["num_hops"]),
                "reasons": reasons,
            }

    validation_for_selection = _filter_hops(validation, hops)
    validation_combined = {
        str(row["uid"]): {
            "predicted_label": verifier_train[str(row["uid"])]["predicted_label"],
            "predicted_evidence": evidence_train[str(row["uid"])]["predicted_evidence"],
        }
        for row in validation_for_selection
    }
    submit_k_results = {
        int(k): hover_scores(validation_for_selection, validation_combined, submit_k=int(k))
        for k in settings.raw["evaluation"]["evidence_submit_ks"]
    }
    submit_k = max(
        submit_k_results,
        key=lambda k: (submit_k_results[k]["hover_score"], submit_k_results[k]["evidence_f1"], -k),
    )

    alignment_path = run_dir / "03_evidence" / "alignment_audit.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    bad_uids = set(alignment.get("metric_excluded_uids", alignment["affected_uids"]))
    for row in selected:
        uid = str(row["uid"])
        kept = official_filtered_evidence(
            evidence[uid]["predicted_evidence"], int(row["num_hops"]), submit_k
        )
        reasons = failure_map.setdefault(
            uid, {"uid": uid, "num_hops": int(row["num_hops"]), "reasons": []}
        )["reasons"]
        if not gold_document_fact_coverage(gold_fact_keys(row), kept):
            reasons.append("evidence_document_coverage_failed")
        if uid in bad_uids:
            reasons.append("known_sentence_alignment_issue")

    evidence_model = run_dir / "03_evidence" / "model" / "best_model"
    verifier_model = run_dir / "04_verifier" / "model" / "best_model"
    evidence_scorer = HFPairScorer(
        evidence_model, max_length=int(settings.raw["evidence"]["max_length"])
    )
    verifier_scorer = HFPairScorer(
        verifier_model, max_length=int(settings.raw["verifier"]["max_length"])
    )
    sentence_store = SentenceStore(
        settings.data_path("sentence_cache"),
        overrides_path=settings.data_path("sentence_overrides"),
    )
    try:
        oracle_doc_config = {
            "schema_version": 1, "stage": "oracle_document_evaluation_only",
            "evidence_config": settings.raw["evidence"],
            "verifier_config": settings.raw["verifier"],
        }
        with JsonlCheckpoint(stage_dir / "oracle_document", oracle_doc_config) as checkpoint:
            label = "evaluate:oracle_document"
            pending = _pending_rows(selected, checkpoint, label)
            started = time.perf_counter()
            for index, row in enumerate(pending, 1):
                uid = str(row["uid"])
                selected_evidence = oracle_document_selection(
                    row,
                    store=sentence_store,
                    scorer=evidence_scorer,
                    sentences_per_document=int(
                        settings.raw["evidence"]["sentences_per_document"]
                    ),
                    save_top_k=int(settings.raw["evidence"]["save_top_k"]),
                    batch_size=int(settings.raw["evidence"]["eval_batch_size"]),
                )
                label_prediction = verify_claim(
                    inference_view(row),
                    selected_evidence["predicted_evidence"],
                    scorer=verifier_scorer,
                    retrieved_k=int(settings.raw["verifier"]["retrieved_sentences"]),
                )
                checkpoint.append({
                    "uid": uid,
                    "predicted_label": label_prediction["predicted_label"],
                    "predicted_evidence": selected_evidence["predicted_evidence"],
                    "evaluation_only": True,
                })
                _log_progress(label, index, len(pending), started)
        gold_evidence_config = {
            "schema_version": 1, "stage": "oracle_gold_evidence_evaluation_only",
            "verifier_config": settings.raw["verifier"],
        }
        with JsonlCheckpoint(stage_dir / "oracle_evidence", gold_evidence_config) as checkpoint:
            eligible_oracle = [row for row in selected if str(row["uid"]) not in bad_uids]
            label = "evaluate:oracle_evidence"
            pending = _pending_rows(eligible_oracle, checkpoint, label)
            started = time.perf_counter()
            for index, row in enumerate(pending, 1):
                uid = str(row["uid"])
                facts = []
                for title, sentence_id in row["supporting_facts"]:
                    document = sentence_store.get(title)
                    facts.append({
                        "title": title,
                        "sentence_id": int(sentence_id),
                        "sentence": document["sentences"][int(sentence_id)],
                        "score": 1.0,
                    })
                label_prediction = verify_claim(
                    inference_view(row), facts, scorer=verifier_scorer,
                    retrieved_k=len(facts),
                )
                checkpoint.append({
                    "uid": uid,
                    "predicted_label": label_prediction["predicted_label"],
                    "predicted_evidence": facts,
                    "evaluation_only": True,
                })
                _log_progress(label, index, len(pending), started)
    finally:
        sentence_store.close()

    oracle_doc = index_by_uid(iter_jsonl(stage_dir / "oracle_document" / "predictions.jsonl"))
    oracle_evidence = index_by_uid(iter_jsonl(stage_dir / "oracle_evidence" / "predictions.jsonl"))
    verified_selected = [row for row in selected if str(row["uid"]) not in bad_uids]
    by_scope = {}
    for scope in ("all", "2", "3", "4"):
        rows = selected if scope == "all" else [
            row for row in selected if int(row["num_hops"]) == int(scope)
        ]
        if rows:
            by_scope[scope] = hover_scores(rows, combined, submit_k=submit_k)

    def oracle_scopes(predictions, rows_available, *, evidence_limit=submit_k):
        output = {}
        for scope in ("all", "2", "3", "4"):
            rows = rows_available if scope == "all" else [
                row for row in rows_available if int(row["num_hops"]) == int(scope)
            ]
            if rows:
                output[scope] = hover_scores(rows, predictions, submit_k=evidence_limit)
        return output

    tfidf_predictions = {
        str(row["uid"]): {
            "ranked_documents": dev_tfidf[str(row["uid"])],
            "candidate_documents": dev_tfidf[str(row["uid"])],
        }
        for row in selected
    }
    report = {
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "submit_k": submit_k,
        "submit_k_selection": (
            "max HoVer score on Train internal Validation filtered to requested hops; "
            "Dev not consulted"
        ),
        "submit_k_selection_hops": sorted(hops),
        "validation_submit_k_results": submit_k_results,
        "retrieved": by_scope,
        "document": {
            "tfidf_top100": document_metrics(
                selected, tfidf_predictions, ks=settings.raw["evaluation"]["document_ks"]
            ),
            "roberta_reranker": document_metrics(
                selected, reranker, ks=settings.raw["evaluation"]["document_ks"]
            ),
            "multihop": document_metrics(
                selected, retrieval, ks=settings.raw["evaluation"]["document_ks"]
            ),
        },
        "evidence": evidence_metrics(
            selected, evidence, submit_k=submit_k, alignment_bad_uids=bad_uids
        ),
        "alignment": {
            "historical_known_issue_claims": int(settings.raw["evidence"]["known_alignment_issue_claims"]),
            "historical_audit_uids_loaded": len(alignment.get("historical_unverified_uids", [])),
            "currently_unresolved_claims": alignment["affected_claims"],
            "metric_excluded_claims": len(bad_uids),
            "sentence_ids_modified": False,
            "validation_bypassed": False,
        },
        "other_team_comparison": {
            "eligible_scope": "2-Hop Dev 1,126 only",
            "reference_hover_score": 0.581,
            "current": by_scope.get("2", {}).get("hover_score"),
        },
        "oracle": {
            "document_oracle_by_hop": oracle_scopes(oracle_doc, selected),
            "document_oracle_alignment_verified_by_hop": (
                oracle_scopes(oracle_doc, verified_selected) if verified_selected else None
            ),
            "gold_evidence_oracle_alignment_verified_by_hop": (
                oracle_scopes(oracle_evidence, verified_selected, evidence_limit=None)
                if verified_selected else None
            ),
            "alignment_issue_claims_excluded_only_from_gold_evidence_oracle": len(selected) - len(verified_selected),
            "used_for_candidate_generation": False,
        },
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(stage_dir / "metrics.json", report)
    write_jsonl_atomic(
        stage_dir / "predictions.jsonl",
        (combined[str(row["uid"])] for row in selected),
    )
    write_jsonl_atomic(
        stage_dir / "failures.jsonl",
        (record for record in failure_map.values() if record["reasons"]),
    )
    return report

#!/usr/bin/env python3
"""Bounded real-data CPU smoke benchmark for multi-hop candidate expansion."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from hover_e2e.config import load_settings
from hover_e2e.data import inference_view, load_claims, load_tfidf
from hover_e2e.io_utils import atomic_write_json
from hover_e2e.metrics import document_metrics
from hover_e2e.multihop import expand_and_rerank
from hover_e2e.retrieval import LexicalPairScorer
from hover_e2e.wiki import TitleMatcher, WikiStore, build_title_index


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/base.json"))
    parser.add_argument("--data-root")
    parser.add_argument("--claims", type=int, default=12, choices=range(10, 21))
    parser.add_argument("--hop", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument(
        "--output", default=str(root / "cpu_reports/multihop_smoke_12.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config, data_root=args.data_root)
    dev = load_claims(settings.data_path("dev_claims"), expected_count=4_000)
    rows = [row for row in dev if int(row["num_hops"]) == args.hop][:args.claims]
    tfidf = load_tfidf(settings.data_path("dev_tfidf"), expected_uids=[row["uid"] for row in dev])
    config = settings.raw["multihop"]
    predictions = {}
    started = time.perf_counter()
    with WikiStore(settings.data_path("wiki_db")) as wiki:
        index = build_title_index(wiki, settings.data_path("title_index"))
        matcher = TitleMatcher(
            index,
            max_ngram=int(config["max_title_ngram"]),
            max_titles_per_surface=int(config["max_titles_per_surface"]),
            max_text_chars=int(config["source_max_chars"]),
        )
        try:
            scorer = LexicalPairScorer()
            for row in rows:
                uid = str(row["uid"])
                ranked = tfidf[uid][:100]
                base = {
                    "uid": uid,
                    "ranked_documents": ranked,
                    "candidate_documents": ranked,
                    "scores": [1.0 / (index + 1) for index in range(len(ranked))],
                }
                predictions[uid] = expand_and_rerank(
                    inference_view(row), base, wiki=wiki, matcher=matcher, scorer=scorer,
                    seed_docs=int(config["seed_docs"]),
                    beam_width=int(config["beam_width"]),
                    candidate_limit=int(config["candidate_limit"]),
                    max_scored_candidates=int(config["max_scored_candidates"]),
                    max_scored_per_depth=int(config["max_scored_per_depth"]),
                    score_alpha=float(config["path_score_alpha"]),
                    batch_size=int(config["eval_batch_size"]),
                )
        finally:
            matcher.close()
    elapsed = time.perf_counter() - started
    runtimes = [predictions[str(row["uid"])]["runtime"] for row in rows]
    report = {
        "status": "CPU_SMOKE_ONLY_NOT_ROBERTA_PERFORMANCE",
        "claims": len(rows),
        "hop": args.hop,
        "base_candidates": "TF-IDF Top-100 in original order",
        "scorer": "LexicalPairScorer CPU test double",
        "wall_seconds_including_io": elapsed,
        "claims_per_second": len(rows) / elapsed if elapsed else None,
        "mean_candidate_documents": sum(
            item["unique_candidate_documents"] for item in runtimes
        ) / len(runtimes),
        "total_scored_instances": sum(item["scored_instances"] for item in runtimes),
        "max_process_peak_rss_mib": max(item["process_peak_rss_mib"] for item in runtimes),
        "budget_exhausted_claims": sum(item["budget_exhausted"] for item in runtimes),
        "candidate_metrics": document_metrics(rows, predictions, ks=(3, 5, 100)),
        "per_uid_runtime": [
            {"uid": str(row["uid"]), **predictions[str(row["uid"])]["runtime"]}
            for row in rows
        ],
    }
    atomic_write_json(args.output, report)
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

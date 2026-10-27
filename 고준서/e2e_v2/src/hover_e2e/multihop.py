from __future__ import annotations

from dataclasses import dataclass
import math
import resource
import sys
import time
from typing import Any, Iterable

from .data import InferenceClaim, gold_titles
from .normalization import nfc_title, stable_unique, title_key
from .retrieval import PairScorer, document_passage
from .wiki import TitleMatcher, WikiStore


@dataclass(frozen=True)
class BeamPath:
    documents: tuple[str, ...]
    score: float


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def expansion_candidates(
    claim: InferenceClaim,
    source_titles: Iterable[str],
    *,
    wiki: WikiStore,
    matcher: TitleMatcher,
    existing: Iterable[str] = (),
    limit: int = 500,
) -> list[str]:
    """No Gold or oracle evidence parameter exists by construction."""
    candidates = list(matcher.match(claim.claim))
    for title in source_titles:
        text = wiki.text(title)
        if text:
            candidates.extend(matcher.match(text))
    blocked = {title_key(title) for title in existing}
    output = [title for title in stable_unique(candidates, key=title_key) if title_key(title) not in blocked]
    return output[:limit]


def _context(claim: InferenceClaim, path: tuple[str, ...], wiki: WikiStore) -> str:
    documents = []
    for title in path:
        documents.append(document_passage(title, wiki.text(title), max_chars=5_000))
    return claim.claim + " </s> " + " </s> ".join(documents)


def expand_and_rerank(
    claim: InferenceClaim,
    base_prediction: dict[str, Any],
    *,
    wiki: WikiStore,
    matcher: TitleMatcher,
    scorer: PairScorer,
    seed_docs: int = 3,
    beam_width: int = 5,
    candidate_limit: int = 100,
    max_scored_candidates: int = 600,
    max_scored_per_depth: int = 250,
    score_alpha: float = 0.7,
    batch_size: int = 64,
) -> dict[str, Any]:
    started = time.perf_counter()
    ranked = [nfc_title(title) for title in base_prediction["ranked_documents"]]
    base_scores = [float(value) for value in base_prediction.get("scores", [])]
    if not base_scores:
        base_scores = [1.0 / (index + 1) for index in range(len(ranked))]
    seeds = ranked[:seed_docs]
    beams = [BeamPath((title,), math.log(max(base_scores[index], 1e-8))) for index, title in enumerate(seeds)]
    all_candidates = list(base_prediction.get("candidate_documents", ranked))
    scored_candidates: list[str] = []
    scored_instances = 0
    score_batches = 0
    budget_exhausted = False
    path_trace: list[dict[str, Any]] = []

    while beams and len(beams[0].documents) < claim.num_hops:
        total_remaining = int(max_scored_candidates) - scored_instances
        depth_remaining = min(int(max_scored_per_depth), total_remaining)
        if depth_remaining <= 0:
            budget_exhausted = True
            break
        next_beams: list[BeamPath] = []
        depth_scored = 0
        for beam_index, beam in enumerate(beams):
            remaining_beams = len(beams) - beam_index
            fair_share = max(1, depth_remaining // max(remaining_beams, 1))
            allowance = min(int(candidate_limit), fair_share, total_remaining)
            if allowance <= 0:
                budget_exhausted = True
                break
            candidates = expansion_candidates(
                claim, [beam.documents[-1]], wiki=wiki, matcher=matcher,
                existing=beam.documents, limit=candidate_limit,
            )
            # Preserve original retrieval candidates as a fallback, without Gold insertion.
            candidates = stable_unique(candidates + ranked[:50], key=title_key)
            candidates = [title for title in candidates if title_key(title) not in {title_key(x) for x in beam.documents}]
            if len(candidates) > allowance:
                budget_exhausted = True
            candidates = candidates[:allowance]
            all_candidates.extend(candidates)
            scored_candidates.extend(candidates)
            scored_instances += len(candidates)
            depth_scored += len(candidates)
            depth_remaining -= len(candidates)
            total_remaining -= len(candidates)
            left = [_context(claim, beam.documents, wiki)] * len(candidates)
            right = [document_passage(title, wiki.text(title)) for title in candidates]
            scores = scorer.score(left, right, batch_size=batch_size) if candidates else []
            score_batches += math.ceil(len(candidates) / batch_size) if candidates else 0
            if len(scores) != len(candidates):
                raise RuntimeError(f"Multi-hop scorer length mismatch: {claim.uid}")
            for title, score in zip(candidates, scores):
                probability = min(max(float(score), 1e-8), 1.0)
                next_beams.append(BeamPath(
                    beam.documents + (title,),
                    score_alpha * beam.score + math.log(probability),
                ))
        next_beams.sort(key=lambda item: (-item.score, tuple(title_key(t) for t in item.documents)))
        deduplicated = []
        seen = set()
        for beam in next_beams:
            key = tuple(title_key(title) for title in beam.documents)
            if key not in seen:
                seen.add(key)
                deduplicated.append(beam)
            if len(deduplicated) >= beam_width:
                break
        if not deduplicated:
            break
        beams = deduplicated
        path_trace.append({
            "depth": len(beams[0].documents),
            "beam_count": len(beams),
            "scored_candidates": depth_scored,
        })

    ranked_from_paths = []
    for beam in sorted(beams, key=lambda item: -item.score):
        ranked_from_paths.extend(beam.documents)
    final_ranked = stable_unique(ranked_from_paths + ranked, key=title_key)
    candidate_documents = stable_unique(
        all_candidates + scored_candidates + ranked_from_paths + final_ranked,
        key=title_key,
    )
    candidate_keys = {title_key(title) for title in candidate_documents}
    if not {title_key(title) for title in scored_candidates + ranked_from_paths} <= candidate_keys:
        raise AssertionError(f"Scored/selected document missing from candidates: {claim.uid}")
    elapsed = time.perf_counter() - started
    return {
        "uid": claim.uid,
        "num_hops": claim.num_hops,
        "ranked_documents": final_ranked,
        "candidate_documents": candidate_documents,
        "paths": [{"documents": list(beam.documents), "score": beam.score} for beam in beams],
        "trace": path_trace,
        "runtime": {
            "elapsed_seconds": elapsed,
            "scored_instances": scored_instances,
            "score_batches": score_batches,
            "unique_candidate_documents": len(candidate_documents),
            "scored_instances_per_second": scored_instances / elapsed if elapsed else None,
            "process_peak_rss_mib": _peak_rss_mib(),
            "candidate_limit_per_beam": int(candidate_limit),
            "max_scored_candidates": int(max_scored_candidates),
            "max_scored_per_depth": int(max_scored_per_depth),
            "budget_exhausted": budget_exhausted,
        },
        "uses_gold_for_generation": False,
        "uses_evidence_for_generation": False,
    }


def build_multihop_training_rows(
    claims: Iterable[dict[str, Any]], base_predictions: dict[str, dict[str, Any]], *,
    wiki: WikiStore, matcher: TitleMatcher, hard_negatives: int = 10,
    candidate_limit: int = 100,
) -> list[dict[str, Any]]:
    """Teacher-forced Train-only pairs with generated hard negatives."""
    claims = list(claims)
    rows = []
    print(f"[multihop:pairs] start claims={len(claims):,}", flush=True)
    for claim_index, row in enumerate(claims, 1):
        claim = InferenceClaim(str(row["uid"]), str(row["claim"]), int(row["num_hops"]))
        gold = gold_titles(row)
        if len(gold) != claim.num_hops:
            raise ValueError(f"Gold document/hop mismatch: {claim.uid}")
        orders = [gold, list(reversed(gold))] if len(gold) > 1 else [gold]
        for order_index, order in enumerate(orders):
            for depth in range(1, len(order)):
                prefix = tuple(order[:depth])
                positive = order[depth]
                generated = expansion_candidates(
                    claim, [prefix[-1]], wiki=wiki, matcher=matcher,
                    existing=prefix, limit=candidate_limit,
                )
                gold_keys = {title_key(title) for title in gold}
                negatives = [title for title in generated if title_key(title) not in gold_keys]
                if len(negatives) < hard_negatives:
                    negatives += [
                        title for title in base_predictions[claim.uid]["ranked_documents"]
                        if title_key(title) not in gold_keys and title_key(title) not in {title_key(x) for x in negatives}
                    ]
                selected = [(positive, 1, "train_gold_positive")]
                selected += [(title, 0, "generated_hard_negative") for title in negatives[:hard_negatives]]
                context = _context(claim, prefix, wiki)
                for title, label, source in selected:
                    rows.append({
                        "uid": claim.uid, "order": order_index, "depth": depth,
                        "path": list(prefix), "title": title, "text_a": context,
                        "text_b": document_passage(title, wiki.text(title)),
                        "label": label, "source": source,
                    })
        if claim_index == 1 or claim_index == len(claims) or claim_index % 250 == 0:
            print(
                f"[multihop:pairs] claims={claim_index:,}/{len(claims):,} "
                f"pairs={len(rows):,}",
                flush=True,
            )
    return rows

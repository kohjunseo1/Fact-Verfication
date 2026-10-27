from __future__ import annotations

import math
import re
from typing import Any, Iterable, Protocol

from .data import InferenceClaim, gold_titles
from .normalization import nfc_title, stable_unique, title_key
from .wiki import WikiStore


class PairScorer(Protocol):
    def score(self, left: list[str], right: list[str], *, batch_size: int = 64) -> list[float]: ...


def document_passage(title: str, text: str | None, max_chars: int = 12_000) -> str:
    return f"{nfc_title(title)}. {(text or '')[:max_chars]}"


def build_reranker_training_rows(
    claims: Iterable[dict[str, Any]], tfidf: dict[str, list[str]], *,
    wiki: WikiStore, hard_negatives: int = 12,
) -> list[dict[str, Any]]:
    """Train-only builder. Gold insertion is legal here and explicitly tagged."""
    claims = list(claims)
    rows = []
    print(f"[retrieval:pairs] start claims={len(claims):,}", flush=True)
    for index, claim in enumerate(claims, 1):
        uid = str(claim["uid"])
        gold = gold_titles(claim)
        gold_keys = {title_key(title) for title in gold}
        negatives = [title for title in tfidf[uid] if title_key(title) not in gold_keys][:hard_negatives]
        selected = [(title, 1, "train_gold_positive") for title in gold]
        selected += [(title, 0, "tfidf_hard_negative") for title in negatives]
        for title, label, source in selected:
            rows.append({
                "uid": uid, "claim": str(claim["claim"]), "title": nfc_title(title),
                "text_a": str(claim["claim"]),
                "text_b": document_passage(title, wiki.text(title)),
                "label": label, "source": source,
            })
        if index == 1 or index == len(claims) or index % 500 == 0:
            print(
                f"[retrieval:pairs] claims={index:,}/{len(claims):,} pairs={len(rows):,}",
                flush=True,
            )
    return rows


def rerank_claim(
    claim: InferenceClaim, candidates: Iterable[str], *, wiki: WikiStore,
    scorer: PairScorer, batch_size: int = 64,
) -> dict[str, Any]:
    titles = stable_unique((nfc_title(title) for title in candidates), key=title_key)
    passages, missing = [], []
    for title in titles:
        text = wiki.text(title)
        if text is None:
            missing.append(title)
        passages.append(document_passage(title, text))
    scores = scorer.score([claim.claim] * len(titles), passages, batch_size=batch_size)
    if len(scores) != len(titles) or not all(math.isfinite(float(score)) for score in scores):
        raise RuntimeError(f"Invalid reranker output for {claim.uid}")
    order = sorted(range(len(titles)), key=lambda index: (-float(scores[index]), index))
    return {
        "uid": claim.uid,
        "num_hops": claim.num_hops,
        "tfidf_top100": titles,
        "ranked_documents": [titles[index] for index in order],
        "scores": [float(scores[index]) for index in order],
        "missing_wiki_titles": missing,
        "candidate_documents": titles,
    }


_WORD = re.compile(r"\w+", re.UNICODE)


class LexicalPairScorer:
    """CPU test double only; it is never the production baseline."""
    def score(self, left: list[str], right: list[str], *, batch_size: int = 64) -> list[float]:
        output = []
        for a, b in zip(left, right):
            a_tokens = {token.casefold() for token in _WORD.findall(a)}
            b_tokens = {token.casefold() for token in _WORD.findall(b)}
            output.append(len(a_tokens & b_tokens) / max(len(a_tokens), 1))
        return output

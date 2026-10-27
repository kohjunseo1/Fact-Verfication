"""Evidence sentence selection with a cross-encoder fine-tuned on HoVer.

train : fine-tune a cross-encoder on (claim, sentence) pairs from HoVer train.
        positives = supporting_facts sentences; negatives = other sentences of the gold docs
        plus sentences of retrieved non-gold docs (from the retrieval results file) or, if that
        file is missing, of docs linked from the gold docs.
        --folds K --exclude-fold k trains on the claims outside fold k (cross-fitting).
score : score every sentence of the retrieved docs and keep the top --keep candidates per claim
        (--folds K --only-fold k scores only fold k, with a model that never saw it).
apply : turn candidate scores into evidence with a selection rule:
        fixed  - top --per-doc per doc, then the global top --n
        hybrid - every sentence with p >= --thr (p = sigmoid(score)), at most --per-doc per doc;
                 a doc with no sentence yet keeps its best one if p >= --doc-thr; at least the best
                 sentence overall; at most --n in total, by score
select: score + apply in one go.

Usage:
  python sentence_selector.py train  --retrieval data/results/train_....jsonl [--folds 2 --exclude-fold 0]
  python sentence_selector.py score  --retrieval data/results/dev_....jsonl --output dev.scores.jsonl
  python sentence_selector.py apply  --scores dev.scores.jsonl --rule hybrid --thr 0.5 --doc-thr 0.2
  python sentence_selector.py select --retrieval data/results/dev_....jsonl [--per-doc 2 --n 5]
"""
import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import (MODELS, DocStore, fold_of, load_hover, read_jsonl, sentence_input, set_seed, setup_torch,
                    write_jsonl)

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUT = MODELS / "sentence_selector"


def build_pairs(store, claims, retrieved, neg_gold, neg_other, rng):
    """(claim, sentence text, label) training pairs."""
    pairs = []
    for c in tqdm(claims, desc="pairs"):
        gold = {(t, i) for t, i in c["supporting_facts"]}
        gold_docs = {t for t, _ in gold}
        negs = []
        for t in gold_docs:
            for i, s in enumerate(store.sents(t)):
                if (t, i) in gold:
                    pairs.append((c["claim"], sentence_input(t, s), 1.0))
                else:
                    negs.append((t, s))
        pairs += [(c["claim"], sentence_input(t, s), 0.0) for t, s in rng.sample(negs, min(neg_gold, len(negs)))]
        if retrieved is not None:
            others = [t for t in retrieved.get(c["uid"], []) if t not in gold_docs]
        else:
            others = [t for g in gold_docs for t in store.neighbours(g) if t not in gold_docs]
        cand = [(t, s) for t in others for s in store.sents(t)]
        pairs += [(c["claim"], sentence_input(t, s), 0.0) for t, s in rng.sample(cand, min(neg_other, len(cand)))]
    return pairs


def train(args):
    set_seed(args.seed)
    device = setup_torch()
    store = DocStore()
    claims = load_hover("train")
    if args.exclude_fold is not None:
        claims = [c for c in claims if fold_of(c["uid"], args.folds) != args.exclude_fold]
        print(f"cross-fitting: training on {len(claims)} claims outside fold {args.exclude_fold}/{args.folds}")
    if args.limit:
        claims = claims[:args.limit]
    retrieved = None
    if args.retrieval:
        retrieved = {r["uid"]: r["opened"] for r in read_jsonl(args.retrieval)}
        print(f"hard negatives from {args.retrieval}")
    else:
        print("no retrieval file: hard negatives from docs linked to the gold docs")
    pairs = build_pairs(store, claims, retrieved, args.neg_gold, args.neg_other, random.Random(args.seed))
    print(f"{len(pairs)} pairs, {sum(p[2] for p in pairs):.0f} positive")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(device)

    def collate(batch):
        enc = tok([b[0] for b in batch], [b[1] for b in batch], truncation=True,
                  max_length=args.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b[2] for b in batch])
        return enc

    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    steps = len(loader) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        bar = tqdm(loader, desc=f"epoch {epoch + 1}")
        for step, batch in enumerate(bar):
            labels = batch.pop("labels").to(device)
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out} ({(time.time() - t0) / 60:.0f} min)")


@torch.no_grad()
def score_rows(args):
    """Candidate scores per claim: [title, sent_id, score], best first, at most args.keep."""
    device = setup_torch()
    store = DocStore()
    tok = AutoTokenizer.from_pretrained(args.out)
    model = AutoModelForSequenceClassification.from_pretrained(args.out).to(device).eval()
    rows = read_jsonl(args.retrieval)
    if args.only_fold is not None:
        rows = [r for r in rows if fold_of(r["uid"], args.folds) == args.only_fold]
    if args.limit:
        rows = rows[:args.limit]
    out = []
    for r in tqdm(rows, desc="score"):
        cands = [(t, i, s) for t in r["opened"] for i, s in enumerate(store.sents(t)) if s.strip()]
        scores = []
        for b in range(0, len(cands), args.batch_size):
            chunk = cands[b:b + args.batch_size]
            enc = tok([r["claim"]] * len(chunk), [sentence_input(t, s) for t, _, s in chunk],
                      truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(device)
            scores += model(**enc).logits.squeeze(-1).float().tolist()
        ranked = sorted(((sc, t, i) for (t, i, _), sc in zip(cands, scores)), reverse=True)[:args.keep]
        out.append({"uid": r["uid"], "claim": r["claim"], "label": r.get("label"), "num_hops": r.get("num_hops"),
                    "cands": [[t, i, round(sc, 4)] for sc, t, i in ranked]})
    return out


def choose(cands, rule, per_doc, n, thr=0.5, doc_thr=0.2):
    """Pick evidence from [title, sent_id, score] candidates (any order); returns the same triples."""
    ranked = sorted(cands, key=lambda c: -c[2])
    taken, count = [], {}

    def take(c):
        taken.append(c)
        count[c[0]] = count.get(c[0], 0) + 1

    if rule == "fixed":
        for c in ranked:
            if count.get(c[0], 0) < per_doc:
                take(c)
        return taken[:n]
    prob = lambda c: 1 / (1 + math.exp(-c[2]))      # noqa: E731
    for c in ranked:                                 # confident sentences
        if prob(c) >= thr and count.get(c[0], 0) < per_doc:
            take(c)
    for c in ranked:                                 # rescue the best sentence of an uncovered doc
        if c[0] not in count and prob(c) >= doc_thr:
            take(c)
    if not taken and ranked:
        take(ranked[0])
    return sorted(taken, key=lambda c: -c[2])[:n]


def apply_rule(score_rows_, args):
    store = DocStore()
    out = []
    for r in score_rows_:
        kept = choose(r["cands"], args.rule, args.per_doc, args.n, args.thr, args.doc_thr)
        out.append({"uid": r["uid"], "claim": r["claim"], "label": r.get("label"), "num_hops": r.get("num_hops"),
                    "evidence": [{"title": t, "sent_id": i, "text": store.sents(t)[i], "score": sc}
                                 for t, i, sc in kept]})
    return out


def evidence_tag(args):
    if args.rule == "fixed":
        return f"evidence_k{args.per_doc}_n{args.n}"
    return f"evidence_hyb{args.thr:g}_{args.doc_thr:g}_k{args.per_doc}_n{args.n}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "score", "apply", "select"])
    p.add_argument("--retrieval", help="retrieval jsonl (train: hard negatives; score/select: input)")
    p.add_argument("--scores", help="apply: candidate-score jsonl from `score`")
    p.add_argument("--base", default=BASE)
    p.add_argument("--out", default=str(OUT), help="model directory")
    p.add_argument("--output", help="score/apply/select: output jsonl path")
    p.add_argument("--rule", choices=["fixed", "hybrid"], default="fixed")
    p.add_argument("--thr", type=float, default=0.5, help="hybrid: keep sentences with p >= thr")
    p.add_argument("--doc-thr", type=float, default=0.2, help="hybrid: rescue an uncovered doc's best sentence")
    p.add_argument("--per-doc", type=int, default=2)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--keep", type=int, default=40, help="score: candidates stored per claim")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--exclude-fold", type=int, help="train: leave this fold out")
    p.add_argument("--only-fold", type=int, help="score/select: only claims of this fold")
    p.add_argument("--neg-gold", type=int, default=4, help="negatives per claim from gold docs")
    p.add_argument("--neg-other", type=int, default=4, help="negatives per claim from other docs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "train":
        return train(args)
    if args.mode in ("score", "select") and not args.retrieval:
        p.error(f"{args.mode} needs --retrieval")
    if args.mode == "apply" and not args.scores:
        p.error("apply needs --scores")
    if args.mode == "score":
        dst = Path(args.output or Path(args.retrieval).with_name(Path(args.retrieval).stem + ".scores.jsonl"))
        write_jsonl(dst, score_rows(args))
    else:
        rows = read_jsonl(args.scores) if args.mode == "apply" else score_rows(args)
        src = Path(args.scores or args.retrieval)
        stem = src.name[:-len(".scores.jsonl")] if src.name.endswith(".scores.jsonl") else src.stem
        dst = Path(args.output or src.with_name(f"{stem}.{evidence_tag(args)}.jsonl"))
        write_jsonl(dst, apply_rule(rows, args))
    print(f"saved {dst}")


if __name__ == "__main__":
    main()

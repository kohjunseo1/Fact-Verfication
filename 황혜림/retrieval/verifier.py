"""Claim verification following the official HoVer recipe.

Official setup (hover-nlp/hover train_claim_verification.sh): bert-base-uncased, input
[CLS] claim [SEP] evidence sentences joined by spaces [SEP], max_seq_length 200,
max_query_length 60, lr 3e-5, per-GPU batch 4 x grad-accumulation 2, max_steps 3000,
trained on the evidence the pipeline selects for the train split.

Usage:
  python verifier.py train   --evidence data/results/train_....evidence_k2_n5.jsonl
  python verifier.py predict --evidence data/results/dev_....evidence_k2_n5.jsonl
"""
import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import LABELS, MODELS, read_jsonl, set_seed, setup_torch, write_jsonl

BASE = "bert-base-uncased"
OUT = MODELS / "verifier"


def encode(tok, claims, contexts, max_len, max_query):
    """[CLS] claim [SEP] evidence [SEP]: claim cut to max_query tokens, evidence fills up to max_len."""
    seqs = []
    for claim, ctx in zip(claims, contexts):
        q = tok.encode(claim, add_special_tokens=False)[:max_query]
        c = tok.encode(ctx, add_special_tokens=False)[:max(0, max_len - len(q) - 3)]
        ids = [tok.cls_token_id] + q + [tok.sep_token_id] + c + [tok.sep_token_id]
        types = [0] * (len(q) + 2) + [1] * (len(c) + 1)
        seqs.append((ids, types))
    width = max(len(ids) for ids, _ in seqs)
    pad = lambda xs, v: xs + [v] * (width - len(xs))
    return {
        "input_ids": torch.tensor([pad(ids, tok.pad_token_id) for ids, _ in seqs]),
        "token_type_ids": torch.tensor([pad(t, 0) for _, t in seqs]),
        "attention_mask": torch.tensor([pad([1] * len(ids), 0) for ids, _ in seqs]),
    }


def context_of(row):
    return " ".join(e["text"].strip() for e in row["evidence"])


def train(args):
    set_seed(args.seed)
    device = setup_torch()
    rows = [r for r in read_jsonl(args.evidence) if r.get("label") in LABELS]
    random.Random(args.seed).shuffle(rows)
    print(f"{len(rows)} training claims from {args.evidence}")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2).to(device)

    def collate(batch):
        enc = encode(tok, [r["claim"] for r in batch], [context_of(r) for r in batch], args.max_len, args.max_query)
        enc["labels"] = torch.tensor([LABELS.index(r["label"]) for r in batch])
        return enc

    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    sched = get_linear_schedule_with_warmup(opt, 0, args.max_steps)
    model.train()
    step, micro, t0 = 0, 0, time.time()
    bar = tqdm(total=args.max_steps, desc="verifier")
    while step < args.max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            bar.update(1)
            if step % 50 == 0:
                bar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")
            if step >= args.max_steps:
                break
    bar.close()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out} ({(time.time() - t0) / 60:.0f} min)")


@torch.no_grad()
def predict(args):
    device = setup_torch()
    tok = AutoTokenizer.from_pretrained(args.out)
    model = AutoModelForSequenceClassification.from_pretrained(args.out).to(device).eval()
    rows = read_jsonl(args.evidence)
    preds = []
    for b in tqdm(range(0, len(rows), args.batch_size * 4), desc="predict"):
        chunk = rows[b:b + args.batch_size * 4]
        enc = encode(tok, [r["claim"] for r in chunk], [context_of(r) for r in chunk], args.max_len, args.max_query)
        prob = model(**{k: v.to(device) for k, v in enc.items()}).logits.softmax(-1)[:, 1].tolist()
        for r, p in zip(chunk, prob):
            preds.append({"uid": r["uid"], "claim": r["claim"], "label": r.get("label"),
                          "pred": LABELS[int(p >= 0.5)], "p_supported": round(p, 4)})
    dst = Path(args.output or Path(args.evidence).with_name(Path(args.evidence).stem + ".pred.jsonl"))
    write_jsonl(dst, preds)
    print(f"saved {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "predict"])
    p.add_argument("--evidence", required=True, help="evidence jsonl from sentence_selector.py select")
    p.add_argument("--base", default=BASE)
    p.add_argument("--out", default=str(OUT), help="model directory")
    p.add_argument("--output", help="predict: output jsonl path")
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--max-query", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    (train if args.mode == "train" else predict)(args)


if __name__ == "__main__":
    main()

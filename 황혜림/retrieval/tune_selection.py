"""Choose the hybrid selection thresholds on out-of-fold train scores (never on dev).

Objective: the highest evidence F1 whose "every gold doc covered" rate (what HoVer score depends
on) stays within --max-cov-drop points of the fixed top-N rule.
Usage: python tune_selection.py --scores train_....scores.jsonl [--split train] [--output params.json]
"""
import argparse
import json
from pathlib import Path

from common import load_hover, read_jsonl
from sentence_selector import choose

THRS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DOC_THRS = [1.01, 0.05, 0.1, 0.2, 0.3, 0.4]      # 1.01 = no rescue


def metrics(rows, gold, rule, per_doc, n, thr=0.5, doc_thr=0.2):
    k = p = r = f = cov = 0
    for row in rows:
        g = gold[row["uid"]]
        sel = {(t, i) for t, i, _ in choose(row["cands"], rule, per_doc, n, thr, doc_thr)}
        hit = len(g & sel)
        pr = hit / len(sel) if sel else 0.0
        rc = hit / len(g)
        k += len(sel); p += pr; r += rc
        f += 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        cov += {t for t, _ in g} <= {t for t, _ in sel}
    m = len(rows)
    return {"sents": k / m, "precision": 100 * p / m, "recall": 100 * r / m, "f1": 100 * f / m,
            "covered": 100 * cov / m}


def main(args):
    rows = read_jsonl(args.scores)
    gold = {c["uid"]: {(t, i) for t, i in c["supporting_facts"]} for c in load_hover(args.split)}
    rows = [r for r in rows if r["uid"] in gold]
    base = metrics(rows, gold, "fixed", args.per_doc, args.n)
    print(f"{len(rows)} claims; fixed top-{args.n}: " + ", ".join(f"{k} {v:.2f}" for k, v in base.items()))
    print("thr | doc_thr | sents | precision | recall | F1 | covered")
    best = None
    for thr in THRS:
        for doc_thr in DOC_THRS:
            if doc_thr < 1 and doc_thr >= thr:
                continue
            m = metrics(rows, gold, "hybrid", args.per_doc, args.n, thr, doc_thr)
            ok = m["covered"] >= base["covered"] - args.max_cov_drop
            print(f"{thr} | {'-' if doc_thr > 1 else doc_thr} | {m['sents']:.2f} | {m['precision']:.1f} | "
                  f"{m['recall']:.1f} | {m['f1']:.1f} | {m['covered']:.1f}{'' if ok else '  (coverage drop)'}")
            if ok and (best is None or m["f1"] > best[2]["f1"]):
                best = (thr, doc_thr, m)
    if best is None:
        print("no setting keeps coverage; use the fixed rule")
        return
    thr, doc_thr, m = best
    params = {"thr": thr, "doc_thr": doc_thr if doc_thr <= 1 else 1.01, "per_doc": args.per_doc, "n": args.n,
              "tuned_on": f"{args.split} out-of-fold ({len(rows)} claims)", "fixed": base, "chosen": m}
    print(f"\nchosen: thr {thr}, doc_thr {doc_thr if doc_thr <= 1 else 'none'} -> F1 {m['f1']:.1f} "
          f"(fixed {base['f1']:.1f}), covered {m['covered']:.1f} (fixed {base['covered']:.1f})")
    if args.output:
        Path(args.output).write_text(json.dumps(params, indent=2), encoding="utf-8")
        print("saved", args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--per-doc", type=int, default=2)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max-cov-drop", type=float, default=1.0)
    p.add_argument("--output")
    main(p.parse_args())

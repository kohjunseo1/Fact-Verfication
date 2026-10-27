"""Stage-by-stage evaluation of the pipeline on a labelled split (default dev).

retrieval : doc recall of the start docs and of the first k opened docs, all-docs-found rate
evidence  : sentence precision / recall / F1 against supporting_facts, all-facts-found rate
verify    : accuracy, macro-F1, confusion, and HoVer score (correct label AND at least one
            selected sentence from every gold supporting document)

Usage: python evaluate.py [--split dev] [--tag B20_M5_d3_out_floor0.7] [--evidence-tag evidence_k2_n5]
"""
import argparse

from common import LABELS, RESULTS, load_hover, read_jsonl

HOPS = (2, 3, 4, "all")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def macro_f1(gold, pred):
    f1s = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return mean(f1s)


def table(title, rows, cols):
    print(f"\n## {title}")
    print("| hops | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for hop, n, vals in rows:
        print(f"| {hop} | {n} | " + " | ".join(f"{v * 100:.1f}" for v in vals) + " |")


def main(args):
    claims = {c["uid"]: c for c in load_hover(args.split)}
    retr = {r["uid"]: r for r in read_jsonl(RESULTS / f"{args.split}_{args.tag}.jsonl")}
    evid = {r["uid"]: r for r in read_jsonl(RESULTS / f"{args.split}_{args.tag}.{args.evidence_tag}.jsonl")}
    pred = {r["uid"]: r for r in read_jsonl(RESULTS / f"{args.split}_{args.tag}.{args.evidence_tag}.pred.jsonl")}
    uids = [u for u in claims if u in retr and u in evid and u in pred]
    print(f"{args.split}: {len(uids)} claims evaluated ({len(claims) - len(uids)} missing)")

    def group(hop):
        return [u for u in uids if hop == "all" or claims[u]["num_hops"] == hop]

    # retrieval
    ks = [k for k in (5, 10, 20) if k <= max(len(r["opened"]) for r in retr.values())]
    rows = []
    for hop in HOPS:
        us = group(hop)
        gold = {u: {t for t, _ in claims[u]["supporting_facts"]} for u in us}
        seeds = {u: {s["title"] for s in retr[u]["seeds"]} for u in us}
        vals = [mean(len(gold[u] & seeds[u]) / len(gold[u]) for u in us)]
        vals += [mean(len(gold[u] & set(retr[u]["opened"][:k])) / len(gold[u]) for u in us) for k in ks]
        vals += [mean(gold[u] <= set(retr[u]["opened"]) for u in us)]
        rows.append((hop, len(us), vals))
    table("1. 후보 문서 검색 (doc recall, %)", rows,
          ["start docs"] + [f"R@{k}" for k in ks] + ["all docs found"])

    # evidence
    rows = []
    for hop in HOPS:
        us = group(hop)
        p, r, f, allf, docs = [], [], [], [], []
        for u in us:
            gold = {(t, i) for t, i in claims[u]["supporting_facts"]}
            sel = {(e["title"], e["sent_id"]) for e in evid[u]["evidence"]}
            hit = len(gold & sel)
            prec = hit / len(sel) if sel else 0.0
            rec = hit / len(gold)
            p.append(prec)
            r.append(rec)
            f.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
            allf.append(gold <= sel)
            docs.append({t for t, _ in gold} <= {t for t, _ in sel})
        rows.append((hop, len(us), [mean(p), mean(r), mean(f), mean(allf), mean(docs)]))
    table("2. 근거 문장 선택 (%)", rows,
          ["precision", "recall", "F1", "all gold sents", "every gold doc covered"])

    # verification
    rows, conf = [], {}
    for hop in HOPS:
        us = group(hop)
        g = [claims[u]["label"] for u in us]
        pr = [pred[u]["pred"] for u in us]
        covered = [{t for t, _ in claims[u]["supporting_facts"]} <= {e["title"] for e in evid[u]["evidence"]}
                   for u in us]
        acc = mean(a == b for a, b in zip(g, pr))
        hover = mean(a == b and c for a, b, c in zip(g, pr, covered))
        rows.append((hop, len(us), [acc, macro_f1(g, pr), hover]))
        if hop == "all":
            for a, b in zip(g, pr):
                conf[(a, b)] = conf.get((a, b), 0) + 1
    table("3. 검증 (%)", rows, ["accuracy", "macro-F1", "HoVer score"])
    print("\nconfusion (gold -> pred):")
    for a in LABELS:
        print(f"  {a:>13}: " + ", ".join(f"{b} {conf.get((a, b), 0)}" for b in LABELS))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--tag", default="B20_M5_d3_out_floor0.7")
    p.add_argument("--evidence-tag", default="evidence_k2_n5")
    main(p.parse_args())

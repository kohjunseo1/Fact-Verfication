"""Sample claims into an annotation sheet in the team format (hover_annotation.csv) plus this
pipeline's outputs, for manual error analysis.

Sample: --per-hop claims per hop count (2/3/4), labels balanced within each hop, fixed seed.
Columns: the team sheet's (uid, hpqa_id, num_hops, label, claim, evidence_str, reasoning,
fail_point, memo) plus pred, p_supported, correct, start_docs, pred_evidence_str,
gold_docs_in_candidates and auto_fail_point (first stage that lost the claim, wrong claims only).

Usage: python make_annotation_sheet.py --tag B20_M5_d3_both_in3d0_floor0.7 \\
           --evidence-tag evidence_hyb0.9_0.4_k2_n5_oof2 --output ../outputs/hover_annotation.csv
"""
import argparse
import csv
import random
from pathlib import Path

from common import RESULTS, load_hover, read_jsonl

COLUMNS = ["uid", "hpqa_id", "num_hops", "label", "pred", "p_supported", "correct", "claim",
           "evidence_str", "pred_evidence_str", "start_docs", "gold_docs_in_candidates",
           "auto_fail_point", "reasoning", "fail_point", "memo"]


def facts_str(facts):
    return " | ".join(f"{t}:{i}" for t, i in facts)


def auto_fail_point(gold_docs, opened, evidence_docs, correct):
    if correct:
        return ""
    if not gold_docs <= set(opened):
        return "문서 검색 (정답 문서가 후보 20개에 없음)"
    if not gold_docs <= evidence_docs:
        return "근거 문장 선택 (정답 문서의 근거 문장을 못 고름)"
    return "판정 (근거는 다 있는데 틀림)"


def main(args):
    claims = {c["uid"]: c for c in load_hover(args.split)}
    retr = {r["uid"]: r for r in read_jsonl(RESULTS / f"{args.split}_{args.tag}.jsonl")}
    evid = {r["uid"]: r for r in read_jsonl(RESULTS / f"{args.split}_{args.tag}.{args.evidence_tag}.jsonl")}
    pred = {r["uid"]: r for r in read_jsonl(RESULTS / f"{args.split}_{args.tag}.{args.evidence_tag}.pred.jsonl")}

    rng = random.Random(args.seed)
    picked = []
    for hop in (2, 3, 4):
        for label in ("SUPPORTED", "NOT_SUPPORTED"):
            pool = sorted(u for u, c in claims.items() if c["num_hops"] == hop and c["label"] == label
                          and u in retr and u in evid and u in pred)
            picked += rng.sample(pool, args.per_hop // 2)
    picked.sort(key=lambda u: (claims[u]["num_hops"], claims[u]["label"], u))

    rows = []
    for u in picked:
        c, r, e, p = claims[u], retr[u], evid[u], pred[u]
        gold_docs = {t for t, _ in c["supporting_facts"]}
        ev_docs = {x["title"] for x in e["evidence"]}
        correct = p["pred"] == c["label"]
        rows.append({
            "uid": u, "hpqa_id": c.get("hpqa_id", ""), "num_hops": c["num_hops"], "label": c["label"],
            "pred": p["pred"], "p_supported": f"{p['p_supported']:.3f}", "correct": "O" if correct else "X",
            "claim": c["claim"],
            "evidence_str": facts_str(c["supporting_facts"]),
            "pred_evidence_str": facts_str((x["title"], x["sent_id"]) for x in e["evidence"]),
            "start_docs": " | ".join(s["title"] for s in r["seeds"]),
            "gold_docs_in_candidates": f"{len(gold_docs & set(r['opened']))}/{len(gold_docs)}",
            "auto_fail_point": auto_fail_point(gold_docs, r["opened"], ev_docs, correct),
            "reasoning": "", "fail_point": "", "memo": "",
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:     # BOM so Excel opens Korean text correctly
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    wrong = [r for r in rows if r["correct"] == "X"]
    print(f"saved {out}: {len(rows)} claims, {len(wrong)} wrong")
    for stage in sorted({r["auto_fail_point"] for r in wrong}):
        print(f"  {stage}: {sum(1 for r in wrong if r['auto_fail_point'] == stage)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--tag", default="B20_M5_d3_both_in3d0_floor0.7")
    p.add_argument("--evidence-tag", default="evidence_hyb0.9_0.4_k2_n5_oof2")
    p.add_argument("--per-hop", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    main(p.parse_args())

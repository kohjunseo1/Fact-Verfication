"""Export per-claim predictions in the team submission format (one JSON object per line):

  uid              claim uid
  predicted_label  "SUPPORTED" / "NOT_SUPPORTED"
  evidence         ranked evidence sentences [{"title", "sentence_id"}, ...] (highest score first)
  docs             ranked document titles (graph-search opening order, best-first)

Usage: python export_predictions.py --tag B20_M5_d3_both_in3d0_floor0.7 \\
           --evidence-tag evidence_hyb0.9_0.4_k2_n5_oof2 --output ../outputs/dev_predictions.jsonl
"""
import argparse

from common import RESULTS, read_jsonl, write_jsonl


def main(args):
    retr = {r["uid"]: r for r in read_jsonl(RESULTS / f"{args.split}_{args.tag}.jsonl")}
    evid = {r["uid"]: r for r in read_jsonl(RESULTS / f"{args.split}_{args.tag}.{args.evidence_tag}.jsonl")}
    pred = read_jsonl(RESULTS / f"{args.split}_{args.tag}.{args.evidence_tag}.pred.jsonl")
    rows = []
    for p in pred:
        u = p["uid"]
        ev = sorted(evid[u]["evidence"], key=lambda e: -e["score"])
        rows.append({
            "uid": u,
            "predicted_label": p["pred"],
            "evidence": [{"title": e["title"], "sentence_id": e["sent_id"]} for e in ev],
            "docs": retr[u]["opened"],
        })
    write_jsonl(args.output, rows)
    short = sum(1 for r in rows if len(r["docs"]) < 10)
    print(f"saved {args.output}: {len(rows)} claims, docs per claim {sum(len(r['docs']) for r in rows) / len(rows):.1f}"
          f" ({short} claims with fewer than 10 docs)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev")
    p.add_argument("--tag", default="B20_M5_d3_both_in3d0_floor0.7")
    p.add_argument("--evidence-tag", default="evidence_hyb0.9_0.4_k2_n5_oof2")
    p.add_argument("--output", required=True)
    main(p.parse_args())

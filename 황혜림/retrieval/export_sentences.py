"""Export every intro paragraph of the doc store as gzip JSON lines: [node_id, title, sentences].

Lets the Colab notebooks run without the Stanford-hosted HotpotQA dump: the retrieval notebook
joins the sentences into intro texts, and build_store.py --sents rebuilds wiki.db from it.
Usage: python export_sentences.py --out ../data/colab_bundle/sents.jsonl.gz
"""
import argparse
import gzip
import json
import sqlite3
import sys
from pathlib import Path

from tqdm import tqdm

from common import DB_PATH


def main(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    total = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    n = 0
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6, newline="\n") as f:
        for nid, title, sents in tqdm(con.execute("SELECT id, title, sents FROM docs ORDER BY id"), total=total):
            f.write(json.dumps([nid, title, json.loads(sents)], ensure_ascii=False) + "\n")
            n += 1
    print(f"exported {n} docs -> {out} ({out.stat().st_size / 2**20:.0f} MB)", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    main(p.parse_args())

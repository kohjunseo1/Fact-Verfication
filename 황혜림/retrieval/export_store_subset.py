"""Export the part of the doc store the evidence/verification stages read, so Colab does not
need to rebuild the full store from the HotpotQA dump.

Keeps every doc that appears in a retrieval file ('opened' or 'gold') or in HoVer
supporting_facts. Same schema as wiki.db (links table left empty: the stages only use links
when no retrieval file is given).
Usage: python export_store_subset.py --out ../data/colab_bundle/store_subset.db
"""
import argparse
import sqlite3
import sys
from pathlib import Path

from common import DB_PATH, RESULTS, load_hover, read_jsonl


def main(args):
    titles = set()
    for path in sorted(RESULTS.glob("*_B*.jsonl")):
        if ".evidence" in path.name or ".pred" in path.name:
            continue
        for r in read_jsonl(path):
            titles.update(r["opened"])
            titles.update(r.get("gold", []))
        print(f"{path.name}: {len(titles)} titles so far", file=sys.stderr)
    for split in ("train", "dev"):
        titles.update(t for c in load_hover(split) for t, _ in c["supporting_facts"])
    print(f"{len(titles)} titles to export", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    dst = sqlite3.connect(out)
    dst.executescript("""
        CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT NOT NULL, sents TEXT NOT NULL);
        CREATE TABLE links (src INTEGER NOT NULL, tgt INTEGER NOT NULL, PRIMARY KEY (src, tgt)) WITHOUT ROWID;
    """)
    batch, missing = [], 0
    for t in titles:
        row = src.execute("SELECT id, title, sents FROM docs WHERE title = ?", (t,)).fetchone()
        if row is None:
            missing += 1
            continue
        batch.append(row)
    dst.executemany("INSERT INTO docs VALUES (?,?,?)", batch)
    dst.executescript("CREATE INDEX links_tgt ON links (tgt, src); CREATE INDEX docs_title ON docs (title);")
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    print(f"exported {len(batch)} docs ({missing} titles not in store) -> {out} "
          f"({out.stat().st_size / 2**20:.0f} MB)", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    main(p.parse_args())

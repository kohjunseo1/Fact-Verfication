"""Build the doc store (default data/store/wiki.db) from the HotpotQA abstracts dump.

docs(id, title, sents): id = node id of the intro-link graph, sents = JSON list of intro
sentences (same segmentation as HoVer supporting_facts sentence ids).
links(src, tgt): deduplicated intro-paragraph links between doc ids.

Inputs default to the local graph build; on Colab pass the bundle files instead:
  python build_store.py --nodes hover_bundle/nodes.tsv.gz --links hover_bundle/links.i32 --sents hover_bundle/sents.jsonl.gz
(--sents replaces --abstracts with the export of export_sentences.py).
"""
import argparse
import bz2
import gzip
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import DATA, DB_PATH

# same sanitising as the graph build (tab, newline, CR, U+2028, U+2029, NEL, VT, FF)
BREAKS = re.compile("[\t\n\r" + "".join(map(chr, (0x2028, 0x2029, 0x85, 0x0B, 0x0C))) + "]")
GRAPH = DATA / "graph" / "abstract"


def read_nodes(path: Path):
    """title -> node id from nodes.tsv(.gz); node id and title are the first two columns."""
    opener = gzip.open if path.suffix == ".gz" else open
    id_of = {}
    with opener(path, "rt", encoding="utf-8", newline="\n") as f:
        for line in f:
            nid, title = line.rstrip("\n").split("\t")[:2]
            id_of[title] = int(nid)
    return id_of


def iter_links(path: Path, id_of):
    """(src, tgt) id pairs from edges.tsv (titles) or links.i32 (int32 pairs)."""
    if path.suffix == ".i32":
        pairs = np.fromfile(path, dtype="<i4").reshape(-1, 2)
        for s in range(0, len(pairs), 500_000):
            yield pairs[s:s + 500_000].tolist()
        return
    batch = []
    with open(path, encoding="utf-8", newline="\n") as f:
        for line in f:
            src, _sent, tgt, _anchor = line.rstrip("\n").split("\t", 3)
            batch.append((id_of[src], id_of[tgt]))
            if len(batch) >= 500_000:
                yield batch
                batch = []
    yield batch


def main(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript("""
        PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;
        CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT NOT NULL, sents TEXT NOT NULL);
        CREATE TABLE links (src INTEGER NOT NULL, tgt INTEGER NOT NULL, PRIMARY KEY (src, tgt)) WITHOUT ROWID;
    """)

    # node ids from the graph build (titles there have line separators replaced by spaces)
    id_of = read_nodes(Path(args.nodes))
    print(f"nodes: {len(id_of)}", file=sys.stderr)

    rows, missing = [], 0
    if args.sents:   # [node_id, title, sentences] lines from export_sentences.py
        with gzip.open(args.sents, "rt", encoding="utf-8", newline="\n") as f:
            for line in tqdm(f, desc="docs"):
                nid, title, sents = json.loads(line)
                rows.append((nid, title, json.dumps(sents, ensure_ascii=False)))
                if len(rows) > 200_000:
                    con.executemany("INSERT INTO docs VALUES (?,?,?)", rows)
                    rows.clear()
    else:
        shards = sorted(Path(args.abstracts).rglob("wiki_*.bz2"))[:args.max_shards]
        for shard in tqdm(shards, desc="docs"):
            with bz2.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    nid = id_of.get(o["title"], id_of.get(BREAKS.sub(" ", o["title"])))
                    if nid is None:
                        missing += 1
                        continue
                    rows.append((nid, o["title"], json.dumps(o["text"], ensure_ascii=False)))
            if len(rows) > 200_000:
                con.executemany("INSERT INTO docs VALUES (?,?,?)", rows)
                rows.clear()
    con.executemany("INSERT INTO docs VALUES (?,?,?)", rows)
    print(f"docs missing node id: {missing}", file=sys.stderr)

    links = Path(args.links)
    if links.exists():
        for batch in tqdm(iter_links(links, id_of), desc="links"):
            con.executemany("INSERT OR IGNORE INTO links VALUES (?,?)", batch)
    else:
        print(f"no link file at {links}: links table left empty", file=sys.stderr)
    print("indexing", file=sys.stderr)
    con.executescript("""
        CREATE INDEX links_tgt ON links (tgt, src);
        CREATE INDEX docs_title ON docs (title);
    """)
    con.commit()
    stats = con.execute("SELECT (SELECT COUNT(*) FROM docs), (SELECT COUNT(*) FROM links)").fetchone()
    print(f"docs {stats[0]}, unique links {stats[1]}", file=sys.stderr)
    con.close()
    tmp.replace(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", default=str(GRAPH / "nodes.tsv"), help="nodes.tsv or nodes.tsv.gz")
    p.add_argument("--links", default=str(GRAPH / "edges.tsv"), help="edges.tsv or links.i32 (optional)")
    p.add_argument("--abstracts", default=str(DATA / "raw" / "abstracts"), help="extracted abstracts dump")
    p.add_argument("--sents", help="sents.jsonl.gz from export_sentences.py (instead of --abstracts)")
    p.add_argument("--out", default=str(DB_PATH))
    p.add_argument("--max-shards", type=int, help="only the first N shards (testing)")
    main(p.parse_args())

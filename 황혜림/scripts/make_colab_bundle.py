"""Pack what Colab cannot download itself into data/colab_bundle/:
  nodes.tsv.gz  node_id \\t title (HotpotQA/HoVer form)
  links.i32     little-endian int32 (src, tgt) pairs, unique, sorted
  hover_{train,dev}_release_v1.1.json
"""
import gzip
import os
import shutil
import sys

import numpy as np

from title_utils import read_lines

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SHIFT = 23   # node ids < 5.3M < 2^23


def main(out=os.path.join(ROOT, "colab_bundle")):
    os.makedirs(out, exist_ok=True)
    id_of = {}
    with gzip.open(os.path.join(out, "nodes.tsv.gz"), "wt", encoding="utf-8", errors="js_replace", newline="\n") as gz:
        for l in read_lines(os.path.join(ROOT, "graph", "abstract", "nodes.tsv")):
            if not l:
                continue
            nid, title = l.split("\t")[:2]
            id_of[title] = int(nid)
            gz.write(f"{nid}\t{title}\n")
    print(f"nodes {len(id_of)}", file=sys.stderr)

    keys, n = [], 0
    for l in read_lines(os.path.join(ROOT, "graph", "abstract", "edges.tsv")):
        if not l:
            continue
        src, _sent, tgt = l.split("\t")[:3]
        keys.append((id_of[src] << SHIFT) | id_of[tgt])
        n += 1
    uniq = np.unique(np.array(keys, dtype=np.int64))
    pairs = np.column_stack([uniq >> SHIFT, uniq & ((1 << SHIFT) - 1)]).astype("<i4")
    pairs.tofile(os.path.join(out, "links.i32"))
    print(f"link occurrences {n}, unique links {uniq.size}", file=sys.stderr)

    for s in ("train", "dev"):
        name = f"hover_{s}_release_v1.1.json"
        shutil.copyfile(os.path.join(ROOT, name), os.path.join(out, name))


if __name__ == "__main__":
    main(*sys.argv[1:2])

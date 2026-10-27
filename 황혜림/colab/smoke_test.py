"""Run the Colab notebook source end-to-end on CPU over a small subgraph.

The subgraph = gold docs of the first LIMIT dev claims, their 2-hop neighbourhood
(capped) and random distractors, re-indexed into a mini bundle.
Usage (repo root): uv run python colab/smoke_test.py
"""
import gzip
import json
import multiprocessing.pool
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "colab" / "hover_retrieval.py"
FULL = ROOT / "data" / "colab_bundle"
SMALL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "smoke_bundle"
LIMIT, DISTRACTORS, NB_CAP = 30, 2000, 60


def make_small_bundle():
    if (SMALL / "nodes.tsv.gz").exists():
        return
    SMALL.mkdir(parents=True, exist_ok=True)
    with gzip.open(FULL / "nodes.tsv.gz", "rt", encoding="utf-8", newline="\n") as f:
        titles = [l.rstrip("\n").split("\t", 1)[1] for l in f]
    id_of = {t: i for i, t in enumerate(titles)}
    links = np.fromfile(FULL / "links.i32", dtype="<i4").reshape(-1, 2)
    claims = json.load(open(FULL / "hover_dev_release_v1.1.json", encoding="utf-8"))[:LIMIT]
    keep = {id_of[t] for c in claims for t, _ in c["supporting_facts"]}
    frontier = set(keep)
    rng = random.Random(0)
    for _ in range(2):
        mask = np.isin(links[:, 0], list(frontier)) | np.isin(links[:, 1], list(frontier))
        nbs = set(links[mask].ravel().tolist()) - keep
        nbs = set(rng.sample(sorted(nbs), min(len(nbs), NB_CAP * len(frontier) // 10 + NB_CAP)))
        keep |= nbs
        frontier = nbs
    keep |= set(rng.sample(range(len(titles)), DISTRACTORS))
    old = np.array(sorted(keep))
    remap = -np.ones(len(titles), np.int64)
    remap[old] = np.arange(len(old))
    with gzip.open(SMALL / "nodes.tsv.gz", "wt", encoding="utf-8", newline="\n") as f:
        for new, o in enumerate(old):
            f.write(f"{new}\t{titles[o]}\n")
    m = (remap[links[:, 0]] >= 0) & (remap[links[:, 1]] >= 0)
    remap[links[m]].astype("<i4").tofile(SMALL / "links.i32")
    json.dump(claims, open(SMALL / "hover_dev_release_v1.1.json", "w", encoding="utf-8"))
    sents_src = FULL / "sents.jsonl.gz"     # exercise the sents.jsonl.gz path when it exists
    if sents_src.exists():
        with gzip.open(sents_src, "rt", encoding="utf-8", newline="\n") as src, \
                gzip.open(SMALL / "sents.jsonl.gz", "wt", encoding="utf-8", newline="\n") as dst:
            for line in src:
                i, title, sents = json.loads(line)
                if remap[i] >= 0:
                    dst.write(json.dumps([int(remap[i]), title, sents], ensure_ascii=False) + "\n")
    print(f"small bundle: {len(old)} docs, {int(m.sum())} links", file=sys.stderr)


def code_cells():
    cells, buf, kind = [], [], None
    for line in SRC.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %%"):
            if kind == "code":
                cells.append("\n".join(buf))
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    if kind == "code":
        cells.append("\n".join(buf))
    return cells


def main():
    make_small_bundle()
    ns = {"__name__": "__notebook__", "display": print}
    for i, cell in enumerate(code_cells()):
        cell = "\n".join(l for l in cell.split("\n")
                         if not l.startswith("!") and "google.colab" not in l and "drive.mount" not in l)
        cell = cell.replace('"/content/work"', repr(str(ROOT / "data" / "raw")))
        cell = cell.replace("from multiprocessing import Pool", "from multiprocessing.pool import ThreadPool as Pool")
        print(f"--- cell {i}", file=sys.stderr)
        exec(compile(cell, f"cell{i}", "exec"), ns)
        if i == 0:   # config overrides
            ns.update(BUNDLE=str(SMALL), CACHE=None, LIMIT=LIMIT, B=10)


if __name__ == "__main__":
    main()

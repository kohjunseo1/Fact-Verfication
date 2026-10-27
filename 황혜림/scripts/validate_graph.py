"""Check how well the 2017-10-01 link graphs cover HoVer supporting documents.

argv: data dir (hover_*.json), abstract graph dir, full graph dir (optional)
"""
import json
import os
import sys

import pandas as pd

from jscompat import js_fixed
from title_utils import html_unescape, read_lines


def pct(a, b):
    return js_fixed(100 * a / b, 1) + "%" if b else "-"


def load_claims(data_dir):
    claims = []
    for s in ("train", "dev"):
        with open(os.path.join(data_dir, f"hover_{s}_release_v1.1.json"), encoding="utf-8") as f:
            for e in json.load(f):
                claims.append({**e, "split": s, "docs": list(dict.fromkeys(t for t, _ in e["supporting_facts"]))})
    return claims


def load_abstract(abs_dir, hover_titles):
    """Edges among HoVer docs: (a, b) -> set of sentence idx the link leaves from."""
    nodes = set()
    for l in read_lines(os.path.join(abs_dir, "nodes.tsv")):
        if l:
            t = l.split("\t")[1]
            if t in hover_titles:
                nodes.add(t)
    edges = {}
    for l in read_lines(os.path.join(abs_dir, "edges.tsv")):
        if not l:
            continue
        s, i, t = l.split("\t")[:3]
        if s in hover_titles and t in hover_titles:
            edges.setdefault((s, t), set()).add(int(i))
    return nodes, edges


def load_full(full_dir, hover_titles):
    # full-graph titles are unescaped Wikipedia titles; HoVer keeps HTML escapes
    by_canon = {html_unescape(t): t for t in hover_titles}
    title_of = {}
    for l in read_lines(os.path.join(full_dir, "nodes.tsv")):
        if l:
            pid, wt = l.split("\t")[:2]
            t = by_canon.get(wt)
            if t is not None:
                title_of[int(pid)] = t
    ids = pd.Index(list(title_of))
    edges = {}
    for chunk in pd.read_csv(os.path.join(full_dir, "edges.tsv.gz"), sep="\t", header=None, names=["s", "t"],
                             dtype="int64", chunksize=20_000_000):
        hit = chunk[chunk.s.isin(ids) & chunk.t.isin(ids)]
        for s, t in zip(hit.s.tolist(), hit.t.tolist()):
            edges[(title_of[s], title_of[t])] = True
    return set(title_of.values()), edges


def report(name, nodes, edges, claims, hover_titles, with_sent):
    print(f"\n=== {name} graph ===")
    print(f"HoVer titles present: {len(nodes)}/{len(hover_titles)} ({pct(len(nodes), len(hover_titles))})")

    def linked(a, b):
        return (a, b) in edges or (b, a) in edges

    for split in ("train", "dev"):
        for h in (2, 3, 4):
            cs = [c for c in claims if c["split"] == split and c["num_hops"] == h]
            all_in = connected = pair_linked = sent_match = 0
            for c in cs:
                docs = c["docs"]
                all_in += all(d in nodes for d in docs)
                # connectivity of the doc set (undirected)
                seen, stack = {docs[0]}, [docs[0]]
                while stack:
                    a = stack.pop()
                    for b in docs:
                        if b not in seen and linked(a, b):
                            seen.add(b)
                            stack.append(b)
                connected += len(seen) == len(docs)
                if h == 2 and linked(docs[0], docs[1]):
                    pair_linked += 1
                    if with_sent:
                        # does a link leave from one of the supporting sentences?
                        sf = {(t, i) for t, i in c["supporting_facts"]}
                        if any((a, i) in sf for a, b in ((docs[0], docs[1]), (docs[1], docs[0]))
                               for i in edges.get((a, b), ())):
                            sent_match += 1
            line = (f"{split} {h}-hop n={len(cs)}: all docs in graph {pct(all_in, len(cs))}, "
                    f"doc set connected {pct(connected, len(cs))}")
            if h == 2:
                line += f", pair directly linked {pct(pair_linked, len(cs))}"
                if with_sent:
                    line += f", link from a supporting sentence {pct(sent_match, len(cs))}"
            print(line)


def main(data_dir, abs_dir, full_dir=None):
    claims = load_claims(data_dir)
    hover_titles = {t for c in claims for t in c["docs"]}
    report("abstract (intro paragraphs)", *load_abstract(abs_dir, hover_titles), claims, hover_titles, True)
    if full_dir:
        report("full-article (pagelinks)", *load_full(full_dir, hover_titles), claims, hover_titles, False)


if __name__ == "__main__":
    main(*sys.argv[1:4])

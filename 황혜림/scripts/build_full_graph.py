"""Build the full-article link graph of English Wikipedia as of 2017-10-01 from the
pagelinks SQL dump, resolving links through 2017-10-01 redirects.

argv  : pages_ns0_20171001.tsv, enwiki-20171001-pagelinks.sql.gz, output directory
Output: nodes.tsv     (page_id, title) -- non-redirect namespace-0 articles
        edges.tsv.gz  (src_page_id, tgt_page_id) sorted, deduplicated
        unresolved_targets.tsv (title, count) top 100k, stats.json
"""
import gzip
import json
import os
import re
import sys
from collections import Counter

import numpy as np

from title_utils import load_redirects, norm_title, open_out, read_lines

NPART = 16
SHIFT = 26                   # page ids < 2^26
FLUSH = 4_000_000
# one `(pl_from, pl_namespace, 'pl_title', pl_from_namespace)` tuple
TUPLE = re.compile(r"\((-?\d+),(-?\d+),'((?:[^'\\]|\\.)*)',(-?\d+)\)", re.S)
ESC = re.compile(r"\\(.)", re.S)
ESC_MAP = {"n": "\n", "t": "\t", "0": "\0"}


def sql_unescape(s: str) -> str:
    return ESC.sub(lambda m: ESC_MAP.get(m.group(1), m.group(1)), s) if "\\" in s else s


def main(pages_tsv, sql_gz, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Page table: articles get ids; redirects map title -> target title.
    id_of = {}
    with open_out(os.path.join(out_dir, "nodes.tsv")) as nodes:
        def on_page(pid, title, target):
            if not target:
                id_of[title] = int(pid)
                nodes.write(f"{pid}\t{title}\n")
        redirects = load_redirects(pages_tsv, on_page)
    print(f"articles {len(id_of)}, redirects {len(redirects)}", file=sys.stderr)

    def resolve_id(raw):
        t = norm_title(raw)
        if not t:
            return -1
        for _ in range(6):
            i = id_of.get(t)
            if i is not None:
                return i
            r = redirects.get(t)
            if r is None:
                return -1
            t = norm_title(r)
        return -1

    article_ids = set(id_of.values())   # sources that are redirects / non-articles are skipped

    # Pass 1: parse SQL, append int32 (src, tgt) pairs to NPART partition files by src.
    parts = [open(os.path.join(out_dir, f"part{k}.bin"), "wb") for k in range(NPART)]
    srcs, tgts = [], []

    def flush():
        if not srcs:
            return
        pairs = np.column_stack([np.array(srcs, np.int32), np.array(tgts, np.int32)])
        part = pairs[:, 0] % NPART
        for k in range(NPART):
            pairs[part == k].astype("<i4").tofile(parts[k])
        srcs.clear()
        tgts.clear()

    st = dict(rows=0, ns0_to_ns0=0, src_not_article=0, kept=0, unresolved=0, self=0)
    unresolved = Counter()
    cache_title, cache_id = None, -1    # rows are grouped by target title
    next_report = 50_000_000
    for line in read_lines(sql_gz):
        if not line.startswith("INSERT INTO"):
            continue
        for m in TUPLE.finditer(line):
            st["rows"] += 1
            if m.group(2) != "0" or m.group(4) != "0":
                continue
            st["ns0_to_ns0"] += 1
            src = int(m.group(1))
            if src not in article_ids:
                st["src_not_article"] += 1
                continue
            title = sql_unescape(m.group(3))
            if title != cache_title:
                cache_title, cache_id = title, resolve_id(title)
            if cache_id < 0:
                st["unresolved"] += 1
                if len(unresolved) < 2_000_000 or title in unresolved:
                    unresolved[title] += 1
                continue
            if cache_id == src:
                st["self"] += 1
                continue
            if src >= 1 << SHIFT or cache_id >= 1 << SHIFT:
                raise ValueError(f"page id overflow: {src} -> {cache_id}")
            srcs.append(src)
            tgts.append(cache_id)
            st["kept"] += 1
            if len(srcs) >= FLUSH:
                flush()
        if st["rows"] >= next_report:
            print(f"rows {st['rows']}, kept {st['kept']}", file=sys.stderr)
            next_report += 50_000_000
    flush()
    for f in parts:
        f.close()
    print("pass1 done " + json.dumps(st), file=sys.stderr)

    # Pass 2: per partition sort + dedupe, append to gzip TSV.
    unique = 0
    with gzip.open(os.path.join(out_dir, "edges.tsv.gz"), "wt", compresslevel=6, encoding="utf-8", newline="\n") as gz:
        for k in range(NPART):
            path = os.path.join(out_dir, f"part{k}.bin")
            pairs = np.fromfile(path, dtype="<i4").reshape(-1, 2).astype(np.int64)
            keys = np.unique((pairs[:, 0] << SHIFT) | pairs[:, 1])
            unique += keys.size
            for s in range(0, keys.size, 1_000_000):
                chunk = keys[s:s + 1_000_000]
                gz.write("".join(f"{a}\t{b}\n" for a, b in zip((chunk >> SHIFT).tolist(),
                                                               (chunk & ((1 << SHIFT) - 1)).tolist())))
            os.remove(path)
            print(f"partition {k}: {len(pairs)} -> cumulative unique {unique}", file=sys.stderr)
    st.update(unique_edges=unique, articles=len(id_of), redirects=len(redirects))
    top = sorted(unresolved.items(), key=lambda kv: -kv[1])[:100_000]
    with open_out(os.path.join(out_dir, "unresolved_targets.tsv")) as f:
        f.write("\n".join(f"{t}\t{c}" for t, c in top) + "\n")
    with open_out(os.path.join(out_dir, "stats.json")) as f:
        json.dump(st, f, indent=2)
    print(json.dumps(st), file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:4])

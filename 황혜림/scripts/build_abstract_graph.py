"""Build the title link graph over the HotpotQA 2017-10-01 intro-paragraph corpus
(the corpus HoVer retrieves from).

stdin : concatenated JSON lines of the withlinks-abstracts dump
argv  : pages_ns0_20171001.tsv (redirect table), output directory [--pass2-only]
Output: nodes.tsv  (node_id, title, wiki_id, n_sents)
        edges.tsv  (src_title, sent_idx, tgt_title, anchor) -- one row per link occurrence
        unresolved_targets.tsv (target, count), stats.json
--pass2-only reuses nodes.tsv and edges.raw.tmp from an earlier pass 1.
"""
import json
import os
import re
import sys
from collections import Counter

from title_utils import (BREAKS, BREAKS_NO_TAB, canon_map, decode_uri_component, html_unescape,
                         load_redirects, open_out, read_lines, resolve, stdin_lines)

LINK = re.compile(r'<a href="([^"]*)">(.*?)</a>', re.S)
TAG = re.compile(r"<[^>]+>")


def pass1(out_dir, raw_path):
    titles, n, links, bad = set(), 0, 0, 0
    with open_out(os.path.join(out_dir, "nodes.tsv")) as nodes, open_out(raw_path) as raw:
        for line in stdin_lines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except ValueError:
                bad += 1
                continue
            titles.add(o["title"])
            title = BREAKS.sub(" ", o["title"])
            nodes.write(f"{n}\t{title}\t{o['id']}\t{len(o['text'])}\n")
            n += 1
            buf = []
            for i, s in enumerate(o["text_with_links"]):
                for m in LINK.finditer(s):
                    t = decode_uri_component(m.group(1))
                    anchor = BREAKS.sub(" ", html_unescape(TAG.sub("", m.group(2))))
                    buf.append(f"{title}\t{i}\t{BREAKS.sub(' ', t)}\t{anchor}\n")
                    links += 1
            raw.writelines(buf)
            if n % 500_000 == 0:
                print(f"pass1: {n} docs, {links} links", file=sys.stderr)
    return titles, n, links, bad


def pass2(out_dir, raw_path, titles, redirects):
    canon = canon_map(titles)
    titles.clear()
    st = dict(direct=0, case_fixed=0, via_redirect=0, self_or_section=0, empty=0, unresolved=0)
    unresolved = Counter()
    with open_out(os.path.join(out_dir, "edges.tsv")) as out:
        for raw in read_lines(raw_path):
            # older pass-1 output may still carry U+2028 etc. inside fields
            src, sent, raw_tgt, anchor = BREAKS_NO_TAB.sub(" ", raw).split("\t")[:4]
            kind, title = resolve(raw_tgt, canon, redirects)
            if kind == "empty":
                st["empty"] += 1
            elif kind == "unresolved":
                st["unresolved"] += 1
                unresolved[title] += 1
            elif title == src:
                st["self_or_section"] += 1
            else:
                st[kind] += 1
                out.write(f"{src}\t{sent}\t{title}\t{anchor}\n")
    top = sorted(unresolved.items(), key=lambda kv: -kv[1])    # stable: ties keep first-seen order
    with open_out(os.path.join(out_dir, "unresolved_targets.tsv")) as f:
        f.write("\n".join(f"{t}\t{c}" for t, c in top) + "\n")
    os.remove(raw_path)
    return st


def main(pages_tsv, out_dir, mode=None):
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "edges.raw.tmp")
    redirects = load_redirects(pages_tsv)
    print(f"redirects loaded: {len(redirects)}", file=sys.stderr)
    if mode == "--pass2-only":
        titles, n = set(), 0
        for l in read_lines(os.path.join(out_dir, "nodes.tsv")):
            titles.add(l.split("\t")[1])
            n += 1
        links, bad = sum(1 for l in read_lines(raw_path) if l), 0
    else:
        titles, n, links, bad = pass1(out_dir, raw_path)
    print(f"pass1 done: {n} docs, {links} links, {bad} bad lines", file=sys.stderr)
    st = pass2(out_dir, raw_path, titles, redirects)
    kept = st["direct"] + st["case_fixed"] + st["via_redirect"]
    stats = dict(docs=n, bad_lines=bad, raw_links=links, kept_edges=kept, **st)
    with open_out(os.path.join(out_dir, "stats.json")) as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:4])

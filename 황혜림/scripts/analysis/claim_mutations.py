"""How NOT_SUPPORTED claims differ from their SUPPORTED siblings, split overlap and hop chaining.
Usage: python claim_mutations.py <data_dir>"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jscompat import js_entries, js_fixed, js_json, log  # noqa: E402

TOKEN = re.compile(r"[a-z0-9']+")


def pct(a, b):
    return js_fixed(100 * a / b, 1) + "%" if b else "NaN%"


def load(data_dir, s):
    with open(os.path.join(data_dir, f"hover_{s}_release_v1.1.json"), encoding="utf-8") as f:
        return json.load(f)


def tok(s):
    return TOKEN.findall(s.lower())


def diff(a, b):
    A, B = tok(a), tok(b)
    sa, sb = set(A), set(B)
    return [x for x in A if x not in sb], [x for x in B if x not in sa]


def main(data_dir):
    tr, dv, te = load(data_dir, "train"), load(data_dir, "dev"), load(data_dir, "test")
    # split leakage
    tr_h = {e["hpqa_id"] for e in tr}
    log("dev hpqa_id in train:", pct(sum(1 for e in dv if e["hpqa_id"] in tr_h), len(dv)))
    tr_t = {f[0] for e in tr for f in e["supporting_facts"]}
    dv_t = list(dict.fromkeys(f[0] for e in dv for f in e["supporting_facts"]))
    log("dev titles seen in train:", pct(sum(1 for t in dv_t if t in tr_t), len(dv_t)))
    tr_c = {e["claim"] for e in tr}
    log("exact dup claims dev∩train:", sum(1 for e in dv if e["claim"] in tr_c),
        " test∩train:", sum(1 for e in te if e["claim"] in tr_c))

    # edit distance between SUP and NOT claims of same hpqa & same hop
    g = {}
    for e in dv:
        g.setdefault(f"{e['hpqa_id']}_{e['num_hops']}", []).append(e)
    sizes, add_w, shown = [], {}, 0
    for _, grp in js_entries(g):
        S = [e for e in grp if e["label"] == "SUPPORTED"]
        N = [e for e in grp if e["label"] == "NOT_SUPPORTED"]
        for n in N:
            if not S:
                continue
            best = None
            for s in S:
                rm, add = diff(s["claim"], n["claim"])
                sz = len(rm) + len(add)
                if best is None or sz < best[0]:
                    best = (sz, rm, add, s)
            sz, rm, add, s = best
            sizes.append(sz)
            for w in add:
                add_w[w] = add_w.get(w, 0) + 1
            if sz <= 4 and shown < 8:
                shown += 1
                print(f"\n[{n['num_hops']}hop] SUP: {s['claim']}\n        NOT: {n['claim']}\n"
                      f"        -{','.join(rm)} +{','.join(add)}")
    sizes.sort()
    log("\nNOT claims paired with a SUP sibling:", len(sizes), "median token diff", sizes[len(sizes) >> 1],
        " <=2 tokens:", pct(sum(1 for x in sizes if x <= 2), len(sizes)),
        " <=4:", pct(sum(1 for x in sizes if x <= 4), len(sizes)))
    top = sorted(js_entries(add_w), key=lambda kv: -kv[1])[:25]
    log("most common inserted words in NOT edits:", js_json([list(kv) for kv in top]))

    # hop progression: same hpqa group, does a k-hop claim extend a (k-1)-hop claim?
    ext = tot = 0
    for e in (e for e in dv if e["num_hops"] > 2):
        lower = [x for x in dv if x["hpqa_id"] == e["hpqa_id"] and x["num_hops"] == e["num_hops"] - 1]
        if not lower:
            continue
        tot += 1
        tl = {f[0] for f in e["supporting_facts"]}
        if any(all(f[0] in tl for f in l["supporting_facts"]) for l in lower):
            ext += 1
    log("\nk-hop claims whose (k-1)-hop sibling doc set is a subset:", ext, "/", tot)

    # examples of 2, 3, 4 hop for one group
    groups = {}
    for e in dv:
        groups.setdefault(e["hpqa_id"], []).append(e)
    ex = next(v for _, v in js_entries(groups) if len({e["num_hops"] for e in v}) == 3)
    for e in sorted(ex, key=lambda e: e["num_hops"]):
        docs = " | ".join(dict.fromkeys(f[0] for f in e["supporting_facts"]))
        print(f"  {e['num_hops']}hop {e['label']}: {e['claim']}\n      docs: {docs}")


if __name__ == "__main__":
    main(sys.argv[1])

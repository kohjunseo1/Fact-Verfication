"""Dataset-level statistics of HoVer: labels x hops, supporting facts, titles, lexical cues,
hpqa_id reuse. Usage: python dataset_stats.py <data_dir>"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jscompat import (JS_DOT, JS_WS_CLASS, js_count, js_entries, js_fixed, js_json, js_sort_key,  # noqa: E402
                      js_sum, js_trim, log)

A, AI = re.ASCII, re.ASCII | re.IGNORECASE
WS_RUN = re.compile(JS_WS_CLASS + "+")
PAREN_LAZY = re.compile(JS_WS_CLASS + r"*\(" + JS_DOT + r"*?\)" + JS_WS_CLASS + "*")
NON_ALNUM = re.compile(r"[^a-z0-9 ]")
HAS_PAREN = re.compile(r"\(" + JS_DOT + r"*\)")
PAREN_GROUP = re.compile(r"\((" + JS_DOT + r"*)\)")
CUES = {
    "negation": re.compile(r"\b(not|never|no|n't|neither|nor)\b", AI),
    "both": re.compile(r"\bboth\b", AI),
    "comparative": re.compile(r"\b(older|younger|earlier|later|more|less|larger|smaller|bigger|longer|shorter|first|before|after)\b", AI),
    "relclause": re.compile(r"\b(that|who|which|whose)\b", AI),
    "also": re.compile(r"\b(also)\b", AI),
    "number": re.compile(r"\d", A),
}


def pct(a, b):
    return js_fixed(100 * a / b, 1) + "%" if b else "NaN%"


def norm(s):
    s = PAREN_LAZY.sub(" ", s.lower())
    return js_trim(WS_RUN.sub(" ", NON_ALNUM.sub(" ", s)))


def words(s):
    return len(WS_RUN.split(s))


def main(data_dir):
    for s in ("train", "dev", "test"):
        with open(os.path.join(data_dir, f"hover_{s}_release_v1.1.json"), encoding="utf-8") as f:
            d = json.load(f)
        print(f"\n===== {s} ({len(d)}) =====")
        log("keys:", ",".join(d[0].keys()))
        if not d[0].get("label"):
            print("no labels")
            continue
        # label x hops
        t = {}
        for e in d:
            row = t.setdefault(str(e["num_hops"]), {})
            row[e["label"]] = row.get(e["label"], 0) + 1
        log("hops x label:", js_json(dict(js_entries(t))))
        # docs per claim vs hops
        mism, sf_per = 0, []
        for e in d:
            titles = {f[0] for f in e["supporting_facts"]}
            mism += len(titles) != e["num_hops"]
            sf_per.append(len(e["supporting_facts"]) - len(titles))
        log("distinct docs != num_hops:", mism, pct(mism, len(d)))
        log("extra sentences beyond 1/doc:", js_json(js_count(sf_per)[:6]))
        # sentence index
        idx = [f[1] for e in d for f in e["supporting_facts"]]
        log("sent idx dist:", js_json(js_count(idx)[:8]), " idx0 share", pct(sum(1 for i in idx if i == 0), len(idx)))
        # title mentioned in claim (bridge entity explicit vs hidden)
        by_hop = {}
        for e in d:
            c = norm(e["claim"])
            titles = list(dict.fromkeys(f[0] for f in e["supporting_facts"]))
            hit = sum(1 for ti in titles if norm(ti) in c)
            by_hop.setdefault(str(e["num_hops"]), []).append(hit / len(titles))
        for h, a in js_entries(by_hop):
            print(f"hop{h}: mean frac titles in claim {js_fixed(js_sum(a) / len(a), 2)}, "
                  f"all titles present {pct(sum(1 for x in a if x == 1), len(a))}, "
                  f"none {pct(sum(1 for x in a if x == 0), len(a))}")
        # disambiguated titles
        all_t = [f[0] for e in d for f in e["supporting_facts"]]
        log("titles w/ parenthetical:", pct(sum(1 for x in all_t if HAS_PAREN.search(x)), len(all_t)))
        parens = [m.group(1) for m in map(PAREN_GROUP.search, all_t) if m and m.group(1)]
        log("top parentheticals:", js_json(js_count(parens)[:10]))
        log("top titles:", js_json(js_count(all_t)[:10]))
        # claim length
        for lab in ("SUPPORTED", "NOT_SUPPORTED"):
            w = [words(e["claim"]) for e in d if e["label"] == lab]
            log(lab, "avg words", js_fixed(sum(w) / len(w), 1))
        for h in (2, 3, 4):
            w = [words(e["claim"]) for e in d if e["num_hops"] == h]
            log("hop", h, "avg words", js_fixed(sum(w) / len(w), 1))
        # lexical cues by label
        for k, rx in CUES.items():
            r = []
            for lab in ("SUPPORTED", "NOT_SUPPORTED"):
                g = [e for e in d if e["label"] == lab]
                r.append(pct(sum(1 for e in g if rx.search(e["claim"])), len(g)))
            print(f"cue {k}: SUP {r[0]} / NOT {r[1]}")
        # hpqa_id reuse
        hq = js_count(e["hpqa_id"] for e in d)
        log("unique hpqa_id", len(hq), "claims per hpqa_id dist", js_json(js_count(x[1] for x in hq)))
        # label consistency within same hpqa group & same doc set
        g = {}
        for e in d:
            g.setdefault(e["hpqa_id"], []).append(e)
        multi = mixed = 0
        for _, v in js_entries(g):
            if len(v) > 1:
                multi += 1
                mixed += len({e["label"] for e in v}) > 1
        log("hpqa groups >1 claim:", multi, "with mixed labels:", mixed)
        ds = {}
        for e in d:
            k = "|".join(sorted(dict.fromkeys(f[0] for f in e["supporting_facts"]), key=js_sort_key))
            ds.setdefault(k, []).append(e["label"])
        dsm = [v for _, v in js_entries(ds) if len(v) > 1]
        log("identical doc-sets shared by >1 claim:", len(dsm), "mixed label:", sum(1 for v in dsm if len(set(v)) > 1))
        # first word / template
        log("top claim openings:", js_json(js_count(" ".join(e["claim"].split(" ")[:2]) for e in d)[:12]))


if __name__ == "__main__":
    main(sys.argv[1])

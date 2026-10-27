"""Finer logical-structure classes of 2-hop HoVer claims (bridge / parallel / comparison / explicit).
Usage: python hop2_structure.py <data_dir> <split> [ex]"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jscompat import JS_DOT, JS_WS_CLASS, js_entries, js_fixed, js_json, js_sort_key, js_trim, log  # noqa: E402

A, AI = re.ASCII, re.ASCII | re.IGNORECASE
NON_ALNUM = re.compile(r"[^a-z0-9 ]")
WS_RUN = re.compile(JS_WS_CLASS + "+")
PAREN_LAZY = re.compile(JS_WS_CLASS + r"*\(" + JS_DOT + r"*?\)" + JS_WS_CLASS + "*")
PAREN_GROUP = re.compile(r"\((" + JS_DOT + r"*)\)")
RE = {
    "both": re.compile(r"\bboth\b|\bneither\b|\beither\b", AI),
    "negation": re.compile(r"\b(not|never|no|neither|nor)\b|n't\b", AI),
    "rel": re.compile(r"\b(that|who|whom|which|whose|where)\b|\b(in|at|on|from|by|for|of|with) (which|whom)\b", AI),
    "part": re.compile(r"\b(directed|produced|written|founded|created|owned|released|recorded|starring|featuring|named|known|located|portrayed|composed|designed|built|developed|published|hosted|managed|formed|based|born|played|sung|performed|acquired|operated|coached|founded)( \w+)? (by|in|as|for|after|on|at|from)\b|\bstarring\b|\bfeaturing\b", AI),
    "poss": re.compile(r"\bthe (\w+ ){0,3}of (the|a|an|this)\b|'s (\w+ ){0,2}(film|album|song|book|novel|band|wife|husband|son|daughter|father|mother|director|author)\b", AI),
    "appos": re.compile(r"^[^,]+, (a|an|the) [^,]+,", AI),
    "and": re.compile(r"\band\b", AI),
}
CMP = re.compile(r"\b(than|the same|different|older|younger|earlier|predate[sd]?|outlive[sd]?|before)\b", AI)
PARALLEL_AND = re.compile(r"^[^,.]{2,80}? and [^,.]{2,80}? (are|were|have|had|both|each)\b", A)


def pct(a, b):
    return js_fixed(100 * a / b, 0) + "%" if b else "-"


def norm(s):
    return " " + js_trim(WS_RUN.sub(" ", NON_ALNUM.sub(" ", s.lower()))) + " "


def core(t):
    return js_trim(PAREN_LAZY.sub(" ", t))


def mentioned(title, claim):
    """Full core title in the claim, or (person-like) its last token of >= 4 chars."""
    c, t = norm(claim), norm(core(title))
    if t in c:
        return True
    toks = js_trim(t).split(" ")
    if len(toks) >= 2:
        last = toks[-1]
        if len(last) >= 4 and f" {last} " in c:
            return True
    return False


def cls(e):
    c = e["claim"]
    titles = list(dict.fromkeys(f[0] for f in e["supporting_facts"]))
    n = sum(1 for t in titles if mentioned(t, c))
    sub = None
    if CMP.search(c):
        top = "C. 비교 R(a,b)"
    elif RE["both"].search(c) or (n == 2 and PARALLEL_AND.search(c)):
        top = "B. 병렬 P(a)∧P(b)"
    elif n == 2:
        top = "D. 명시 연결 R(a,b) / P(a)∧Q(b)"
    else:
        top = "A. 브리지 합성 P(f(a))"
    if top.startswith("A"):
        sub = ("관계절(that/who/which)" if RE["rel"].search(c) else "분사구(directed by …)" if RE["part"].search(c)
               else "소유·of 연쇄(the X of Y)" if RE["poss"].search(c) else "동격" if RE["appos"].search(c)
               else "기타(서술형 묘사)")
    if top.startswith("D"):
        sub = ("b에 관계절·동격 부가" if RE["rel"].search(c) or RE["appos"].search(c)
               else "and 연결" if RE["and"].search(c) else "단일 관계 R(a,b)")
    return {"e": e, "top": top, "sub": sub, "n": n, "neg": bool(RE["negation"].search(c))}


def main(data_dir, split, show_ex=None):
    with open(os.path.join(data_dir, f"hover_{split}_release_v1.1.json"), encoding="utf-8") as f:
        d = [e for e in json.load(f) if e["num_hops"] == 2]
    rs = [cls(e) for e in d]
    print(f"\n##### {split} 2-hop ({len(d)}) #####")
    log("제목 언급 수: 0개", pct(sum(1 for r in rs if r["n"] == 0), len(d)),
        " 1개", pct(sum(1 for r in rs if r["n"] == 1), len(d)),
        " 2개", pct(sum(1 for r in rs if r["n"] == 2), len(d)))
    print("\n유형 | n | 비율 | SUP% | 부정어% | 평균단어")
    for t in sorted(dict.fromkeys(r["top"] for r in rs), key=js_sort_key):
        g = [r for r in rs if r["top"] == t]
        w = [len(WS_RUN.split(r["e"]["claim"])) for r in g]
        print(f"{t} | {len(g)} | {pct(len(g), len(d))} | "
              f"{pct(sum(1 for r in g if r['e']['label'] == 'SUPPORTED'), len(g))} | "
              f"{pct(sum(1 for r in g if r['neg']), len(g))} | {js_fixed(sum(w) / len(w), 1)}")
        for s in dict.fromkeys(r["sub"] for r in g if r["sub"]):
            h = [r for r in g if r["sub"] == s]
            print(f"   └ {s} | {len(h)} | {pct(len(h), len(g))} | "
                  f"SUP {pct(sum(1 for r in h if r['e']['label'] == 'SUPPORTED'), len(h))}")
    # bridge: which doc is hidden -> sentence idx of hidden doc
    bridge = [r for r in rs if r["top"].startswith("A") and r["n"] == 1]
    hid_idx = [f[1] for r in bridge for f in r["e"]["supporting_facts"] if not mentioned(f[0], r["e"]["claim"])]
    vis_idx = [f[1] for r in bridge for f in r["e"]["supporting_facts"] if mentioned(f[0], r["e"]["claim"])]
    print(f"\n브리지형: 숨은 문서 근거문장 idx0 {pct(hid_idx.count(0), len(hid_idx))} / "
          f"명시 문서 idx0 {pct(vis_idx.count(0), len(vis_idx))}")
    cnt = {}
    for r in bridge:
        for f in r["e"]["supporting_facts"]:
            if not mentioned(f[0], r["e"]["claim"]):
                m = PAREN_GROUP.search(f[0])
                k = m.group(1) if m else "(없음)"
                cnt[k] = cnt.get(k, 0) + 1
    top = sorted(js_entries(cnt), key=lambda kv: -kv[1])[:8]
    log("숨은 문서 제목 괄호 유형 top:", js_json([list(kv) for kv in top]))
    if show_ex:
        s = 5

        def rnd():
            nonlocal s
            s = (s * 16807) % 2147483647
            return s / 2147483647
        keys = sorted(dict.fromkeys(r["top"] + " :: " + (r["sub"] or "") for r in rs), key=js_sort_key)
        for k in keys:
            g = [x for x in rs if x["top"] + " :: " + (x["sub"] or "") == k]
            print("\n# " + k)
            for _ in range(3):
                x = g[int(rnd() * len(g))]
                docs = " | ".join(("" if mentioned(t, x["e"]["claim"]) else "*") + t
                                  for t in dict.fromkeys(f[0] for f in x["e"]["supporting_facts"]))
                print(f"  [{x['e']['label'][0]}] {x['e']['claim']}\n      docs: {docs}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

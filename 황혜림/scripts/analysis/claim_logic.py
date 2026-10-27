"""Rule-based logical-structure classes of HoVer claims (composition, conjunction, comparison, ...).
Usage: python claim_logic.py <data_dir> <split> [ex]"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jscompat import JS_DOT, JS_WS_CLASS, js_fixed, js_json, js_sum, js_trim, log  # noqa: E402

A, AI = re.ASCII, re.ASCII | re.IGNORECASE
WS = JS_WS_CLASS


def rx(p, flags=A):
    return re.compile(p, flags)


# sentence split (avoid abbreviations / initials); JS used a variable-width lookbehind
ABBR = rx(r"\b(Jr|Sr|Dr|Mr|Mrs|Ms|St|U\.S|Inc|Co|Ltd|vs|No)\.")
INITIAL = rx(r"\b([A-Z])\.")
SPLIT = rx(r"(?:(?<=[.!?])|(?<=[.!?][\"”]))" + WS + r"+(?=[A-Z\"“])")


def sents(c):
    c = INITIAL.sub(r"\1<D>", ABBR.sub(r"\1<D>", c))
    return [s for s in SPLIT.split(c) if js_trim(s)]


ANAPHORA_1 = rx(r"^(This|That|These|Those|It|He|She|They|His|Her|Its|Their|The (same )?(latter|former)|Both|So |Another|The (\w+ ){0,2}(also|too)\b)")
ANAPHORA_2 = rx(r"^The \w+(" + WS + r"\w+)? (also|too)\b")
RELCLAUSE = rx(r"\b(that|who|whom|which|whose)\b|\b(in|at|on|from|by|for|of|with) (which|whom)\b|\bwhere\b(?!" + JS_DOT + r"*\bdoesn'?t\b)", AI)
PARTICIPLE = rx(r"\b(directed|produced|written|founded|created|owned|released|recorded|starring|featuring|named|based|known|located|born|played|portrayed|composed|designed|built|developed|published|hosted|managed|formed|performed|sung)( \w+)? (by|in|as|for|after|on|at)\b", AI)
POSS_CHAIN = rx(r"\bthe (\w+ ){0,2}(of|for) the\b", AI)
BOTH = rx(r"\bboth\b|\bneither\b|\beither\b", AI)
PARALLEL_1 = rx(r"\b(so (is|was|were|did|does|has|had)|as (was|is|did|does)|as well|too)\b[.,]?", AI)
ALSO = rx(r"\balso\b", AI)
COORD_SUBJ = rx(r"^[^,]{3,80}? and [^,]{3,60}? (are|were|have|had|both)\b")
COMPARISON = rx(r"\b(than|older|younger|earlier|later|longer|shorter|taller|larger|smaller|bigger|higher|lower|more|less|fewer|first|before|after|same|different|predate[sd]?|outlive[sd]?)\b", AI)
NEGATION = rx(r"\b(not|never|no|none|neither|nor|without)\b|n't\b|n’t\b", AI)
CONTRAST = rx(r"\b(but|while|whereas|however|unlike|although|though)\b|,? where [^.]*\b(n't|not)\b", AI)
EXISTENTIAL = rx(r"^(An?|One|Some|Another|There (is|was|are|were)) ")
DISJ = rx(r"\b(either|or)\b", AI)
OR_SO = rx(r"\bor so\b")
NUMERIC = rx(r"\b\d{1,4}\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million)\b", AI)

F = {
    "multi": lambda c: len(sents(c)) > 1,
    "anaphora": lambda c: any(ANAPHORA_1.search(js_trim(s)) or ANAPHORA_2.search(js_trim(s)) for s in sents(c)[1:]),
    "relclause": lambda c: bool(RELCLAUSE.search(c)),
    "participle": lambda c: bool(PARTICIPLE.search(c)),
    "possessiveChain": lambda c: bool(POSS_CHAIN.search(c)),
    "both": lambda c: bool(BOTH.search(c)),
    "parallel": lambda c: bool(PARALLEL_1.search(c) or ALSO.search(c)),
    "coordSubj": lambda c: bool(COORD_SUBJ.search(c)),
    "comparison": lambda c: bool(COMPARISON.search(c)),
    "negation": lambda c: bool(NEGATION.search(c)),
    "contrast": lambda c: bool(CONTRAST.search(c)),
    "existential": lambda c: any(EXISTENTIAL.search(js_trim(s)) for s in sents(c)),
    "disjunction": lambda c: bool(DISJ.search(c)) and not OR_SO.search(c),
    "numeric": lambda c: bool(NUMERIC.search(c)),
}

# nesting depth: number of bridge markers
D_REL = rx(r"\b(that|who|whom|which|whose|where)\b", AI)
D_PART = rx(r"\b(directed|produced|written|founded|created|owned|released|recorded|starring|featuring|named|known|located|portrayed|composed|designed|built|developed|published|hosted|managed|formed) (by|for|as|in)\b", AI)
D_OF = rx(r"\bthe \w+( \w+)? of (the|a|an)\b", AI)
COMPARE_KEY = rx(r"\b(than|same|different|before|after|earlier|later|older|younger|predate|outlive)", AI)
PARALLEL_KEY = rx(r"\b(so (is|was|did)|as well|too)\b", AI)
CONJ_AND = rx(r",? and (also |later |was |is |has |had |were |\w+ed\b)", AI)


def depth(c):
    return len(D_REL.findall(c)) + len(D_PART.findall(c)) + len(D_OF.findall(c))


def form(c):
    f = lambda k: F[k](c)  # noqa: E731
    if f("comparison") and COMPARE_KEY.search(c):
        return "COMPARE  R(f(a),g(b))"
    if (f("negation") or f("contrast")) and (f("parallel") or f("contrast") or f("multi")):
        return "CONTRAST P(a)∧¬P(b)"
    if f("both") or f("coordSubj") or PARALLEL_KEY.search(c):
        return "PARALLEL P(a)∧P(b)"
    if f("multi"):
        return "CONJ-SEQ P(x)∧Q(x) (multi-sentence)"
    if CONJ_AND.search(c):
        return "CONJ P(x)∧Q(x) (and)"
    if depth(c) > 0:
        return "COMPOSE P(f(g(a)))"
    return "ATOMIC/APPOSITIVE"


def pct(a, b):
    return js_fixed(100 * a / b, 0) + "%" if b else "-"


def main(data_dir, split, show_ex=None):
    with open(os.path.join(data_dir, f"hover_{split}_release_v1.1.json"), encoding="utf-8") as f:
        d = json.load(f)
    L, H = ["SUPPORTED", "NOT_SUPPORTED"], [2, 3, 4]
    print(f"\n##### {split} ({len(d)}) #####")
    print("\n-- primary logical form: share by hop | SUP rate --")
    forms = [form(e["claim"]) for e in d]
    rows = []
    for fm in dict.fromkeys(forms):
        g = [e for e, x in zip(d, forms) if x == fm]
        rows.append([fm, len(g),
                     *(pct(sum(1 for e in g if e["num_hops"] == h), sum(1 for e in d if e["num_hops"] == h)) for h in H),
                     pct(sum(1 for e in g if e["label"] == "SUPPORTED"), len(g))])
    rows.sort(key=lambda r: -r[1])
    print("form | n | 2hop | 3hop | 4hop | SUP%")
    for r in rows:
        print(" | ".join(str(x) for x in r))
    print("\n-- feature prevalence: 2hop/3hop/4hop || SUP/NOT --")
    for k, fn in F.items():
        by_hop = "/".join(pct(sum(1 for e in g if fn(e["claim"])), len(g))
                          for g in ([e for e in d if e["num_hops"] == h] for h in H))
        by_lab = "/".join(pct(sum(1 for e in g if fn(e["claim"])), len(g))
                          for g in ([e for e in d if e["label"] == l] for l in L))
        log(k.ljust(16), by_hop.ljust(14), "||", by_lab)
    print("\n-- nesting depth & sentence count by hop --")
    for h in H:
        g = [e for e in d if e["num_hops"] == h]
        dp = [depth(e["claim"]) for e in g]
        sc = [len(sents(e["claim"])) for e in g]
        dist = [pct(sum(1 for x in sc if (x >= 4 if n == 4 else x == n)), len(g)) for n in (1, 2, 3, 4)]
        print(f"hop{h}: mean depth {js_fixed(js_sum(dp) / len(g), 2)}, depth0 {pct(dp.count(0), len(g))}, "
              f"mean sents {js_fixed(js_sum(sc) / len(g), 2)}, sents dist {js_json(dist)}")
    if show_ex:
        print("\n-- examples --")
        s = 11

        def r():
            nonlocal s
            s = (s * 16807) % 2147483647
            return s / 2147483647
        for fm in (row[0] for row in rows):
            print("\n# " + fm)
            g = [e for e, x in zip(d, forms) if x == fm]
            for _ in range(3):
                e = g[int(r() * len(g))]
                print(f"  [{e['num_hops']}h {e['label'][0]}] {e['claim']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

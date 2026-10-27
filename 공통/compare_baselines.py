"""세 명의 베이스라인을 같은 규칙으로 채점해 비교표를 만든다 (공식 dev 2-hop 1,126건).

사용법:
  python 공통/compare_baselines.py --gold hover_dev_release_v1.1.json \
      한기선=hangiseon.jsonl 황혜림=hwanghyerim.jsonl 고준서=gojunseo.jsonl
  python 공통/compare_baselines.py        # 인자 없으면 자체 점검만 실행

예측 파일: dev 2-hop claim마다 한 줄(jsonl). 아래 필드 이름 중 아무거나 쓰면 된다.
  uid
  라벨      pred_label | predicted_label | pred      "SUPPORTED"/"NOT_SUPPORTED" 또는 1/0
  문서      docs | ranked_documents | reranked | opened    순위순 제목 목록
  근거      evidence | predicted_evidence | pred_sents     순위순 [제목, 문장 번호, ...] 또는 {"title", "sent_id"|"sentence_id"}
근거는 각자 val로 고른 제출 개수만큼 넣는다. 채점기가 공식 규칙(앞 k+1개 문서, 문서마다 앞 2문장)으로 거른다.
"""
import json
import random
import sys
import unicodedata
from statistics import mean

LABEL_KEYS = ("pred_label", "predicted_label", "pred")
DOC_KEYS = ("docs", "ranked_documents", "reranked", "opened")
EVIDENCE_KEYS = ("evidence", "predicted_evidence", "pred_sents")


def key(title):
    return unicodedata.normalize("NFC", str(title)).casefold()


def pick(row, names):
    for n in names:
        if n in row:
            return row[n]
    raise KeyError(f"{row.get('uid')}: {names} 중 하나가 필요하다")


def to_label(v):
    return int(v) if isinstance(v, (int, bool)) else int(str(v).upper() == "SUPPORTED")


def to_fact(e):
    if isinstance(e, dict):
        return key(e["title"]), int(e.get("sent_id", e.get("sentence_id")))
    return key(e[0]), int(e[1])


def official_filter(evidence, num_hops):
    """HoVer 홈페이지 규칙: 처음 등장한 순서로 앞 k+1개 문서, 문서마다 앞 2문장."""
    per_doc, kept = {}, set()
    for t, i in map(to_fact, evidence):
        if t not in per_doc:
            if len(per_doc) == num_hops + 1:
                continue
            per_doc[t] = 0
        if per_doc[t] < 2 and (t, i) not in kept:
            per_doc[t] += 1
            kept.add((t, i))
    return kept


def macro_f1(gold, pred):
    f1s = []
    for c in (0, 1):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        denom = sum(g == c for g in gold) + sum(p == c for p in pred)
        f1s.append(2 * tp / denom if denom else 0.0)
    return mean(f1s)


def per_claim(claim, row):
    gold = {(key(t), int(i)) for t, i in claim["supporting_facts"]}
    gold_docs = {t for t, _ in gold}
    docs = [key(t) for t in pick(row, DOC_KEYS)]
    kept = official_filter(pick(row, EVIDENCE_KEYS), claim["num_hops"])
    tp = len(kept & gold)
    p = tp / len(kept) if kept else 0.0
    r = tp / len(gold)
    ev_ok = all(any(g in kept for g in gold if g[0] == d) for d in gold_docs)
    y, yhat = to_label(claim["label"]), to_label(pick(row, LABEL_KEYS))
    return {
        "y": y, "yhat": yhat, "ev_ok": ev_ok, "hover": ev_ok and y == yhat,
        "hit2": gold_docs <= set(docs[:2]), "hit5": gold_docs <= set(docs[:5]), "hit10": gold_docs <= set(docs[:10]),
        "ev_p": p, "ev_r": r, "ev_f1": 2 * p * r / (p + r) if tp else 0.0, "ev_em": kept == gold, "n_ev": len(kept),
    }


def summarize(rows):
    ys, yh = [r["y"] for r in rows], [r["yhat"] for r in rows]
    ok = [r for r in rows if r["ev_ok"]]
    ns = [r for r in rows if r["y"] == 0]
    return {
        "문서 all-hit@2": mean(r["hit2"] for r in rows),
        "문서 all-hit@5": mean(r["hit5"] for r in rows),
        "문서 all-hit@10": mean(r["hit10"] for r in rows),
        "근거 제출 문장 수(평균)": mean(r["n_ev"] for r in rows),
        "근거 precision": mean(r["ev_p"] for r in rows),
        "근거 recall": mean(r["ev_r"] for r in rows),
        "근거 F1": mean(r["ev_f1"] for r in rows),
        "근거 EM": mean(r["ev_em"] for r in rows),
        "evidence_ok (정답 문서마다 정답 문장 ≥1)": mean(r["ev_ok"] for r in rows),
        "accuracy": mean(y == p for y, p in zip(ys, yh)),
        "macro-F1": macro_f1(ys, yh),
        "NOT_SUPPORTED recall": mean(r["yhat"] == 0 for r in ns) if ns else float("nan"),
        "evidence_ok claim의 accuracy": mean(r["y"] == r["yhat"] for r in ok) if ok else float("nan"),
        "HoVer score (공식 규칙)": mean(r["hover"] for r in rows),
    }


def bootstrap_ci(rows, metric, n=1000, seed=0):
    # ponytail: 단순 percentile bootstrap. 두 시스템 차이의 유의성이 필요하면 paired bootstrap으로 바꾼다
    rng = random.Random(seed)
    vals = sorted(summarize(rng.choices(rows, k=len(rows)))[metric] for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def score_file(claims, path):
    preds = {str(r["uid"]): r for r in map(json.loads, open(path, encoding="utf-8")) if r}
    missing = [u for u in claims if u not in preds]
    if missing:
        raise SystemExit(f"{path}: dev 2-hop {len(claims)}건 중 {len(missing)}건 예측 없음 (예: {missing[0]})")
    return [per_claim(c, preds[u]) for u, c in claims.items()]


def main(argv):
    gold_path = argv[argv.index("--gold") + 1]
    systems = [a.split("=", 1) for a in argv if "=" in a]
    claims = {str(c["uid"]): c for c in json.load(open(gold_path, encoding="utf-8")) if c["num_hops"] == 2}
    results = {name: score_file(claims, path) for name, path in systems}
    tables = {name: summarize(rows) for name, rows in results.items()}
    names = list(tables)
    print(f"공식 dev 2-hop {len(claims)}건, 같은 채점 규칙\n")
    print("| 지표 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for m in tables[names[0]]:
        print(f"| {m} | " + " | ".join(f"{tables[n][m]:.3f}" if m != "근거 제출 문장 수(평균)" else f"{tables[n][m]:.1f}" for n in names) + " |")
    for m in ("macro-F1", "HoVer score (공식 규칙)"):
        print(f"| {m} 95% CI | " + " | ".join("%.3f–%.3f" % bootstrap_ci(results[n], m) for n in names) + " |")


def _selftest():
    claim = {"uid": "a", "label": "SUPPORTED", "num_hops": 2, "supporting_facts": [["A", 0], ["B", 1]]}
    good = {"uid": "a", "pred_label": "SUPPORTED", "docs": ["A", "B"], "evidence": [["A", 0], ["A", 1], ["A", 2], ["B", 1]]}
    r = per_claim(claim, good)
    assert r["ev_ok"] and r["hover"] and r["hit2"] and r["n_ev"] == 3  # A의 3번째 문장은 규칙상 버려짐
    # 4번째 문서로 밀린 정답 근거는 k+1=3개 문서 제한에 걸린다
    late = {"uid": "a", "predicted_label": 1, "ranked_documents": ["A", "C", "D", "B"],
            "predicted_evidence": [{"title": "A", "sentence_id": 0}, ["C", 0], ["D", 0], ["B", 1]]}
    r = per_claim(claim, late)
    assert not r["ev_ok"] and not r["hover"] and not r["hit2"] and r["hit5"]
    assert macro_f1([1, 0], [1, 0]) == 1.0 and to_label("NOT_SUPPORTED") == 0
    print("selftest ok")


if __name__ == "__main__":
    main(sys.argv[1:]) if len(sys.argv) > 1 else _selftest()

"""노트북 05 → 08 → 06 → 07의 추론 부분을 claim 하나에 대해 그대로 이어 붙인 파이프라인.

다른 점은 두 가지다.
- 1차 검색: 공식 TF-IDF 결과는 HoVer claim에만 있어서, 새 claim은 FTS5 BM25로 찾는다 (build_index.py)
- 문장 분리: CoreNLP(Java) 대신 정규식. 정답 문장 번호를 맞출 필요가 없어서다
"""
import os
import re
import sqlite3
import threading
import time
import unicodedata
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WIKI_DB = os.environ.get('WIKI_DB', '/data/wiki_wo_links.db')
INDEX_DB = os.environ.get('INDEX_DB', '/data/wiki_index.db')
MODEL_DIR = os.environ.get('MODEL_DIR', '/models')
LABELS = ['NOT_SUPPORTED', 'SUPPORTED']
K_CAND, B, K_DOC, K_SENT = 20, 2, 5, 5  # 05 재정렬 후보 수, 08 hop 1 문서 수, 06 문서 수, 07 근거 문장 수
N_DIV = 2  # 06: 서로 다른 문서 N_DIV개의 1위 문장을 맨 앞에 둔다
BATCH = 32

nfc = lambda s: unicodedata.normalize('NFC', s)
con = sqlite3.connect(f'file:{INDEX_DB}?mode=ro', uri=True, check_same_thread=False)
con.execute(f"ATTACH DATABASE 'file:{WIKI_DB}?mode=ro' AS w")
# ponytail: 요청 하나씩 처리하는 전역 락. CPU 추론이라 동시에 돌려도 빨라지지 않는다
lock = threading.Lock()

models = {}
for name in ['doc_rerank', 'hop2', 'sent_select', 'verifier']:
    path = f'{MODEL_DIR}/{name}'
    models[name] = (AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).eval())


@torch.inference_mode()
def score(name, first, second, max_len, truncation=True):
    """(first[i], second[i]) 쌍마다 양성(1) 확률"""
    tok, model = models[name]
    out = []
    for i in range(0, len(first), BATCH):
        enc = tok(first[i:i + BATCH], second[i:i + BATCH], truncation=truncation, max_length=max_len,
                  padding=True, return_tensors='pt')
        out += model(**enc).logits.softmax(-1)[:, 1].tolist()
    return out


# ---------- 위키 ----------
@lru_cache(maxsize=10_000)
def doc_text(title):  # title은 위키 DB의 id 그대로 (NFD)
    row = con.execute('SELECT text FROM w.documents WHERE id = ?', (title,)).fetchone()
    return row[0] if row else ''


STOP = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'and', 'or', 'to', 'for', 'by', 'with', 'from', 'as', 'is', 'was',
        'he', 'she', 'it', 'his', 'her', 'its', 'this', 'that', 'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'}


def bm25(claim, k=K_CAND):
    words = dict.fromkeys(w for w in re.findall(r'\w+', claim.lower()) if w not in STOP and len(w) > 1)
    if not words:
        return []
    q = ' OR '.join(f'"{w}"' for w in words)
    rows = con.execute('SELECT d.id FROM (SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?) f '
                       'JOIN w.documents d ON d.rowid = f.rowid', (q, k)).fetchall()
    return [t for (t,) in rows]


# ---------- hop 2 후보 (08 노트북의 mentioned_titles) ----------
TOK = re.compile(r"[\w][\w'.\-&$]*|[^\w\s]")


def mentioned_titles(s, max_n=8):
    toks = TOK.findall(nfc(s))
    spans = set()
    for i in range(len(toks)):
        if not (toks[i][0].isupper() or toks[i][0].isdigit()):
            continue
        for n in range(1, min(max_n, len(toks) - i) + 1):
            span = re.sub(r" ([^\w\s])", r"\1", ' '.join(toks[i:i + n])).strip(" .,;:'\"").lower()
            if not (n == 1 and (span in STOP or len(span) < 3)):
                spans.add(span)
    spans, out = list(spans), set()
    for i in range(0, len(spans), 500):
        chunk = spans[i:i + 500]
        out.update(t for (t,) in con.execute(
            f'SELECT d.id FROM titles t JOIN w.documents d ON d.rowid = t.rid WHERE t.base IN ({",".join("?" * len(chunk))})', chunk))
    return out


@lru_cache(maxsize=2_000)
def doc_mentions(title):
    return frozenset(mentioned_titles(doc_text(title)))


def merge(paths, hop1, k=20):
    # 08 노트북과 같다: 최고 경로의 두 문서를 맨 앞에, 나머지는 hop 1 순위와 다음 경로들을 번갈아 채운다
    out = []
    def add(t):
        if t not in out and len(out) < k:
            out.append(t)
    for t in (paths[0] if paths else ()):
        add(t)
    rest = [t for pair in paths[1:] for t in pair]
    for i in range(max(len(hop1), len(rest))):
        if i < len(hop1): add(hop1[i])
        if i < len(rest): add(rest[i])
    return out


# ponytail: 정규식 문장 분리. "P. S. Krøyer" 같은 이니셜은 넘기지만 약어가 많은 문서는 잘못 자를 수 있다. 필요하면 CoreNLP로 바꾼다
SENT_END = re.compile(r'(?<=[.!?])(?<!\b[A-Z]\.)(?<!\b(?:Mr|Dr|St|Jr|Sr|Ms|Mt|No)\.)\s+(?=["\'(]?[A-Z0-9])')


def split_sents(text):
    return [s for s in SENT_END.split(text) if s.strip()]


# ---------- 파이프라인 ----------
def verify(claim):
    with lock:
        t, timing = time.time(), {}
        def lap(name):
            nonlocal t
            timing[name] = round(time.time() - t, 2); t = time.time()

        cands = bm25(claim)
        lap('bm25')
        if not cands:
            return None

        s1 = score('doc_rerank', [claim] * len(cands), [f'{nfc(d)}. {doc_text(d)}' for d in cands], 256, 'only_second')
        reranked = sorted(zip(cands, s1), key=lambda x: -x[1])
        lap('rerank')

        pairs = [(a, sa, c) for a, sa in reranked[:B] for c in (mentioned_titles(claim) | doc_mentions(a)) - {a}]
        s2 = score('hop2', [f'{claim} | {nfc(a)}. {nfc(doc_text(a))}' for a, _, _ in pairs],
                   [f'{nfc(c)}. {nfc(doc_text(c))}' for _, _, c in pairs], 256)
        paths = sorted(((sa * p, a, c, sa, p) for (a, sa, c), p in zip(pairs, s2)), key=lambda x: -x[0])
        docs = merge([(a, c) for _, a, c, _, _ in paths], [d for d, _ in reranked])[:K_DOC]
        lap('hop2')

        sents = [(d, i, s) for d in docs for i, s in enumerate(split_sents(doc_text(d)))]
        s3 = score('sent_select', [claim] * len(sents), [f'{nfc(d)}. {s}' for d, _, s in sents], 128)
        ranked = sorted(zip(sents, s3), key=lambda x: -x[1])
        best = {}
        for x in ranked:
            best.setdefault(x[0][0], x)  # 문서별 1위 문장, 점수 순
        first = list(best.values())[:N_DIV]
        top = (first + [x for x in ranked if x not in first])[:K_SENT]
        lap('sentence')

        tok, model = models['verifier']
        enc = tok(claim, ' '.join(s for (_, _, s), _ in top), truncation=True, max_length=384, return_tensors='pt')
        with torch.inference_mode():
            p = model(**enc).logits.softmax(-1)[0].tolist()
        lap('verify')

    return {
        'claim': claim,
        'label': LABELS[int(p[1] > p[0])],
        'prob_supported': round(p[1], 4),
        'bm25': [nfc(d) for d in cands],
        'reranked': [{'title': nfc(d), 'score': round(s, 4)} for d, s in reranked[:K_DOC]],
        'paths': [{'hop1': nfc(a), 'hop2': nfc(c), 'score': round(s, 4), 'p_hop1': round(sa, 4), 'p_link': round(pl, 4)}
                  for s, a, c, sa, pl in paths[:5]],
        'n_hop2_pairs': len(pairs),
        'docs': [nfc(d) for d in docs],
        'n_sents': len(sents),
        'evidence': [{'title': nfc(d), 'sid': i, 'sent': nfc(s), 'score': round(sc, 4),
                      'first': ((d, i, s), sc) in first} for (d, i, s), sc in top],
        'timing': timing,
    }

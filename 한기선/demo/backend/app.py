# 팀 공통 모델 컨테이너 규격: POST /predict {claim} → {label, evidence, latency_ms}, GET /health, GET /info
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pipeline

app = FastAPI(title='HoVer 2-hop fact verification')


class PredictRequest(BaseModel):
    claim: str


@app.get('/health')
def health():
    return {'ok': True}


@app.get('/info')
def info():
    return {
        'name': 'hangiseon-2hop-baseline',
        'pipeline': 'BM25(FTS5) → roberta 문서 재정렬 → hop2 경로 점수 → roberta 문장 선택 → roberta 검증기',
        'models': list(pipeline.models),
        'params': {'k_cand': pipeline.K_CAND, 'hop1': pipeline.B, 'k_doc': pipeline.K_DOC, 'k_sent': pipeline.K_SENT},
    }


@app.post('/predict')
def predict(req: PredictRequest):
    claim = req.claim.strip()
    if not 5 <= len(claim) <= 500:
        raise HTTPException(400, 'claim은 5~500자여야 합니다')
    t = time.time()
    out = pipeline.verify(claim)
    if out is None:
        raise HTTPException(422, '검색할 단어가 없습니다')
    return {**out, 'latency_ms': round((time.time() - t) * 1000)}

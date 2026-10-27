# 팀 공통 모델 컨테이너 규격: POST /predict {claim} → {label, evidence, latency_ms}, GET /health, GET /info
import itertools
import json
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import pipeline
import torch

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
        'colab': 'COLAB_RELEASE_TAG' in os.environ,  # Colab 런타임에만 있는 환경 변수
        'device': pipeline.DEVICE,
        'gpu': torch.cuda.get_device_name() if pipeline.DEVICE == 'cuda' else None,
        'params': {'k_cand': pipeline.K_CAND, 'hop1': pipeline.B, 'k_doc': pipeline.K_DOC, 'k_sent': pipeline.K_SENT},
    }


def check(req: PredictRequest):
    claim = req.claim.strip()
    if not 5 <= len(claim) <= 500:
        raise HTTPException(400, 'claim은 5~500자여야 합니다')
    return claim


@app.post('/predict_stream')
def predict_stream(req: PredictRequest):
    # 데모 화면용: 단계가 끝날 때마다 {"stage": ..., "time": ..., 결과...} 한 줄(NDJSON)
    gen = pipeline.stages(check(req))
    first = next(gen, None)  # BM25까지는 응답 전에 돌려서, 검색어가 없으면 422로 돌려준다
    if first is None:
        raise HTTPException(422, '검색할 단어가 없습니다')

    def lines():
        for name, part in itertools.chain([first], gen):
            yield json.dumps({'stage': name, **part}, ensure_ascii=False) + '\n'
    return StreamingResponse(lines(), media_type='application/x-ndjson')


@app.post('/predict')
def predict(req: PredictRequest):
    claim = check(req)
    t = time.time()
    out = pipeline.verify(claim)
    if out is None:
        raise HTTPException(422, '검색할 단어가 없습니다')
    return {**out, 'latency_ms': round((time.time() - t) * 1000)}

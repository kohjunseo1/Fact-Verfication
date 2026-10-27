# HoVer E2E v2

성능 중심의 HoVer 2/3/4-Hop 파이프라인이다. TF-IDF Top-100 이후의 세 모델은
모두 `roberta-base`를 **HoVer Train으로 fine-tune**한다. Zero-shot 모델은 실험
대체물로 사용하지 않는다.

현재 상태: **코드/CPU 검증 완료, `NEEDS_COLAB_RUN`**. 로컬 macOS에서는 모델
학습이나 전체 Dev 추론을 실행하지 않았다.

## 실행 전 배치

Colab에서는 Google Drive의 기존 프로젝트를 마운트한다. 개별 노트북을 서로 다른
런타임에서 직접 시작하려면 이 폴더를 다음 경로에 둔다.

```text
/content/drive/MyDrive/hover-project/고준서/e2e_v2/
```

`00_preflight.ipynb`, 후속 단계 `02`~`05`, `run_all.ipynb`에는 현재 Python 코드
bundle이 포함되어 있어 Drive에 코드 폴더가 없어도 `/content/e2e_v2`로 복원한다.
후속 노트북은 런타임에 남은 이전 코드도 최신 bundle로 동기화한다. 입력은 기존 Drive의
`hover-project/data/`에서 찾고 Wikipedia DB·Train/Dev·TF-IDF를
`/content/hover_data/`로 한 번만 복사한다. 이후 학습 중에는 Drive 원본을 반복
조회하지 않는다. 모델·예측·체크포인트는 아래 Drive 경로에 저장한다. Mac 로컬 CLI는
별도로 `/Users/gojunseo/Fact-Verfication/고준서/data/`를 계속 직접 사용할 수 있다.

```text
/content/drive/MyDrive/hover-project/고준서/e2e_v2_runs/roberta_e2e_v2/
```

Drive 원본을 직접 반복 조회하지 않는다. Wikipedia DB 및 TF-IDF 입력은 Colab
로컬 SSD에서 읽고, append-only `predictions.jsonl`과 Hugging Face checkpoint만
Drive에 남긴다.

## 내일 Colab 실행 순서

Colab GPU 런타임을 선택한 뒤 다음 순서로 실행한다.

1. `notebooks/00_preflight.ipynb`
2. `notebooks/01_retrieval.ipynb`
3. `notebooks/02_multihop.ipynb`
4. `notebooks/03_evidence.ipynb`
5. `notebooks/04_verifier.ipynb`
6. `notebooks/05_e2e.ipynb`

한 번에 실행하려면 `notebooks/run_all.ipynb`를 사용한다. 이 노트북은 2-Hop
1,126건을 먼저 완주하고 지표를 출력한다. 마지막 셀은 기본
`RUN_3_4_HOP=False`라 자동으로 넘어가지 않는다. 2-Hop 결과를 확인한 뒤 `True`로
바꾸면 같은 모델과 체크포인트로 3-Hop 1,835건, 4-Hop 1,039건을 추가한다.

`02`~`05`와 `run_all`은 Python unbuffered 모드로 실행한다. 화면에는 데이터 로드,
코드·입력 재사용, pair 생성, tokenizer/model 로드, checkpoint 재개, 추론 진행률,
처리속도, 경과시간과 ETA가 즉시 출력된다. 이미 실행 중인 프로세스에는 이 변경이
소급 적용되지 않으며 다음 단계 또는 다음 재실행부터 적용된다.

개별 노트북 기본값은 `HOVER_HOPS=2`다. 2-Hop 확인 후 런타임에서 다음 값을
설정하고 01부터 다시 실행하면 기존 완료 UID를 건너뛴다.

```python
import os
os.environ["HOVER_HOPS"] = "2,3,4"
```

## 구조

```text
e2e_v2/
├── configs/base.json
├── notebooks/
│   ├── 00_preflight.ipynb
│   ├── 01_retrieval.ipynb
│   ├── 02_multihop.ipynb
│   ├── 03_evidence.ipynb
│   ├── 04_verifier.ipynb
│   ├── 05_e2e.ipynb
│   └── run_all.ipynb
├── src/hover_e2e/
│   ├── data.py                 # UID, hpqa_id 그룹 분할, inference redaction
│   ├── retrieval.py            # Top-100 hard-negative reranker
│   ├── wiki.py                 # 재개 가능한 제목 인덱스와 제목 매칭
│   ├── multihop.py             # 2/3/4-Hop beam expansion + context reranking
│   ├── sentences.py            # CoreNLP/원본 배열, UTF-16 offset, 정합성 감사
│   ├── evidence.py             # sentence selector pair/추론
│   ├── verifier.py             # retrieved/gold-mix verifier
│   ├── metrics.py              # 독립 지표 및 HoVer homepage rule
│   ├── modeling.py             # RoBERTa fine-tuning/validation/checkpoint
│   ├── stages.py               # 단계 orchestration
│   └── colab_runtime.py        # Drive→로컬 입력, 공유 캐시
└── tests/
```

## 모델과 누수 경계

### Retrieval

- 입력: 공식 TF-IDF Top-100.
- 학습: Train Gold 문서를 positive로 넣고 상위 TF-IDF 오답을 hard negative로 사용.
- 선택: 동일 `hpqa_id`가 겹치지 않는 Train 내부 Validation.
- 공식 Train/Dev 자체에서 겹치는 77개 `hpqa_id`의 Train 422건은 원본을
  변경하지 않고 전 모델 학습·Validation·Train 예측에서 격리한다.
- Dev: `InferenceClaim(uid, claim, num_hops)`만 전달된다. label과
  `supporting_facts`가 없는 타입이다.

### Multi-hop

- Mini-hop Top-3 seed와 Claim/본문에서 실제 Wikipedia 제목을 찾는다.
- `(claim + 현재 경로 문맥, 다음 후보 문서)` RoBERTa 점수로 beam을 확장한다.
- 2-Hop은 한 번, 3/4-Hop은 목표 깊이까지 같은 코드로 반복한다.
- 실제 채점·beam 선택에 사용한 문서는 모두 `candidate_documents`에 보존한다.
  문서당 100개, depth당 250개, claim당 600개 채점 상한과 배치 크기로 폭주를 막고
  `runtime_metrics.json`에 후보 수·처리량·process peak RSS를 남긴다.
- Train teacher forcing에서만 Gold positive를 넣는다. Dev candidate 함수에는 Gold,
  Oracle Evidence, label 인자가 없다.

### Evidence

- Top-5 문서의 기존 sentence array와 **원본 sentence ID**만 사용한다.
- 기존 CoreNLP cache와 검증된 Hotpot 원본 배열 override를 우선 재사용한다.
- 누락 문서만 CoreNLP `tokenize,ssplit`로 처리한다. Java UTF-16 offset을 Python
  code-point offset으로 변환하고 DB 원문 slice가 정확히 일치할 때만 저장한다.
  이 offset 검사는 DB 원문 무결성일 뿐 Gold 정합성 검증으로 간주하지 않는다.
  별도 원본 Gold 텍스트와 `(title, sentence_id, text)`가 일치한 fact만 검증 완료다.
- sentence ID 이동, 빈 문장 삽입, 검증 우회는 없다.
- 기존 86개 정합성 영향 Claim은 `alignment_audit.json`에 별도 보존하고 신뢰
  가능한 Evidence/Gold-Oracle 지표에서 분리한다. 전체 예측 자체는 유지한다.

### Verifier

- 기본 학습 입력은 retrieved evidence다.
- Gold 혼합은 기본 `gold_mix_ratio=0.0`으로 비활성화했다. 값을 올려도 원본 Gold
  텍스트 정합성이 확인된 Train UID만 혼합하고, 나머지는 retrieved evidence를 쓴다.
- 내부 Validation은 항상 retrieved-only이며 macro-F1 최고 checkpoint를 사용한다.
- evidence 제출 개수도 Train 내부 Validation의 HoVer Score로 선택한다.

## 평가

`metrics.json`은 전체와 Hop별로 다음을 저장한다.

- TF-IDF / RoBERTa / Multi-hop Candidate Recall 및 All-document Hit@K
- Evidence F1, exact match, 문서별 최소 한 **Gold fact** Coverage
- Label Accuracy와 Macro-F1
- HoVer Score
- Retrieved, Gold Document Oracle, Gold Evidence Oracle
- UID별 실패 원인

HoVer Score는 공식 홈페이지 설명을 독립 구현했다. k-hop claim에서 예측 근거에
처음 등장한 k+1개 문서만 보고, 각 문서의 처음 2문장만 사용하며, 모든 Gold
문서에서 정확한 Gold `(title, sentence_id)`를 최소 하나 찾고 label도 맞은 경우만
성공이다. 제목만 맞고 sentence ID가 틀리면 실패한다. 공식 repository의
document/sentence metric과도 별도 단위 테스트한다.

다른 팀의 58.1%는 **Dev 2-Hop 1,126건 HoVer Score**에서만 비교한다. 전체 또는
3/4-Hop 수치와 섞지 않는다.

## 결과 및 체크포인트 계약

각 단계는 다음 구조를 사용한다.

```text
<output_root>/<run_name>/
├── 00_preflight/{report.json,split_manifest.json,alignment_audit.json}
├── 01_retrieval/{model/,train/,dev/,metrics.json}
├── 02_multihop/{model/,train/,dev/,metrics.json}
├── 03_evidence/{model/,train/,dev/,metrics.json,alignment_audit.json,failures.jsonl}
├── 04_verifier/{model/,train/,dev/,failures.jsonl}
├── 05_e2e/{metrics.json,predictions.jsonl,failures.jsonl,oracle_document/,oracle_evidence/}
└── shared/{wiki_title_index.sqlite,02_corenlp_sentence_cache_v2.sqlite}
```

각 `train/`과 `dev/`에는 append-only `predictions.jsonl`과 작은
`checkpoint.json`이 있다. 설정 fingerprint가 다르면 기존 결과에 이어 쓰지 않고
즉시 중단한다. 전체 JSON을 매 checkpoint마다 다시 쓰지 않으므로 Drive 병목을
피한다. 모델도 `model_contract.json`에 설정·seed·Subtrain/Validation UID 해시를
기록하며, 계약이 다른 모델을 같은 run 이름으로 재사용하지 않는다. `run_all`은
Evidence 단계에서만 CoreNLP JVM을 띄워 Retrieval 학습 중 RAM을 점유하지 않는다.
이번 후보 예산·Verifier Gold-mix 설정은 기존 fingerprint와 다르다. 이전 run과
체크포인트는 삭제·덮어쓰기하지 않으며, 이미 GPU run이 있다면
`HOVER_RUN_NAME=roberta_e2e_v2_alignment_fix`처럼 새 이름을 지정해야 한다.

## 로컬에서 실제 완료한 검증

2026-10-27 macOS CPU에서 다음을 확인했다.

- 모든 Python 파일 `compileall` 통과
- pytest 21개 통과
- 7개 노트북 JSON, cell ID, 코드 문법, 미실행 output 상태 검사 통과
- NFC/NFD 제목 동일성 및 중복 제거
- UID 중복/누락 계약과 checkpoint config 혼합 방지
- 공식 Train/Dev 원본에서 겹친 `hpqa_id` 77개 그룹, Train 422건 확인 및 격리
- 격리 후 유효 Train 17,749건 → Subtrain 15,969 / 내부 Validation 1,780,
  Train/Validation 및 유효 Train/Dev `hpqa_id` overlap 0
- Dev inference 함수 Gold 인자/`supporting_facts` 참조 없음
- 공식형 document/evidence/label/HoVer metric 수작업 fixture 통과
- CoreNLP UTF-16 → Python offset 변환 fixture 통과
- 작은 SQLite Wikipedia에서 TF-IDF → 본문 제목으로 누락 2-Hop 복구 → Evidence
  → Verifier → HoVer Score CPU 통합 테스트 통과
- 실제 Dev 2-Hop 12건 제한형 CPU smoke: 제목 인덱스 재사용 시 2.10초
  (5.72 claims/s), 평균 후보 253.6개, 총 3,000 candidate-context 채점,
  process peak RSS 151.1MiB. 12건 모두 depth 상한이 작동했다. 이는 lexical test scorer의 실행 계약 점검이며
  RoBERTa 성능 수치가 아니다. 상세: `cpu_reports/multihop_smoke_12.json`.
  최초 5.0M-title 인덱스 생성은 로컬에서 약 4.8분, 291MiB였으며 이후 재사용한다.
- 실제 데이터 preflight: Train 18,171, Dev 4,000, Hop 1,126/1,835/1,039,
  Train/Dev TF-IDF UID 전부 일치

실제 Dev Gold를 **평가에만** 사용해 NFC 정규화 후 재계산한 TF-IDF 기준선은
다음과 같다. 이는 학습 결과가 아니라 평가 코드와 입력 계약을 확인한 CPU 결과다.

| 범위 | All-doc Hit@5 | All-doc Hit@100 (= Candidate Recall) |
|---|---:|---:|
| 2-Hop | 39.70% | 77.53% |
| 3-Hop | 8.50% | 36.08% |
| 4-Hop | 0.48% | 15.69% |
| 전체 | 15.20% | 42.45% |

실행 명령:

```bash
cd 고준서/e2e_v2
PYTHONPATH=src python -m pytest -q
```

## `NEEDS_COLAB_RUN`

다음은 로컬에서 검증했다고 주장하지 않는다.

- CUDA, fp16, 실제 `roberta-base` 다운로드 및 fine-tuning
- Hugging Face Trainer의 실제 GPU checkpoint resume
- 5.2M Wikipedia 전체 제목 인덱스 완주 시간
- 실제 CoreNLP Java 서버 및 전체 sentence cache 보강
- Drive의 `02_gold_sentence_alignment.jsonl` 원본 Gold 텍스트 커버리지와 기존
  정합성 audit 86개 UID 재확인. 참조가 없는 fact는 offset이 맞아도 미검증으로 남는다.
- Dev 2/3/4-Hop 전체 추론 성능과 Oracle 수치
- GPU RoBERTa에서 Multi-hop 후보 상한의 처리량/VRAM 및 recall trade-off

## 성능 개선 우선순위

1. 2-Hop Candidate Recall: seed 2/3/5, beam width, 제목 ambiguity cap을 내부
   Validation으로 조정한다.
2. Reranker: Top-100 hard negative 수, max length, epoch를 조정한다.
3. Multi-hop context: 마지막 문서만 확장하는 설정과 전체 path 문맥을 비교한다.
4. Evidence: 문서당 Top-2 제약 전 global score와 공식 필터 후 Coverage를 함께 본다.
5. Verifier: retrieved-only를 기본으로 두고, 정합성 확인 UID에 한해서만 Train Gold
   10/25/50% 혼합을 내부 Validation에서 비교한다.

Dev Gold를 사용해 위 설정을 선택하지 않는다.

## 참고 범위

- 기존 `고준서/00_eda.ipynb`, `01_retrieval.ipynb`, `02_evidence.ipynb`
- 읽기 전용 `한기선/notebooks/05`–`08` 및 baseline guide
- HoVer 공식 코드: <https://github.com/hover-nlp/hover>
- HoVer 공식 평가 설명: <https://hover-nlp.github.io/>

기존 파일 중 명백한 중복·중단 실험은 발견되지 않아 `_archive/`로 이동한 파일은
없다. 판단 기록은 `고준서/_archive/README.md`에 남겼다.

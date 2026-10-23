# Fact-Verfication 연구 계획서

### HoVer 기반 2-hop 사실 검증으로 AI 답변 신뢰도 개선

## 1. Problem Definition

### 1.1 문제점

가상의 지식 검색 서비스 운영사가 LLM 기반 "AI 답변" 기능을 출시했다. 답변은 자연스럽지만, 사용자 신고의 상당수가 "그럴듯하지만 틀린 사실"(할루시네이션)이고, 특히 "A 영화의 감독이 태어난 도시"처럼 두 개 이상의 사실을 연결해야 하는 답변에서 오류가 집중된다. 현재는 운영자가 신고된 답변을 위키 문서와 직접 대조해 검수하므로 비용이 크고 대응이 느리다.

요구사항은 "AI가 생성한 주장(claim)을 위키 근거와 대조해 자동으로 참/거짓을 판정하고, 판정 근거 문서를 함께 제시하라"이다.

### 1.2 문제의 중요도

- **신뢰도:** LLM은 성능이 높아져도 할루시네이션이 남아 있어, 생성 결과를 근거와 대조하는 검증 단계가 서비스 신뢰도의 마지막 방어선이다.
- **운영 비용:** 사람이 모든 답변을 위키와 대조하는 것은 불가능하다. 자동 판정 + 근거 제시로 검수자는 "근거를 확인하고 승인"만 하면 된다.
- **설명 가능성:** 판정과 함께 근거 문서를 보여주므로 사용자·운영자 모두 결과를 확인할 수 있다.
- **확장성:** 같은 구조를 뉴스 팩트체크, 사내 문서 기반 챗봇 검증, 상품 정보 검수 등에 재사용할 수 있다.

## 2. Background & Baseline

### 2.1 Related Works

- **FEVER (Thorne et al., 2018):** 위키 문장을 근거로 주장을 SUPPORTS / REFUTES / NOT ENOUGH INFO로 분류하는 대표 벤치마크. 대부분 1개 문서로 판정 가능한 single-hop 위주이며, "문서 검색 → 문장 선택 → NLI" pipeline이 표준 구조로 자리 잡았다.
- **HoVer (Jiang et al., 2020):** HotpotQA 질문-답 쌍을 주장으로 변환해 만든 2~4-hop 검증 데이터셋. 라벨은 SUPPORTED / NOT_SUPPORTED 2종이며, HotpotQA 팀이 가공한 위키 덤프를 검색 코퍼스로 쓰도록 권장한다. 공식 baseline은 TF-IDF 문서 검색 → BERT 문서 재순위 → BERT 문장 선택 → BERT 검증기 pipeline이다.
- **Multi-hop Dense Retrieval (MDR, Xiong et al., 2021):** 1-hop에서 찾은 문서를 질의에 이어 붙여 2-hop 문서를 다시 검색하는 반복 검색. bridge entity를 명시적으로 추출하지 않아도 다음 hop을 찾을 수 있다.
- **Baleen (Khattab et al., 2021):** hop마다 찾은 근거를 짧게 "압축(condense)"해 다음 질의에 반영하는 multi-hop 검색기. HoVer 리더보드에서 HoVer score 57.53으로, 공식 baseline(15.32) 대비 큰 차이를 보인다 (HoVer 공식 페이지).
- **LLM 기반 분해 검증 (예: ProgramFC, Pan et al., 2023):** LLM이 주장을 하위 질문들로 분해한 뒤 각각 검색·답변하고 결과를 종합한다. 학습 없이 적용 가능하지만 비용과 지연시간이 크다.

### 2.2 Baseline

- **Primary baseline:** 기본 2-hop 반복 검색(BM25) + NLI 검증기 — MDR 방식을 BM25로 단순화한 구조
  1. **Hop-1:** 주장으로 BM25 검색, 상위 k1개 문서 확보
  2. **Hop-2:** 각 1-hop 문서에 대해 `주장 + 1-hop 문서 제목 + 첫 문장`을 질의로 BM25 재검색, 상위 k2개 문서 확보
  3. **문서 선택:** (doc1, doc2) 후보 쌍을 두 단계 BM25 점수의 합으로 정렬해 상위 쌍 선택
  4. **검증:** 선택된 두 문서의 문장을 주장과 함께 사전학습 NLI 모델(예: RoBERTa / DeBERTa-v3 계열)에 입력, HoVer 2-hop train으로 fine-tune해 이진 판정

## 3. Dataset

### 3.1 Dataset

- **HoVer:** train / dev / test 제공, test는 주장만 공개(라벨 없음). 라이선스 CC BY-SA 4.0.
- **검색 코퍼스:** HoVer 팀이 배포하는 위키 DB `wiki_wo_links.db`(SQLite)를 사용

## 4. Experiment Design

### 4.1 Evaluation Metrics 및 검증

- **지표 설계**
  - **검증 정확도 — Label Accuracy (주 지표 1):** 예측 라벨이 정답과 일치한 비율. 클래스 불균형을 고려해 Macro-F1과 클래스별 Recall도 함께 보고하며, 특히 NOT_SUPPORTED Recall = 잘못된 답변을 놓치지 않는 비율로 비즈니스 핵심 지표로 해석한다.
  - **검색 성능 — Recall@k (주 지표 2):**
    - Doc Recall@k: 정답 근거 문서 중 상위 k개 안에 들어온 비율
    - Full Recall@k: 정답 근거 문서 2개가 모두 상위 k개 안에 들어온 주장의 비율 (2-hop 검증의 실질 성공 조건)
    - hop별 분리 보고: 주장에 이름이 나오는 문서 vs bridge 문서
- **지표 검증 방법**
  - **타당성:** Recall@k와 Accuracy의 상관을 확인 — 검색 Recall이 오른 만큼 Accuracy가 오르는지
  - **신뢰성:** 3개 seed 평균 ± 표준편차, baseline 대비 차이는 bootstrap으로 유의성 확인
  - **사람 평가:** dev에서 무작위 50~100건을 뽑아, 모델이 제시한 근거만 보고 사람이 판정에 동의하는지(근거 충분성) 측정

## 5. Plan (일자별)

| 일자       | 수행 내용                     | 산출물                 |
| ---------- | ----------------------------- | ---------------------- |
| 10/21      | 팀 구성 및 주제 후보 선택     |                        |
| 10/22      | 데이터 분석 및 후보 모델 탐색 | Baseline 노트북        |
| 10/23 - 25 | Baseline 공유                 | Baseline 1차 모델·성능 |
| 10/26      | 역할 분담, 고도화             | 실험 표, 학습 곡선     |
| 10/27      | 결과 정리, 발표자료 작성      | 시연 영상, 발표자료    |
| 10/28      | 발표 진행                     | 최종 발표              |

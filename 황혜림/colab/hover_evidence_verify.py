# %% [markdown]
# # HoVer: 근거 문장 선택 + 검증 (Colab GPU)
#
# `hover_retrieval.ipynb`가 만든 후보 문서(`hover_bundle/results/{train,dev}_….jsonl`)를 입력으로
# 저장소의 `retrieval/` 스크립트를 GPU에서 실행합니다.
#
# 1. 첫 단락 문장 저장소 준비 — Drive의 `hover_bundle/store_subset.db`를 쓰고,
#    없으면 HotpotQA 2017 덤프를 받아 Colab 디스크에 전체 저장소를 만듦
# 2. cross-encoder 학습 — 양성: 정답 근거 문장, 음성: 정답 문서의 다른 문장 + 검색된 오답 문서의 문장.
#    train 전체 모델(dev용) + `FOLDS`개 fold 모델(train 교차 예측용: 자기가 학습하지 않은 claim만 점수)
# 3. 근거 문장 선택 — train은 교차 예측 점수, dev는 전체 모델 점수. `RULE = "hybrid"`면 train 교차 예측
#    점수로 기준값을 고르고(dev는 쓰지 않음), 확률 기준값 이상 문장 + 빈 문서 구제 규칙을 적용
# 4. 검증 모델 학습 — HoVer 공식 레시피 (bert-base-uncased, 길이 200, claim ≤ 60토큰, lr 3e-5, 배치 4×2, 3,000스텝)
# 5. dev 예측 — `…pred.jsonl`
# 6. dev 단계별 평가
#
# 모델은 Drive `hover_bundle/models/<RETRIEVAL_TAG>/`, 결과는 `hover_bundle/results`에 저장되고,
# 다시 실행하면 이미 끝난 단계는 건너뜁니다. 런타임 → GPU(L4 또는 A100).

# %%
# ---- 설정 ----
BUNDLE = "/content/drive/MyDrive/hover_bundle"
RETRIEVAL_TAG = "B20_M5_d3_both_in3d0_floor0.7_snt"   # hover_retrieval.ipynb 결과 파일 이름의 설정 부분
# 이전 설정: "B20_M5_d3_both_in3d0_floor0.7"(NORP 포함·the 그대로), "B20_M5_d3_out_floor0.7", "B20_M5_d3_both_in3_floor0.7"
PER_DOC = 2                                 # 문서마다 남길 문장 수
N_SENT = 5                                  # 전체 근거 문장 상한 (검색 노트북의 N과 같은 커널을 쓸 수 있어 이름을 구분)
RULE = "hybrid"                             # "fixed" = 항상 상위 N_SENT개, "hybrid" = 확률 기준값 + 빈 문서 구제
FOLDS = 2                                   # train 근거를 교차 예측할 fold 수
MAX_COV_DROP = 1.0                          # 기준값 고를 때 허용하는 "정답 문서 모두 커버" 하락 (%p)
REPO = "https://github.com/poohgajoah/hover-wiki-graph.git"
FORCE = False                               # True면 Drive에 결과가 있어도 모든 단계를 다시 실행

# %%
from google.colab import drive
drive.mount("/content/drive")
!apt-get -qq install -y aria2 > /dev/null
!rm -rf /content/hover-wiki-graph && git clone -q --depth 1 {REPO} /content/hover-wiki-graph

# %%
import json
import os
import subprocess
import sys

REPO_DIR = "/content/hover-wiki-graph"
DATA = "/content/data"                      # HOVER_DATA: retrieval/ 스크립트가 쓰는 데이터 폴더
WORK = "/content/work"
os.makedirs(f"{DATA}/store", exist_ok=True)
os.makedirs(WORK, exist_ok=True)
for name in ("results", "models"):          # 결과·모델은 Drive에 바로 쓰이도록 연결
    os.makedirs(f"{BUNDLE}/{name}", exist_ok=True)
    if not os.path.islink(f"{DATA}/{name}"):
        os.symlink(f"{BUNDLE}/{name}", f"{DATA}/{name}")
for split in ("train", "dev"):
    dst = f"{DATA}/hover_{split}_release_v1.1.json"
    if not os.path.exists(dst):
        os.symlink(f"{BUNDLE}/hover_{split}_release_v1.1.json", dst)

RES = f"{DATA}/results"
RETRIEVAL = {s: f"{RES}/{s}_{RETRIEVAL_TAG}.jsonl" for s in ("train", "dev")}
SCORES = {"train": f"{RES}/train_{RETRIEVAL_TAG}.oof{FOLDS}.scores.jsonl",   # 교차 예측 점수
          "dev": f"{RES}/dev_{RETRIEVAL_TAG}.scores.jsonl"}
PARAMS = f"{RES}/{RETRIEVAL_TAG}.oof{FOLDS}.selection_params.json"
# 검색 설정마다 모델을 따로 둬서 설정끼리 그대로 비교할 수 있게 함
MODEL_ROOT = f"{DATA}/models/{RETRIEVAL_TAG}"
CE_DIR = f"{MODEL_ROOT}/sentence_selector"                       # train 전체로 학습 (dev 점수용)
CE_FOLD = [f"{MODEL_ROOT}/sentence_selector_fold{k}of{FOLDS}" for k in range(FOLDS)]
for s, p in RETRIEVAL.items():
    assert os.path.exists(p), f"{s} 검색 결과가 없습니다: {p}"
    print(s, sum(1 for _ in open(p, encoding="utf-8")), "claims")


def run(*args):
    """retrieval/ 스크립트를 GPU 환경에서 실행.
    Colab은 자식 프로세스의 출력을 셀에 보여 주지 않으므로 한 줄씩 받아서 직접 출력한다
    (오류가 나면 Traceback도 여기에 찍힘)."""
    env = {**os.environ, "HOVER_DATA": DATA, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, *args], cwd=f"{REPO_DIR}/retrieval", env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait():
        raise RuntimeError(f"{args[0]} failed (exit {proc.returncode}) — 위 출력의 Traceback 참고")


def done(path):
    return not FORCE and os.path.exists(path)

# %%
# ---- 1. 첫 단락 문장 저장소 ----
# 우선순위: ① Drive의 sents.jsonl.gz(전체 문서, export_sentences.py로 만든 것)로 전체 저장소 생성
#          ② store_subset.db(이전 검색 결과·정답 문서만) ③ Stanford 서버의 HotpotQA 2017 덤프
import shutil

ABS_URL = "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
SENTS = f"{BUNDLE}/sents.jsonl.gz"
SUBSET = f"{BUNDLE}/store_subset.db"
STORE_CACHE = f"{BUNDLE}/cache/wiki.db"     # 한 번 만든 전체 저장소(약 2.4GB)를 Drive에 보관
if not os.path.exists(f"{DATA}/store/wiki.db") and os.path.exists(STORE_CACHE):
    shutil.copyfile(STORE_CACHE, f"{DATA}/store/wiki.db")   # Colab 디스크로 복사해서 읽음 (Drive보다 빠름)
    print("using", STORE_CACHE)
if not os.path.exists(f"{DATA}/store/wiki.db") and os.path.exists(SENTS):
    run("build_store.py", "--nodes", f"{BUNDLE}/nodes.tsv.gz", "--links", f"{BUNDLE}/links.i32",
        "--sents", SENTS, "--out", f"{DATA}/store/wiki.db")
    os.makedirs(os.path.dirname(STORE_CACHE), exist_ok=True)
    shutil.copyfile(f"{DATA}/store/wiki.db", STORE_CACHE)
    print("cached", STORE_CACHE)
if not os.path.exists(f"{DATA}/store/wiki.db") and os.path.exists(SUBSET):
    shutil.copyfile(SUBSET, f"{DATA}/store/wiki.db")
    print("using", SUBSET)
if not os.path.exists(f"{DATA}/store/wiki.db"):
    if not os.path.exists(f"{WORK}/abstracts"):
        subprocess.run(["aria2c", "-q", "-x", "16", "-s", "16", "-d", WORK, "-o", "abstracts.tar.bz2", ABS_URL], check=True)
        os.makedirs(f"{WORK}/abstracts", exist_ok=True)
        subprocess.run(["tar", "-xjf", f"{WORK}/abstracts.tar.bz2", "-C", f"{WORK}/abstracts"], check=True)
        os.remove(f"{WORK}/abstracts.tar.bz2")
    run("build_store.py", "--nodes", f"{BUNDLE}/nodes.tsv.gz", "--links", f"{BUNDLE}/links.i32",
        "--abstracts", f"{WORK}/abstracts", "--out", f"{DATA}/store/wiki.db")

# 검색 결과에 나온 문서가 저장소에 다 있는지 확인 (빠지면 근거 문장이 조용히 비게 됨)
import sqlite3
con = sqlite3.connect(f"file:{DATA}/store/wiki.db?mode=ro", uri=True)
needed = set()
for p in RETRIEVAL.values():
    with open(p, encoding="utf-8") as f:
        for line in f:
            needed.update(json.loads(line)["opened"])
missing = [t for t in needed if con.execute("SELECT 1 FROM docs WHERE title = ?", (t,)).fetchone() is None]
con.close()
assert not missing, (f"{len(missing)}/{len(needed)} retrieved docs are not in the store "
                     f"(e.g. {missing[:3]}); upload sents.jsonl.gz to {BUNDLE}")
print(f"store ready: all {len(needed)} retrieved docs present")

# %%
# ---- 2. cross-encoder 학습 (전체 1개 + fold별 FOLDS개) ----
for out, extra in [(CE_DIR, [])] + [(CE_FOLD[k], ["--folds", str(FOLDS), "--exclude-fold", str(k)])
                                    for k in range(FOLDS)]:
    if done(f"{out}/config.json"):
        print("skip: cross-encoder already trained at", out)
    else:
        run("sentence_selector.py", "train", "--retrieval", RETRIEVAL["train"], "--out", out, *extra)

# %%
# ---- 3. 근거 문장 선택 ----
# 3a. 후보 문장 점수: train은 fold 모델로 교차 예측, dev는 전체 모델
if not done(SCORES["train"]):
    parts = []
    for k in range(FOLDS):
        part = f"{SCORES['train']}.fold{k}"
        run("sentence_selector.py", "score", "--retrieval", RETRIEVAL["train"], "--out", CE_FOLD[k],
            "--folds", str(FOLDS), "--only-fold", str(k), "--batch-size", "256", "--output", part)
        parts.append(part)
    with open(SCORES["train"], "w", encoding="utf-8") as out:
        for part in parts:
            with open(part, encoding="utf-8") as f:
                out.write(f.read())
            os.remove(part)
if not done(SCORES["dev"]):
    run("sentence_selector.py", "score", "--retrieval", RETRIEVAL["dev"], "--out", CE_DIR,
        "--batch-size", "256", "--output", SCORES["dev"])

# 3b. 선택 규칙: hybrid면 train 교차 예측 점수로 기준값 선택 (dev는 쓰지 않음)
rule_args = ["--rule", RULE, "--per-doc", str(PER_DOC), "--n", str(N_SENT)]
if RULE == "hybrid":
    if not done(PARAMS):
        run("tune_selection.py", "--scores", SCORES["train"], "--split", "train", "--per-doc", str(PER_DOC),
            "--n", str(N_SENT), "--max-cov-drop", str(MAX_COV_DROP), "--output", PARAMS)
    params = json.load(open(PARAMS, encoding="utf-8"))
    rule_args += ["--thr", str(params["thr"]), "--doc-thr", str(params["doc_thr"])]
    EVID_TAG = f"evidence_hyb{params['thr']:g}_{params['doc_thr']:g}_k{PER_DOC}_n{N_SENT}_oof{FOLDS}"
else:
    EVID_TAG = f"evidence_k{PER_DOC}_n{N_SENT}_oof{FOLDS}"
EVIDENCE = {s: f"{RES}/{s}_{RETRIEVAL_TAG}.{EVID_TAG}.jsonl" for s in ("train", "dev")}
PRED = f"{RES}/dev_{RETRIEVAL_TAG}.{EVID_TAG}.pred.jsonl"
VER_DIR = f"{MODEL_ROOT}/verifier_{EVID_TAG}"
print("evidence:", EVID_TAG)

# 3c. 규칙 적용
for split in ("train", "dev"):
    if done(EVIDENCE[split]):
        print("skip:", EVIDENCE[split])
        continue
    run("sentence_selector.py", "apply", "--scores", SCORES[split], *rule_args, "--output", EVIDENCE[split])

for split in ("train", "dev"):
    with open(EVIDENCE[split], encoding="utf-8") as f:
        sizes = [len(json.loads(l)["evidence"]) for l in f]
    print(f"{split}: mean evidence sentences {sum(sizes) / len(sizes):.2f}")
with open(EVIDENCE["dev"], encoding="utf-8") as f:
    ex = json.loads(f.readline())
print(ex["claim"])
for e in ex["evidence"]:
    print(f"  {e['score']:7.3f}  [{e['title']} #{e['sent_id']}] {e['text'].strip()[:120]}")

# %%
# ---- 4. 검증 모델 학습 ----
if done(f"{VER_DIR}/config.json"):
    print("skip: verifier already trained at", VER_DIR)
else:
    run("verifier.py", "train", "--evidence", EVIDENCE["train"], "--out", VER_DIR)

# %%
# ---- 5. dev 예측 ----
if done(PRED):
    print("skip:", PRED)
else:
    run("verifier.py", "predict", "--evidence", EVIDENCE["dev"], "--out", VER_DIR, "--output", PRED)

from collections import Counter
with open(PRED, encoding="utf-8") as f:
    preds = [json.loads(l) for l in f]
print(len(preds), "predictions:", dict(Counter(p["pred"] for p in preds)))
print("saved", PRED)

# %%
# ---- 6. dev 단계별 평가 (retrieval/evaluate.py) ----
run("evaluate.py", "--split", "dev", "--tag", RETRIEVAL_TAG, "--evidence-tag", EVID_TAG)

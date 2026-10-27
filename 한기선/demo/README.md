# 2-hop 사실 검증 데모 (k8s: kubeadm + containerd)

```
브라우저 → frontend (nginx: index.html, /api/* → backend)
             → backend (FastAPI: POST /predict, GET /health, GET /info)
                 → 위키 DB · FTS5 인덱스 · 모델 4개 (hostPath, 읽기 전용)
```

## 준비 (한 번)

`한기선/data/` 아래에 둔다 (gitignore됨).

| 경로 | 만드는 법 |
|---|---|
| `data/raw/wiki_wo_links.db` | https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db |
| `data/index/wiki_index.db` | `python3 demo/build_index.py data/raw/wiki_wo_links.db data/index/wiki_index.db` (약 1분) |
| `data/models/{doc_rerank,hop2,sent_select,verifier}` | 노트북 05·08·06·07의 저장 셀 → Drive `MyDrive/fact-verification/models` 에서 내려받기 |

파이프라인은 발표의 v3와 같다 (재정렬 후보 100개, hop 1 상위 2개, 문서 5개, 근거 5문장). 1차 검색만 TF-IDF 목록 대신 BM25다. 모델은 v3 학습본(10-27)을 받아야 수치가 맞는다.

로컬 클러스터 (macOS, Lima):

```sh
brew install lima
limactl start --name=k8s --cpus=4 --memory=8 --tty=false \
  --set '.mounts=[{"location":"~/Documents/Dev/fact-verification/한기선","mountPoint":"/mnt/fv","writable":false}]' \
  template:k8s
mkdir -p ~/.kube && cp ~/.lima/k8s/copied-from-guest/kubeconfig.yaml ~/.kube/config   # 기존 ~/.kube/config가 없을 때만
```

## 배포

```sh
# 노드의 containerd에 직접 빌드 (레지스트리 없이)
limactl shell k8s -- sudo nerdctl --namespace k8s.io build -t fv-backend:dev /mnt/fv/demo/backend
kubectl apply -k 한기선/demo
kubectl -n fv-demo port-forward svc/frontend 8080:80   # → http://localhost:8080
```

코드를 바꾼 뒤: 백엔드는 다시 빌드하고 `kubectl -n fv-demo rollout restart deploy/backend`, 프론트는 `kubectl apply -k` 만 하면 된다.

정리: `kubectl delete -k 한기선/demo`, VM 끄기 `limactl stop k8s`.

## Colab GPU로 띄우기 (발표용)

CPU에서는 claim 하나에 수십 초가 걸린다. [colab_serve.ipynb](colab_serve.ipynb)를 GPU 런타임에서 위부터 실행하면 백엔드와 화면이 Colab에서 같이 뜨고, cloudflared 임시 터널 주소(`https://….trycloudflare.com`)가 출력된다. 노트북이 GitHub에서 코드를 받으므로 데모 코드를 바꿨으면 먼저 push한다. 모델은 Drive `models/`, 위키 DB는 Drive `data/raw/`에서 읽는다.

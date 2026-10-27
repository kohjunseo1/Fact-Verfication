# 위키피디아 제목 그래프 (2017-10-01, HoVer/HotpotQA 스냅숏)

## abstract/ — 첫 단락 링크 그래프 (HoVer 검색 코퍼스)
출처: HotpotQA `enwiki-20171001-pages-meta-current-withlinks-abstracts` (문서 5,233,329개).
- `nodes.tsv`: `node_id  title  wiki_page_id  n_sentences`. 제목은 HotpotQA/HoVer 형식이라
  HTML 이스케이프가 그대로 있고(`Sugar &amp; Spice`), `supporting_facts`의 제목과 정확히 일치합니다.
- `edges.tsv`: `src_title  sent_idx  tgt_title  anchor_text`. 링크가 한 번 나올 때마다 한 줄입니다
  (23,014,264줄). `sent_idx`는 HoVer `supporting_facts`와 같은 문장 분리 기준입니다.
- 링크 대상은 정확 일치 → 첫 글자 대소문자 보정 → 2017-10-01 리다이렉트 순서로 찾습니다.
  찾지 못한 대상(1.6%, 대부분 `Croatia`처럼 HotpotQA 코퍼스에 없는 문서)은
  `unresolved_targets.tsv`에 있습니다.

## full/ — 본문 전체 링크 그래프
출처: `enwiki-20171001-pagelinks.sql.gz` (archive.org, md5 df7d24fe92c5838a2b39a70097157b2c).
- `nodes.tsv`: `page_id  title`. 리다이렉트가 아닌 ns0 문서 5,486,204개입니다.
  제목은 위키피디아 원래 형태(`Sugar & Spice`)라서, HoVer 제목과 맞추려면
  `scripts/title_utils.py`의 `html_unescape`를 쓰면 됩니다.
- `edges.tsv.gz`: `src_page_id  tgt_page_id`. 리다이렉트를 푼 뒤 정렬하고 중복을 없앴습니다
  (엣지 413,594,217개). 틀·내비게이션 박스에서 나온 링크도 들어 있습니다.

## pages_ns0_20171001.tsv
ns0 문서 13,477,652개(리다이렉트 7,991,447개 포함)의 `page_id  title  redirect_target`입니다.
`enwiki-20171001-pages-articles.xml.bz2`에서 뽑았습니다.

## 다시 만들기
```
scripts/fetch_ranges.sh <xml url> <tmp> 256 4 | bzip2 -dc | uv run python scripts/extract_pages.py data/graph/pages_ns0_20171001.tsv
find data/raw/abstracts -name 'wiki_*.bz2' | sort | while read f; do bzip2 -dc "$f"; done \
  | uv run python scripts/build_abstract_graph.py data/graph/pages_ns0_20171001.tsv data/graph/abstract
# pagelinks 덤프는 그래프를 만든 뒤 지웠으므로 먼저 다시 받아야 합니다:
# scripts/fetch_ranges.sh https://archive.org/download/enwiki-20171001/enwiki-20171001-pagelinks.sql.gz <tmp> > data/raw/enwiki-20171001-pagelinks.sql.gz
uv run python scripts/build_full_graph.py data/graph/pages_ns0_20171001.tsv data/raw/enwiki-20171001-pagelinks.sql.gz data/graph/full
uv run python scripts/validate_graph.py data data/graph/abstract data/graph/full
```
마지막 검증 결과는 `validation.txt`에 있습니다.

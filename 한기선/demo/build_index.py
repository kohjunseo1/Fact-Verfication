"""위키 DB로 데모용 검색 인덱스(wiki_index.db)를 만든다. 표준 라이브러리만 쓴다.

- fts: BM25 1차 검색용 FTS5 (본문은 저장하지 않고 documents의 rowid만 가리킨다)
- titles: hop 2 후보를 찾는 제목 사전 (괄호를 뗀 소문자 이름 → 실제 제목), 08 노트북의 base2titles와 같다

사용: python build_index.py <wiki_wo_links.db> <wiki_index.db>
"""
import re
import sqlite3
import sys
import time
import unicodedata

wiki_path, out_path = sys.argv[1], sys.argv[2]
out = sqlite3.connect(out_path)
out.execute(f"ATTACH DATABASE 'file:{wiki_path}?mode=ro' AS w")
out.executescript("""
PRAGMA journal_mode = OFF;
PRAGMA synchronous = OFF;
DROP TABLE IF EXISTS fts;
DROP TABLE IF EXISTS titles;
CREATE VIRTUAL TABLE fts USING fts5(title, text, content='', tokenize='unicode61 remove_diacritics 2');
CREATE TABLE titles (base TEXT, rid INTEGER);
""")

t0, n = time.time(), 0
cur = out.execute('SELECT rowid, id, text FROM w.documents')
while rows := cur.fetchmany(50_000):
    out.executemany('INSERT INTO fts(rowid, title, text) VALUES (?, ?, ?)', [(r, t, x) for r, t, x in rows])
    out.executemany('INSERT INTO titles VALUES (?, ?)',
                    [(re.sub(r'\s*\(.*\)$', '', unicodedata.normalize('NFC', t)).lower(), r) for r, t, _ in rows])
    n += len(rows)
    print(f'{n:,} 문서 {time.time() - t0:.0f}s', flush=True)

out.execute('CREATE INDEX titles_base ON titles(base)')
out.execute("INSERT INTO fts(fts) VALUES ('optimize')")
out.commit()
print(f'완료 {time.time() - t0:.0f}s')

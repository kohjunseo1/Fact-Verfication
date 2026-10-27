# %% [markdown]
# # HoVer: entity-seeded best-first search over the 2017 Wikipedia link graph
#
# Pipeline
# 1. claim → spaCy `en_core_web_trf` entities (date/number labels dropped) + claim n-grams matched against the title list
# 2. mention → title: exact match → lower-case/parenthesis-stripped match → title-embedding search (dropped below `SIM_FLOOR`)
# 3. resolved docs = start candidates → best-first search on the intro-paragraph link graph
#    - score(v | u) = cos(intro(v), claim + intro(u)); queue priority = mean score along the path
#    - max depth `MAX_DEPTH`, opened docs skipped, stop after `B` opened docs
#    - per opened doc: top-`M` outgoing-link neighbours, plus (`DIRECTION = "both"`, docs at depth
#      <= `IN_MAX_DEPTH`) top-`M_IN` incoming-link neighbours ranked separately
#
# **Before running**: upload the local folder `data/colab_bundle/` to Google Drive as `MyDrive/hover_bundle/`
# (`nodes.tsv.gz`, `links.i32`, `hover_*_release_v1.1.json`). Runtime → GPU (L4 or A100).
#
# Intro paragraphs come from `hover_bundle/sents.jsonl.gz` when present (made locally by
# `retrieval/export_sentences.py`), otherwise from the Stanford-hosted HotpotQA 2017 dump (1.55 GB).
# The run embeds 5.2M titles and intro paragraphs;
# with `CACHE` set (default) these are stored on Drive (≈ 10 GB) on the first run and loaded afterwards.

# %%
# ---- configuration ----
BUNDLE = "/content/drive/MyDrive/hover_bundle"
CACHE = f"{BUNDLE}/cache"   # Drive cache: intro texts ≈ 2 GB + embeddings 2 × 4 GB, reused next time (None = off)
SPLIT = "dev"
LIMIT = None                # e.g. 300 for a quick run
B = 20                      # documents opened per claim
M = 5                       # neighbours enqueued per opened document
MAX_DEPTH = 3
SIM_FLOOR = 0.70            # min cosine for the title-embedding fallback
DROP_NORP = True            # skip nationality/group entities (British, Chinese, ...): 0.3% of them were evidence docs
STRIP_THE = True            # retry title lookup without a leading "the" ("the Ohio River" -> "Ohio River")
DIRECTION = "both"          # "out" = links leaving a doc only, "both" = also links pointing to it
M_IN = 3                    # with "both": incoming-link neighbours enqueued per doc, chosen separately
                            # so hub in-links (tens of thousands) cannot crowd out the outgoing links
IN_MAX_DEPTH = 0            # follow incoming links only from docs at depth <= this (0 = start docs only).
                            # Following them at every depth spent the budget on 3/4-hop claims.
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TRF_WHEEL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"

# %%
from google.colab import drive
drive.mount("/content/drive")
# Colab already ships spaCy 3.8 and cupy; the [cuda12x] extra pins a conflicting cupy, so only add the model
!pip -q install {TRF_WHEEL}
!apt-get -qq install -y aria2 > /dev/null

# %%
import bz2, glob, gzip, heapq, html, itertools, json, os, pickle, re, subprocess, time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
from tqdm.auto import tqdm

WORK = "/content/work"
os.makedirs(WORK, exist_ok=True)
if CACHE:
    os.makedirs(CACHE, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32
print("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

# node ids are 0..N-1 in file order; titles are in HotpotQA/HoVer form ("Sugar &amp; Spice")
with gzip.open(f"{BUNDLE}/nodes.tsv.gz", "rt", encoding="utf-8", newline="\n") as f:
    titles = []
    for i, line in enumerate(f):
        nid, t = line.rstrip("\n").split("\t", 1)
        assert int(nid) == i
        titles.append(t)
N = len(titles)
id_of = {t: i for i, t in enumerate(titles)}
print(f"{N:,} titles")

# %%
# ---- intro paragraphs (HotpotQA 2017-10-01 abstracts) ----
ABS_URL = "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
BREAKS = re.compile("[\t\n\r" + "".join(map(chr, (0x2028, 0x2029, 0x85, 0x0B, 0x0C))) + "]")   # same sanitising as the graph build


def parse_shard(path):
    out = []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out.append((o["title"], "".join(o["text"]).strip()))
    return out


SENTS = f"{BUNDLE}/sents.jsonl.gz"   # export_sentences.py output; avoids the Stanford download


def load_texts():
    cached = f"{CACHE}/texts.pkl" if CACHE else None
    if cached and os.path.exists(cached):
        with open(cached, "rb") as f:
            return pickle.load(f)
    if os.path.exists(SENTS):
        texts = [""] * N
        with gzip.open(SENTS, "rt", encoding="utf-8", newline="\n") as f:
            for line in tqdm(f, total=N, desc="sents.jsonl.gz"):
                i, _title, sents = json.loads(line)
                texts[i] = "".join(sents).strip()
        return texts
    tar = f"{WORK}/abstracts.tar.bz2"
    if not os.path.exists(f"{WORK}/abstracts"):
        subprocess.run(["aria2c", "-q", "-x", "16", "-s", "16", "-d", WORK, "-o", "abstracts.tar.bz2", ABS_URL], check=True)
        os.makedirs(f"{WORK}/abstracts", exist_ok=True)
        subprocess.run(["tar", "-xjf", tar, "-C", f"{WORK}/abstracts"], check=True)
        os.remove(tar)
    shards = sorted(glob.glob(f"{WORK}/abstracts/**/wiki_*.bz2", recursive=True))
    texts, missing = [""] * N, 0
    with Pool(os.cpu_count()) as pool:
        for rows in tqdm(pool.imap_unordered(parse_shard, shards, chunksize=16), total=len(shards), desc="abstracts"):
            for t, text in rows:
                i = id_of.get(t, id_of.get(BREAKS.sub(" ", t)))
                if i is None:
                    missing += 1
                else:
                    texts[i] = text
    print("docs without node id:", missing)
    if cached:
        with open(cached, "wb") as f:
            pickle.dump(texts, f, protocol=5)
    return texts


texts = load_texts()
print("empty intros:", sum(1 for t in texts if not t))

# %%
# ---- link graph as CSR (out and in) ----
links = np.fromfile(f"{BUNDLE}/links.i32", dtype="<i4").reshape(-1, 2)


def csr(src, tgt):
    order = np.argsort(src, kind="stable")
    ptr = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=ptr[1:])
    return ptr, tgt[order].astype(np.int64)


OUT_PTR, OUT_IDX = csr(links[:, 0], links[:, 1])
IN_PTR, IN_IDX = csr(links[:, 1], links[:, 0])
del links
print(f"{OUT_IDX.size:,} links, mean out-degree {OUT_IDX.size / N:.2f}")


def neighbour_groups(u, depth):
    """(candidate ids, how many to enqueue) per link direction for a doc opened at `depth`."""
    groups = [(OUT_IDX[OUT_PTR[u]:OUT_PTR[u + 1]], M)]
    if DIRECTION == "both" and depth <= IN_MAX_DEPTH:
        groups.append((IN_IDX[IN_PTR[u]:IN_PTR[u + 1]], M_IN))
    return groups

# %%
# ---- embeddings: titles and intro paragraphs (GPU, cached on Drive) ----
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(EMB_MODEL, device=DEV)
if DEV == "cuda":
    model.half()
DIM = model.get_sentence_embedding_dimension()


def encode(strings, bs=256):
    v = model.encode(strings, batch_size=bs, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    return v.to(DEV, DTYPE)


MODEL_SLUG = EMB_MODEL.split("/")[-1]


def embed_all(name, strings, bs):
    """Embeddings of all strings; cached on Drive per model, with the compute time in a .seconds.json."""
    path = f"{CACHE}/{name}.{MODEL_SLUG}.f16.npy" if CACHE else None
    if path and os.path.exists(path):
        arr = np.load(path)
        print(f"{name}: loaded from {path}")
    else:
        t0 = time.time()
        arr = np.empty((len(strings), DIM), np.float16)
        step = 262_144
        for s in tqdm(range(0, len(strings), step), desc=name):
            arr[s:s + step] = encode(strings[s:s + step], bs).float().cpu().numpy().astype(np.float16)
        if path:
            np.save(path, arr)
            with open(path.replace(".npy", ".seconds.json"), "w") as f:
                json.dump({"seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"}, f)
    return torch.from_numpy(arr).to(DEV, DTYPE)


t0 = time.time()
TE = embed_all("title_emb", [html.unescape(t) for t in titles], 2048)
PE = embed_all("para_emb", texts, 512)
print(f"embeddings ready in {time.time() - t0:.0f}s", TE.shape, PE.shape)

# %%
# ---- title dictionaries ----
PAREN = re.compile(r"\s*\([^)]*\)")


def norm_title(s):
    return " ".join(PAREN.sub("", html.unescape(s)).split()).lower()


exact_ids = {}                 # unescaped title -> id
norm_ids = defaultdict(list)   # loose key -> ids
for i, t in enumerate(titles):
    u = html.unescape(t)
    exact_ids.setdefault(u, i)
    norm_ids[norm_title(u)].append(i)
print(f"{len(exact_ids):,} exact keys, {len(norm_ids):,} loose keys")

# %%
# ---- claims, NER and n-gram mentions ----
import spacy

print("spaCy", spacy.__version__, "on GPU" if spacy.prefer_gpu() else "on CPU (cupy not available)")
nlp = spacy.load("en_core_web_trf", disable=["parser", "lemmatizer"])
DROP = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"} | ({"NORP"} if DROP_NORP else set())
MAX_N = 8

claims = json.load(open(f"{BUNDLE}/hover_{SPLIT}_release_v1.1.json", encoding="utf-8"))
if LIMIT:
    claims = claims[:LIMIT]


def lookup_variants(text):
    """The mention itself, then (STRIP_THE) the mention without a leading "the"."""
    out = [text]
    if STRIP_THE and text[:4].lower() == "the " and len(text) > 4:
        out.append(text[4:])
    return out


def in_dictionary(text):
    return any(v in exact_ids or norm_title(v) in norm_ids for v in lookup_variants(text))


def ngram_mentions(doc, covered):
    """Longest-first, non-overlapping claim n-grams that are titles, outside NER spans."""
    used, found = set(covered), []
    for n in range(MAX_N, 0, -1):
        for i in range(len(doc) - n + 1):
            if any(j in used for j in range(i, i + n)):
                continue
            span = doc[i:i + n]
            words = [t for t in span if not t.is_punct]
            if not words or span[0].is_punct or span[-1].is_punct or all(t.is_stop for t in words):
                continue
            if not any(t.is_title or t.is_upper for t in words):     # needs a capitalised token
                continue
            if n == 1 and (span[0].pos_ != "PROPN" or i == 0):        # lone token: proper noun, not sentence-initial
                continue
            if in_dictionary(span.text):
                found.append(span.text)
                used.update(range(i, i + n))
    return found


mentions = []
for c, doc in zip(claims, tqdm(nlp.pipe((c["claim"] for c in claims), batch_size=64), total=len(claims), desc="ner")):
    ents = [e for e in doc.ents if e.label_ not in DROP]
    covered = {j for e in ents for j in range(e.start, e.end)}
    mentions.append({"ner": [e.text for e in ents], "ngram": ngram_mentions(doc, covered)})
print("mean NER mentions", np.mean([len(m["ner"]) for m in mentions]),
      "mean extra n-gram mentions", np.mean([len(m["ngram"]) for m in mentions]))

# %%
# ---- resolve mentions to start documents ----
claim_vecs = encode([c["claim"] for c in claims], 256)


def resolve_dictionary(text, cvec):
    variants = lookup_variants(text)
    for v in variants:                  # exact match first, for every variant
        if v in exact_ids:
            return exact_ids[v], "exact" if v == text else "exact-the"
    for v in variants:
        cands = norm_ids.get(norm_title(v))
        if cands:   # several titles share the loose key: keep the one whose intro is closest to the claim
            best = cands[int(torch.argmax(PE[cands] @ cvec))] if len(cands) > 1 else cands[0]
            return best, "loose" if v == text else "loose-the"
    return None, None


seeds, fallback = [], []      # fallback: (claim index, mention)
for ci, (c, m) in enumerate(zip(claims, mentions)):
    s = {}
    for text in m["ner"] + m["ngram"]:
        d, how = resolve_dictionary(text, claim_vecs[ci])
        if d is not None:
            s.setdefault(d, (text, how))
        else:
            fallback.append((ci, text))
    seeds.append(s)

# title-embedding search for mentions with no dictionary match (batched brute force on GPU)
qv = encode([t for _, t in fallback], 256) if fallback else None
kept = 0
for s in range(0, len(fallback), 64):
    sims = TE @ qv[s:s + 64].T                    # N x 64
    val, idx = sims.max(dim=0)
    for (ci, text), v, i in zip(fallback[s:s + 64], val.tolist(), idx.tolist()):
        if v >= SIM_FLOOR:
            seeds[ci].setdefault(i, (text, f"emb:{v:.2f}"))
            kept += 1
print(f"fallback mentions {len(fallback)}, kept {kept} (cos ≥ {SIM_FLOOR})")
print("mean start docs per claim", np.mean([len(s) for s in seeds]))

# %%
# ---- best-first search ----
def best_first(ci):
    claim, cvec = claims[ci]["claim"], claim_vecs[ci]
    start = list(seeds[ci])
    heap, tie, opened, seen = [], itertools.count(), [], set()
    if start:
        for d, sc in zip(start, (PE[start] @ cvec).float().tolist()):
            heapq.heappush(heap, (-sc, next(tie), d, 0, sc, 1))   # (-priority, tie, doc, depth, score sum, path len)
    while heap and len(opened) < B:
        _, _, u, depth, ssum, plen = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        opened.append(u)
        if depth >= MAX_DEPTH:
            continue
        q, pushed = None, set()
        for nb, m in neighbour_groups(u, depth):   # outgoing and incoming links are ranked separately
            if nb.size:
                nb = nb[~np.isin(nb, list(seen))]
            if not nb.size or m <= 0:
                continue
            if q is None:
                q = encode([claim + " " + texts[u]])[0]
            sc = PE[torch.from_numpy(nb).to(DEV)] @ q
            val, idx = torch.topk(sc.float(), min(m, nb.size))
            for v, j in zip(val.tolist(), idx.tolist()):
                d = int(nb[j])
                if d in pushed:
                    continue
                pushed.add(d)
                heapq.heappush(heap, (-(ssum + v) / (plen + 1), next(tie), d, depth + 1, ssum + v, plen + 1))
    return opened


t0 = time.time()
results = [best_first(ci) for ci in tqdm(range(len(claims)), desc="search")]
print(f"search: {time.time() - t0:.0f}s, mean opened {np.mean([len(r) for r in results]):.1f}")

# %%
# ---- evaluation ----
def gold(c):
    return {id_of[t] for t, _ in c["supporting_facts"]}


def dense_topk(ci, k):        # reference: claim-only dense retrieval over all intros
    return torch.topk((PE @ claim_vecs[ci]).float(), k).indices.tolist()


dense = [dense_topk(ci, B) for ci in range(len(claims))]
KS = [k for k in (2, 5, 10, 20, 50) if k <= B] + ([B] if B not in (2, 5, 10, 20, 50) else [])
rows = []
for hop in (2, 3, 4, "all"):
    idx = [i for i, c in enumerate(claims) if hop == "all" or c["num_hops"] == hop]
    if not idx:
        continue
    g = [gold(claims[i]) for i in idx]
    row = {"hops": hop, "n": len(idx),
           "seed recall": np.mean([len(gi & set(seeds[i])) / len(gi) for gi, i in zip(g, idx)]),
           "seed all-found": np.mean([gi <= set(seeds[i]) for gi, i in zip(g, idx)])}
    for k in KS:
        row[f"R@{k}"] = np.mean([len(gi & set(results[i][:k])) / len(gi) for gi, i in zip(g, idx)])
        row[f"all@{k}"] = np.mean([gi <= set(results[i][:k]) for gi, i in zip(g, idx)])
    row[f"dense R@{B}"] = np.mean([len(gi & set(dense[i])) / len(gi) for gi, i in zip(g, idx)])
    row[f"dense all@{B}"] = np.mean([gi <= set(dense[i]) for gi, i in zip(g, idx)])
    rows.append(row)

import pandas as pd
report = pd.DataFrame(rows).set_index("hops")
display(report.style.format({c: "{:.3f}" for c in report.columns if c != "n"}))

# %%
# ---- save per-claim output ----
os.makedirs(f"{BUNDLE}/results", exist_ok=True)
in_part = ""
if DIRECTION == "both":
    in_part = f"_in{M_IN}" + (f"d{IN_MAX_DEPTH}" if IN_MAX_DEPTH < MAX_DEPTH - 1 else "")
seed_part = ("_s" + ("n" if DROP_NORP else "") + ("t" if STRIP_THE else "")) if (DROP_NORP or STRIP_THE) else ""
tag = f"{SPLIT}_B{B}_M{M}_d{MAX_DEPTH}_{DIRECTION}{in_part}_floor{SIM_FLOOR}{seed_part}"
with open(f"{BUNDLE}/results/{tag}.jsonl", "w", encoding="utf-8") as f:
    for c, m, s, r in zip(claims, mentions, seeds, results):
        f.write(json.dumps({
            "uid": c["uid"], "claim": c["claim"], "num_hops": c["num_hops"], "label": c["label"],
            "gold": sorted({t for t, _ in c["supporting_facts"]}),
            "mentions": m,
            "seeds": [{"title": titles[d], "mention": t, "how": how} for d, (t, how) in s.items()],
            "opened": [titles[d] for d in r],
        }, ensure_ascii=False) + "\n")
report.to_csv(f"{BUNDLE}/results/{tag}.metrics.csv")
print("saved", tag)

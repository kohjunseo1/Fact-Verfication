"""Shared paths, I/O helpers and doc-store access for the local evidence/verification stages."""
import hashlib
import html
import json
import os
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("HOVER_DATA", ROOT / "data"))   # HOVER_DATA relocates everything (Colab)
STORE = DATA / "store"
DB_PATH = STORE / "wiki.db"
RESULTS = DATA / "results"      # Colab retrieval outputs (*.jsonl) go here
MODELS = DATA / "models"
LABELS = ["NOT_SUPPORTED", "SUPPORTED"]


def fold_of(uid: str, folds: int) -> int:
    """Deterministic fold of a claim (for cross-fitting the sentence selector)."""
    return int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16) % folds


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_torch():
    """Use all cores on CPU; returns the device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    torch.set_num_threads(os.cpu_count() or 4)
    return torch.device("cpu")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_hover(split: str):
    return json.load(open(DATA / f"hover_{split}_release_v1.1.json", encoding="utf-8"))


def sentence_input(title: str, sentence: str) -> str:
    """Text shown to the cross-encoder for one candidate sentence (title resolves pronouns)."""
    return f"{html.unescape(title)} : {sentence.strip()}"


class DocStore:
    """Read-only access to intro sentences (HoVer sentence ids) and the intro-link graph."""

    def __init__(self, path: Path = DB_PATH):
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.con.execute("PRAGMA mmap_size = 4000000000")
        self._cache = {}

    def sents(self, title: str):
        """Sentence list of a doc by HotpotQA/HoVer title ([] if unknown)."""
        if title not in self._cache:
            row = self.con.execute("SELECT sents FROM docs WHERE title = ?", (title,)).fetchone()
            if len(self._cache) > 200_000:
                self._cache.clear()
            self._cache[title] = json.loads(row[0]) if row else []
        return self._cache[title]

    def neighbours(self, title: str, limit: int = 50):
        """Titles linked from a doc's intro paragraph."""
        q = """SELECT d2.title FROM docs d1 JOIN links l ON l.src = d1.id JOIN docs d2 ON d2.id = l.tgt
               WHERE d1.title = ? LIMIT ?"""
        return [r[0] for r in self.con.execute(q, (title, limit))]

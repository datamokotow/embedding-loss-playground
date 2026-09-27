"""Offline smoke test: runs python/train.py with every loss on a tiny local model.

Builds a small static-embedding model from the datasets' own vocabulary, so it
needs no downloads. It checks that each loss trains end to end, not that it
reaches a particular score.

    python python/smoke_test.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer, models, pre_tokenizers

try:
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
except ImportError:
    from sentence_transformers.models import StaticEmbedding

HERE = Path(__file__).parent
LOSSES = ["mnrl", "mnrl_hn", "triplet", "cosent", "cosine", "batchall", "batchhard"]


def tiny_model(path: Path) -> None:
    text = (HERE / "datasets.json").read_text().lower()
    vocab = ["[UNK]", *sorted(set(re.findall(r"[a-z0-9]+", text)))]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    weights = np.random.default_rng(0).normal(size=(len(vocab), 32)).astype("float32")
    SentenceTransformer(modules=[StaticEmbedding(tok, embedding_weights=weights)]).save(str(path))


def main() -> int:
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "tiny"
        tiny_model(model_dir)
        for loss in LOSSES:
            dataset = {"cosent": "similarity", "cosine": "similarity", "batchall": "clustering", "batchhard": "clustering"}.get(loss, "retrieval")
            cmd = [sys.executable, str(HERE / "train.py"), "--loss", loss, "--dataset", dataset, "--model", str(model_dir),
                   "--epochs", "2", "--batch-size", "8", "--lr", "0.05", "--output-dir", str(Path(tmp) / loss)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            ok = res.returncode == 0 and "top-1 accuracy" in res.stdout
            summary = next((l for l in res.stdout.splitlines() if l.startswith("top-1")), res.stderr.strip().splitlines()[-1:] or "")
            print(f"{'PASS' if ok else 'FAIL'}  {loss:<10} {dataset:<11} {summary}")
            if not ok:
                failed.append(loss)
    print("\nAll losses trained." if not failed else f"\nFailed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

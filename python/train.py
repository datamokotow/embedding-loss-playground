"""Fine-tune a real embedding model with the same datasets and losses as the playground.

Examples
--------
    pip install -r python/requirements.txt
    python python/train.py --loss mnrl
    python python/train.py --loss cosent --dataset similarity
    python python/train.py --loss batchhard --dataset clustering --epochs 5

The script prints held-out top-1 accuracy and the right-vs-wrong cosine gap
before and after training, the same numbers the playground shows.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments

try:  # sentence-transformers >= 5.4
    from sentence_transformers.base.sampler import BatchSamplers
    from sentence_transformers.sentence_transformer import losses
except ImportError:  # older releases
    from sentence_transformers import losses
    from sentence_transformers.training_args import BatchSamplers

DATASETS = json.loads((Path(__file__).parent / "datasets.json").read_text())

# loss id -> (data format, batch sampler)
LOSS_FORMAT = {
    "mnrl": ("pairs", BatchSamplers.NO_DUPLICATES),
    "mnrl_hn": ("triplets", BatchSamplers.NO_DUPLICATES),
    "triplet": ("triplets", BatchSamplers.NO_DUPLICATES),
    "cosent": ("scored", BatchSamplers.BATCH_SAMPLER),
    "cosine": ("scored", BatchSamplers.BATCH_SAMPLER),
    "batchall": ("labels", BatchSamplers.GROUP_BY_LABEL),
    "batchhard": ("labels", BatchSamplers.GROUP_BY_LABEL),
}


def make_loss(name: str, model: SentenceTransformer, margin: float):
    cos_dist = losses.BatchHardTripletLossDistanceFunction.cosine_distance
    return {
        "mnrl": lambda: losses.MultipleNegativesRankingLoss(model),
        "mnrl_hn": lambda: losses.MultipleNegativesRankingLoss(model),
        "triplet": lambda: losses.TripletLoss(
            model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=margin
        ),
        "cosent": lambda: losses.CoSENTLoss(model),
        "cosine": lambda: losses.CosineSimilarityLoss(model),
        "batchall": lambda: losses.BatchAllTripletLoss(model, distance_metric=cos_dist, margin=margin),
        "batchhard": lambda: losses.BatchHardTripletLoss(model, distance_metric=cos_dist, margin=margin),
    }[name]()


def build_dataset(world: dict, fmt: str, seed: int = 12345) -> Dataset:
    docs, train = world["documents"], world["train"]
    related = {frozenset(p) for p in world["related"]}

    def gold(a: str, b: str) -> float:
        return 1.0 if a == b else 0.4 if frozenset((a, b)) in related else 0.0

    if fmt == "pairs":
        return Dataset.from_dict({"anchor": [r["text"] for r in train], "positive": [docs[r["doc"]] for r in train]})
    if fmt == "triplets":
        return Dataset.from_dict(
            {
                "anchor": [r["text"] for r in train],
                "positive": [docs[r["doc"]] for r in train],
                "negative": [docs[world["hard_negative"][r["doc"]]] for r in train],
            }
        )
    if fmt == "scored":
        # every training text against every document, scored 1.0 / 0.4 / 0.0
        rows = [(r["text"], text, gold(r["doc"], d)) for r in train for d, text in docs.items()]
        random.Random(seed).shuffle(rows)
        a, b, s = zip(*rows)
        return Dataset.from_dict({"sentence1": list(a), "sentence2": list(b), "score": list(s)})
    # labels: every training text and every document, labelled by document id
    ids = {d: i for i, d in enumerate(docs)}
    texts = [r["text"] for r in train] + list(docs.values())
    labels = [ids[r["doc"]] for r in train] + list(ids.values())
    return Dataset.from_dict({"text": texts, "label": labels})


def evaluate(model: SentenceTransformer, world: dict) -> dict:
    doc_ids = list(world["documents"])
    d = model.encode([world["documents"][i] for i in doc_ids], normalize_embeddings=True)
    q = model.encode([r["text"] for r in world["test"]], normalize_embeddings=True)
    sims = q @ d.T
    hits, gap = 0, 0.0
    for row, r in zip(sims, world["test"]):
        right = doc_ids.index(r["doc"])
        hits += int(row.argmax() == right)
        gap += row[right] - (row.sum() - row[right]) / (len(row) - 1)
    n = len(world["test"])
    return {"top1": hits / n, "gap": float(gap / n)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loss", choices=LOSS_FORMAT, default="mnrl")
    p.add_argument("--dataset", choices=DATASETS, default="retrieval")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--margin", type=float, default=0.3, help="for triplet-style losses (cosine distance)")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    world = DATASETS[args.dataset]
    fmt, sampler = LOSS_FORMAT[args.loss]
    model = SentenceTransformer(args.model)
    train_dataset = build_dataset(world, fmt)
    print(f"{args.dataset}: {len(train_dataset)} {fmt} rows, columns {train_dataset.column_names}")

    before = evaluate(model, world)
    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir or f"out/{args.dataset}-{args.loss}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        batch_sampler=sampler,
        save_strategy="no",
        report_to="none",
        logging_steps=5,
    )
    SentenceTransformerTrainer(
        model=model, args=trainer_args, train_dataset=train_dataset, loss=make_loss(args.loss, model, args.margin)
    ).train()
    after = evaluate(model, world)

    print(f"\n{'':<22}{'base':>8}{'fine-tuned':>12}")
    print(f"{'top-1 accuracy':<22}{before['top1']:>8.0%}{after['top1']:>12.0%}")
    print(f"{'right vs wrong gap':<22}{before['gap']:>8.3f}{after['gap']:>12.3f}")


if __name__ == "__main__":
    main()

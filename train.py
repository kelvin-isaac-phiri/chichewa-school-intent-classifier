#!/usr/bin/env python3
"""
Split data.json and fine-tune Afro-XLMR for Chichewa intent classification.

  python train.py
  python train.py --model-name Davlan/afro-xlmr-mini --epochs 4 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split(entries: list[dict], val_size: float, test_size: float, seed: int):
    labels = [e["label"] for e in entries]
    train, rest, _, rest_y = train_test_split(
        entries, labels, test_size=val_size + test_size, random_state=seed, stratify=labels
    )
    val, test = train_test_split(
        rest, test_size=test_size / (val_size + test_size), random_state=seed, stratify=rest_y
    )
    return train, val, test


def add_ids(entries: list[dict], label2id: dict[str, int]) -> list[dict]:
    return [{"text": e["text"], "label": e["label"], "label_id": label2id[e["label"]]} for e in entries]


def to_ds(entries: list[dict]) -> Dataset:
    return Dataset.from_dict({"text": [e["text"] for e in entries], "label": [e["label_id"] for e in entries]})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "f1_macro": f1_score(labels, pred, average="macro"),
        "f1_weighted": f1_score(labels, pred, average="weighted"),
        "precision_macro": precision_score(labels, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, pred, average="macro", zero_division=0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune Afro-XLMR on data.json")
    p.add_argument("--input", default="data.json")
    p.add_argument("--splits-dir", default="data")
    p.add_argument("--model-name", default="Davlan/afro-xlmr-base")
    p.add_argument("--output-dir", default="model_output")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=2)
    args = p.parse_args()

    set_seed(args.seed)
    entries = load_json(Path(args.input))
    labels = sorted({e["label"] for e in entries})
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for name, i in label2id.items()}

    train, val, test = split(entries, args.val_size, args.test_size, args.seed)
    out = Path(args.splits_dir)
    save_json(out / "label_map.json", {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}})
    save_json(out / "train.json", add_ids(train, label2id))
    save_json(out / "val.json", add_ids(val, label2id))
    save_json(out / "test.json", add_ids(test, label2id))
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  labels={labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = to_ds(add_ids(train, label2id)).map(tok, batched=True, remove_columns=["text"])
    val_ds = to_ds(add_ids(val, label2id)).map(tok, batched=True, remove_columns=["text"])

    use_cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=use_cuda,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print(f"Training {args.model_name}  cuda={use_cuda}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    print("Validation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    save_json(Path(args.output_dir) / "training_config.json", vars(args))
    print(f"Saved model to {args.output_dir}/")


if __name__ == "__main__":
    main()

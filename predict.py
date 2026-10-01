#!/usr/bin/env python3
"""
Use a saved model.

  python predict.py --text "Kodi ndalama za sukulu ndi zingati?"
  python predict.py --test-file data/test.json
  python predict.py --interactive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clf(model_dir: Path, top_k=None):
    return pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        truncation=True,
        top_k=top_k,
    )


def rows_from(output) -> list[dict]:
    if isinstance(output, dict):
        return [output]
    if output and isinstance(output[0], dict):
        return output
    return output[0]


def show(text: str, rows) -> None:
    print(f"Text: {text}")
    for row in rows_from(rows):
        print(f"  {row['label']}: {row['score']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Predict or evaluate the fine-tuned model.")
    p.add_argument("--model-dir", default="model_output")
    p.add_argument("--text", default=None)
    p.add_argument("--test-file", default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"No model at {model_dir}. Train first with: python train.py")

    if args.text:
        show(args.text, clf(model_dir, top_k=args.top_k)(args.text))
        return

    if args.interactive:
        nlp = clf(model_dir, top_k=args.top_k)
        print("Type Chichewa text. quit to exit.\n")
        while True:
            text = input("> ").strip()
            if text.lower() in {"quit", "exit", "q"}:
                break
            if text:
                show(text, nlp(text))
                print()
        return

    test_file = Path(args.test_file or "data/test.json")
    rows = load(test_file)
    texts = [r["text"] for r in rows]
    y_true = [r["label"] for r in rows]
    y_pred = [r["label"] for r in clf(model_dir)(texts, batch_size=32)]

    print(f"{test_file}: {len(texts)} examples")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    names = sorted(set(y_true) | set(y_pred))
    print("Confusion matrix (rows=true, cols=pred)")
    print("labels:", names)
    print(confusion_matrix(y_true, y_pred, labels=names))


if __name__ == "__main__":
    main()

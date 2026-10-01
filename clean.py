#!/usr/bin/env python3
"""
Clean a {"text","label"} JSON dataset.

  python clean.py
  python clean.py --input data.json --output data.json --merge extra.json --near-dup
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PUNCT = re.compile(r"[?!.,;:\"'`]+")
SPACE = re.compile(r"\s+")


def load_entries(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path} must be a JSON list of objects")
    return data


def key(text: str) -> str:
    text = text.casefold().strip()
    text = PUNCT.sub("", text)
    return SPACE.sub(" ", text)


def clean(
    entries: list[dict],
    min_length: int,
    near_dup: bool,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    stats = Counter()
    kept: list[dict[str, str]] = []
    seen_exact: set[tuple[str, str]] = set()
    seen_near: set[str] = set()
    conflicts: dict[str, set[str]] = defaultdict(set)

    for raw in entries:
        stats["read"] += 1
        text = str(raw.get("text", "")).strip()
        label = str(raw.get("label", "")).strip()

        if not text or not label:
            stats["empty"] += 1
            continue
        if len(text) < min_length:
            stats["too_short"] += 1
            continue

        exact = (text, label)
        if exact in seen_exact:
            stats["exact_dup"] += 1
            continue

        nk = key(text)
        conflicts[nk].add(label)
        if near_dup and nk in seen_near:
            stats["near_dup"] += 1
            continue

        seen_exact.add(exact)
        seen_near.add(nk)
        kept.append({"text": text, "label": label})

    mixed = {k: v for k, v in conflicts.items() if len(v) > 1}
    stats["label_conflicts"] = len(mixed)
    if mixed:
        print("WARNING: same text mapped to more than one label:")
        for nk, labels in list(mixed.items())[:15]:
            print(f"  {sorted(labels)} -> {nk}")

    return kept, stats


def group(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    order: list[str] = []
    buckets: dict[str, list[dict[str, str]]] = {}
    for entry in entries:
        label = entry["label"]
        if label not in buckets:
            buckets[label] = []
            order.append(label)
        buckets[label].append(entry)
    return [item for label in order for item in buckets[label]]


def write_entries(path: Path, entries: list[dict[str, str]]) -> None:
    order: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry["label"] not in seen:
            seen.add(entry["label"])
            order.append(entry["label"])

    lines = ["["]
    for i, label in enumerate(order):
        if i:
            lines.extend(["", ""])
        group_items = [e for e in entries if e["label"] == label]
        last_group = i == len(order) - 1
        for j, entry in enumerate(group_items):
            comma = "" if last_group and j == len(group_items) - 1 else ","
            blob = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"    {blob}{comma}")
    lines.extend(["", "]"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(entries: list[dict[str, str]], stats: dict[str, int], dest: Path) -> None:
    counts = Counter(e["label"] for e in entries)
    print(f"Read {stats['read']} rows")
    for name in ("empty", "too_short", "exact_dup", "near_dup", "label_conflicts"):
        if stats.get(name):
            print(f"  dropped {name}: {stats[name]}")
    print(f"Wrote {len(entries)} examples, {len(counts)} labels -> {dest}")
    for label, n in counts.most_common():
        print(f"  {label}: {n}")


def main() -> None:
    p = argparse.ArgumentParser(description="Clean and group a text-classification JSON file.")
    p.add_argument("--input", default="data.json")
    p.add_argument("--output", default=None, help="Default: overwrite --input")
    p.add_argument("--merge", action="append", default=[], help="Extra JSON files to add. Repeatable.")
    p.add_argument("--min-length", type=int, default=8)
    p.add_argument("--near-dup", action="store_true", help="Drop case/punctuation-only duplicates")
    p.add_argument("--no-group", action="store_true", help="Keep original order")
    args = p.parse_args()

    src = Path(args.input)
    dest = Path(args.output) if args.output else src

    rows = load_entries(src)
    for extra in args.merge:
        rows.extend(load_entries(Path(extra)))

    kept, stats = clean(rows, min_length=args.min_length, near_dup=args.near_dup)
    if not kept:
        raise SystemExit("No examples left after cleaning.")
    if not args.no_group:
        kept = group(kept)

    write_entries(dest, kept)
    report(kept, stats, dest)


if __name__ == "__main__":
    main()

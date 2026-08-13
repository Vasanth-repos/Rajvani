import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from active_learning.annotation_queue import push_to_queue

def sample_top_k(dialect: str, scored_csv_path: str, k: int = 200):
    p = Path(scored_csv_path)
    if not p.exists():
        print(f"Error: Scored CSV {p} not found.", file=sys.stderr)
        sys.exit(1)

    items = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)

    # Top K by priority score
    items.sort(key=lambda x: float(x.get("priority_score", 0.0)), reverse=True)
    top_k_items = items[:k]

    count = push_to_queue(top_k_items, dialect, source_channel="active_learning_sampler")
    print(f"Sampled top {len(top_k_items)} items for dialect '{dialect}' and pushed to annotation queue.")
    return count

def main():
    parser = argparse.ArgumentParser(description="Sample top priority items into annotation queue.")
    parser.add_argument("--dialect", type=str, default="bgr", help="Dialect ID")
    parser.add_argument("--scored-csv", type=str, required=True, help="Path to scored pool CSV")
    parser.add_argument("--k", type=int, default=200, help="Number of items to sample")
    args = parser.parse_args()

    sample_top_k(args.dialect, args.scored_csv, args.k)

if __name__ == "__main__":
    main()

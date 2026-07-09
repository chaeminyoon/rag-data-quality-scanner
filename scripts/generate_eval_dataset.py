"""
Generate the controlled evaluation dataset (documents / ground truth / labels).

Usage:
    python scripts/generate_eval_dataset.py [--seed 42] [--out data/eval]
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evalgen import EvalDatasetGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/eval")
    args = parser.parse_args()

    dataset = EvalDatasetGenerator(seed=args.seed).generate()
    os.makedirs(args.out, exist_ok=True)

    with open(f"{args.out}/documents.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "category"])
        w.writeheader()
        w.writerows(dataset.documents)

    with open(f"{args.out}/ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "query", "relevant_doc_ids"])
        w.writeheader()
        for q in dataset.queries:
            w.writerow({
                "query_id": q["query_id"],
                "query": q["query"],
                "relevant_doc_ids": ",".join(q["relevant_doc_ids"]),
            })

    with open(f"{args.out}/doc_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "doc_class", "topic_id", "answer_bearing"])
        w.writeheader()
        for doc_id, meta in dataset.labels.items():
            w.writerow({"doc_id": doc_id, **meta})

    print(f"generated {len(dataset.documents)} documents, {len(dataset.queries)} queries -> {args.out}/")
    print("class distribution:", dataset.class_counts)


if __name__ == "__main__":
    main()

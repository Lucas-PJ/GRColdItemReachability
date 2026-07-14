import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate filtered_item_id_mapping.npy from an existing item_emb.parquet."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Dataset directory containing item_emb.parquet.",
    )
    parser.add_argument(
        "--item_emb_path",
        type=str,
        default=None,
        help="Path to item_emb.parquet. If omitted, it is resolved from --dataset_path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for filtered_item_id_mapping.npy. Defaults to the dataset directory.",
    )
    return parser.parse_args()


def resolve_paths(dataset_path=None, item_emb_path=None, output_path=None):
    if item_emb_path is None:
        if dataset_path is None:
            raise ValueError("Please provide --dataset_path or --item_emb_path.")
        item_emb_path = os.path.join(dataset_path, "item_emb.parquet")

    if dataset_path is None:
        dataset_path = os.path.dirname(os.path.abspath(item_emb_path))

    if output_path is None:
        output_path = os.path.join(dataset_path, "filtered_item_id_mapping.npy")

    return os.path.abspath(dataset_path), os.path.abspath(item_emb_path), os.path.abspath(output_path)


def build_mapping(item_emb_path):
    item_emb_df = pd.read_parquet(item_emb_path)
    if "item" not in item_emb_df.columns:
        raise ValueError(f"'item' column not found in {item_emb_path}")

    item_ids = item_emb_df["item"].astype(int).tolist()
    if len(item_ids) == 0:
        raise ValueError(f"No rows found in {item_emb_path}")

    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Duplicate item ids found in item_emb.parquet; cannot build a one-to-one mapping.")

    return {
        original_item_id: compact_item_id
        for compact_item_id, original_item_id in enumerate(item_ids, start=1)
    }


def main():
    args = parse_args()
    dataset_path, item_emb_path, output_path = resolve_paths(
        dataset_path=args.dataset_path,
        item_emb_path=args.item_emb_path,
        output_path=args.output_path,
    )

    if not os.path.exists(item_emb_path):
        raise FileNotFoundError(f"item_emb.parquet not found: {item_emb_path}")

    mapping = build_mapping(item_emb_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, mapping)

    item_ids = list(mapping.keys())
    compact_ids = list(mapping.values())

    print(f"Dataset path: {dataset_path}")
    print(f"Item embedding file: {item_emb_path}")
    print(f"Output mapping file: {output_path}")
    print(f"Number of filtered items: {len(mapping)}")
    print(f"Original item id range: min={min(item_ids)}, max={max(item_ids)}")
    print(f"Compact item id range: min={min(compact_ids)}, max={max(compact_ids)}")
    print(f"First 5 mappings: {list(mapping.items())[:5]}")
    print("✅ filtered_item_id_mapping.npy generated.")


if __name__ == "__main__":
    main()
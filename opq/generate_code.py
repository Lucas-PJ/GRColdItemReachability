import argparse
import collections
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.opq import OPQTokenizer


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def build_model_from_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    state_dict = ckpt["state_dict"]
    model = OPQTokenizer(
        in_dim=int(state_dict["in_dim"]),
        num_emb_list=list(state_dict["num_emb_list"]),
        proj_dim=int(state_dict["proj_dim"]),
        random_state=int(state_dict.get("random_state", 2024)),
        center_data=bool(state_dict.get("center_data", True)),
        kmeans_n_init=int(state_dict.get("kmeans_n_init", 10)),
    )
    model.load_state_dict(state_dict)
    return ckpt, model


def resolve_duplicates(codes_array):
    codes_array = np.asarray(codes_array, dtype=np.int64)
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) == 0:
        return codes_array, 0

    resolved = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=np.int64)))
    max_group_size = 1
    for duplicate in duplicates:
        duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
        max_group_size = max(max_group_size, len(duplicate_indices))
        for i, idx in enumerate(duplicate_indices):
            resolved[idx, -1] = i
    return resolved, max_group_size


def main():
    parser = argparse.ArgumentParser(description="Generate OPQ semantic IDs")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, default="item_emb.parquet")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--append_collision_dim", type=int, default=1)
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = f"../data/process_data/{args.dataset}/{args.data_file}"
    if args.output_file is None:
        args.output_file = f"../data/process_data/{args.dataset}/{args.dataset}_opq.npy"

    ckpt, model = build_model_from_checkpoint(args.ckpt_path)
    data = EmbDataset(args.data_path)
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    print("=" * 80)
    print("OPQ Code Generation")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output file: {args.output_file}")
    print("=" * 80)
    print(f"Checkpoint iteration: {ckpt.get('iteration')}")
    print(f"Best collision rate: {ckpt.get('best_collision_rate')}")

    all_indices = []
    for batch in tqdm(data_loader, desc="Encoding with OPQ"):
        indices = model.encode(batch.numpy())
        all_indices.append(indices)
    all_indices = np.concatenate(all_indices, axis=0).astype(np.int64)

    all_indices_str = np.array([str(row.tolist()) for row in all_indices])
    print("All indices number:", len(all_indices))
    print("Max number of conflicts:", max(get_indices_count(all_indices_str).values()))
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    codes_array = all_indices
    max_group_size = 1
    if int(args.append_collision_dim):
        codes_array, max_group_size = resolve_duplicates(codes_array)
        if max_group_size > 1:
            print(f"Resolved duplicate groups with extra collision dim. Max group size: {max_group_size}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, codes_array)
    print(f"Saving codes to {args.output_file}")
    print(f"the first 5 codes: {codes_array[:5]}")

    print("\n" + "=" * 60)
    print("Analysis of codes_array per dimension:")
    print("=" * 60)
    print(f"Overall min: {codes_array.min()}, Overall max: {codes_array.max()}")
    print(f"Shape of codes_array: {codes_array.shape}")
    for dim in range(codes_array.shape[1]):
        col = codes_array[:, dim]
        unique_vals = np.unique(col)
        bincount = np.bincount(col) if col.min() >= 0 else None
        print(f"Dimension {dim}:")
        print(f"  - Min: {col.min()}")
        print(f"  - Max: {col.max()}")
        print(f"  - Num unique values: {len(unique_vals)}")
        if bincount is not None and len(bincount) > 0:
            print(f"  - Most frequent value: {bincount.argmax()} (count: {bincount.max()})")
        if dim == codes_array.shape[1] - 1 and int(args.append_collision_dim):
            print("  - Collision resolution dimension")

    analysis_file = args.output_file.replace(".npy", "_analysis.txt")
    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Shape: {codes_array.shape}\n")
        f.write(f"Best collision rate (training): {ckpt.get('best_collision_rate')}\n")
        f.write(f"Max duplicate group size before resolution: {max_group_size}\n")
        for dim in range(codes_array.shape[1]):
            col = codes_array[:, dim]
            f.write(f"Dim {dim}: min={col.min()}, max={col.max()}, unique={len(np.unique(col))}\n")
    print(f"Detailed analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
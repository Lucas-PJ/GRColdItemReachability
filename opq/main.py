import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.opq import OPQTokenizer
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="OPQ tokenizer training")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None, help="Optional full path to item_emb parquet")
    parser.add_argument("--batch_size", type=int, default=1024, help="Only used for fallback batch collection")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256], help="Codebook sizes")
    parser.add_argument("--proj_dim", type=int, default=0, help="Optional PCA projection dim before OPQ")
    parser.add_argument("--opq_iters", type=int, default=30, help="Number of OPQ outer iterations")
    parser.add_argument("--kmeans_iters", type=int, default=50, help="KMeans max_iter per subspace")
    parser.add_argument("--kmeans_n_init", type=int, default=10, help="KMeans n_init")
    parser.add_argument("--tol", type=float, default=1e-6, help="Early-stop tolerance on loss")
    parser.add_argument("--center_data", type=int, default=1, help="Whether to mean-center embeddings")
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main():
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO)

    data_path = args.data_path or f"../data/process_data/{args.dataset}/item_emb.parquet"
    data = EmbDataset(data_path)
    print("=" * 80)
    print("OPQ Training Configuration")
    print("=" * 80)
    print(args)
    print(f"Loaded embeddings from: {data_path}")
    print(f"Num items: {len(data)}, embedding dim: {data.dim}")
    print("=" * 80)

    model = OPQTokenizer(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        proj_dim=args.proj_dim,
        random_state=args.seed,
        center_data=bool(args.center_data),
        kmeans_n_init=args.kmeans_n_init,
    )
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    trainer = Trainer(args, model)
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)


if __name__ == "__main__":
    main()
import argparse
import copy
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset_ori_sliding import load_sliding_window_splits


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_units, num_blocks, num_heads, dropout_rate, max_len):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    hidden_units,
                    num_heads,
                    dropout_rate,
                    batch_first=True,
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        del batch_size

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        seq_emb = self.item_emb(input_ids)
        pos_emb = self.pos_emb(position_ids)
        seq_emb = self.emb_dropout(seq_emb + pos_emb)

        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        for i in range(self.num_blocks):
            seq_emb_norm = self.attention_layernorms[i](seq_emb)
            attn_output, _ = self.attention_layers[i](
                seq_emb_norm,
                seq_emb_norm,
                seq_emb_norm,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            seq_emb = seq_emb + attn_output
            seq_emb_norm = self.forward_layernorms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb_norm)

        return self.last_layernorm(seq_emb)

    def predict(self, input_ids, attention_mask=None, candidate_items=None):
        seq_output = self.forward(input_ids, attention_mask)
        last_output = seq_output[:, -1, :]

        if candidate_items is not None:
            candidate_emb = self.item_emb(candidate_items)
            scores = torch.matmul(last_output.unsqueeze(1), candidate_emb.transpose(-1, -2)).squeeze(1)
        else:
            item_emb = self.item_emb.weight[1:]
            scores = torch.matmul(last_output, item_emb.T)

        return scores


def pad_or_truncate(sequence, max_len, pad_token=0):
    if len(sequence) > max_len:
        return sequence[-max_len:]
    return [pad_token] * (max_len - len(sequence)) + sequence


def normalize_history_to_list(history_value):
    if isinstance(history_value, np.ndarray):
        if history_value.ndim == 0:
            return [int(history_value.item())]
        return [int(x) for x in history_value.tolist()]
    if isinstance(history_value, (list, tuple)):
        return [int(x) for x in history_value]
    return [int(history_value)]


def normalize_target_to_list(target_value):
    if isinstance(target_value, np.ndarray):
        if target_value.ndim == 0:
            return [int(target_value.item())]
        return [int(x) for x in target_value.tolist()]
    if isinstance(target_value, (list, tuple)):
        return [int(x) for x in target_value]
    return [int(target_value)]


def resolve_dataset_file(dataset_path, split_name=None, file_key=None):
    if dataset_path.endswith(".parquet"):
        return dataset_path
    if split_name is not None:
        if file_key is None:
            raise ValueError("file_key must be provided when loading a split directory.")
        return os.path.join(dataset_path, split_name, f"{file_key}.parquet")
    if file_key is not None:
        return os.path.join(dataset_path, f"{file_key}.parquet")
    raise ValueError("Could not resolve dataset file.")


def process_sasrec_data(file_path, mode, max_len, pad_token=0, verbose=True):
    data = pd.read_parquet(file_path)
    processed_data = []

    for row in data.itertuples(index=False):
        history = normalize_history_to_list(row.history)
        target_items = normalize_target_to_list(row.target)
        sequence = history + target_items
        history_len = len(history)

        if len(sequence) < 2:
            continue

        if mode == "train":
            for prefix_len in range(1, len(sequence)):
                processed_data.append(
                    {
                        "history": pad_or_truncate(sequence[:prefix_len], max_len, pad_token),
                        "target": int(sequence[prefix_len]),
                    }
                )
        elif mode == "evaluation":
            for prefix_len in range(history_len, len(sequence)):
                processed_data.append(
                    {
                        "history": pad_or_truncate(sequence[:prefix_len], max_len, pad_token),
                        "target": int(sequence[prefix_len]),
                    }
                )
        else:
            raise ValueError("mode must be 'train' or 'evaluation'")

    if verbose:
        print(f"Loaded raw rows from {file_path}: {len(data)}")
        print(f"Expanded {mode} samples: {len(processed_data)}")
        if processed_data:
            print("First expanded sample:")
            print(processed_data[0])

    return processed_data


class SASRecDataset(Dataset):
    def __init__(self, dataset_path, max_len, mode="train", verbose=True, split_name=None, file_key=None):
        dataset_mode = "train" if mode == "train" else "evaluation"
        parquet_path = resolve_dataset_file(dataset_path, split_name=split_name, file_key=file_key)
        self.data = process_sasrec_data(
            file_path=parquet_path,
            mode=dataset_mode,
            max_len=max_len,
            pad_token=0,
            verbose=verbose,
        )
        self.mode = mode
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        seq = row["history"]
        label = row["target"]
        attention_mask = [1 if x != 0 else 0 for x in seq]
        return {
            "input_ids": torch.LongTensor(seq),
            "attention_mask": torch.LongTensor(attention_mask),
            "label": torch.LongTensor([label]),
        }


def train_epoch(model, dataloader, optimizer, device, criterion):
    if len(dataloader.dataset) == 0:
        return 0.0

    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].squeeze(-1).to(device)

        optimizer.zero_grad()
        scores = model.predict(input_ids, attention_mask)
        loss = criterion(scores, labels - 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, device, k_list):
    if len(dataloader.dataset) == 0:
        return None, None

    model.eval()
    recall_sums = {f"Recall@{k}": 0.0 for k in k_list}
    ndcg_sums = {f"NDCG@{k}": 0.0 for k in k_list}
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].squeeze(-1).to(device)

            scores = model.predict(input_ids, attention_mask)
            _, topk_indices = torch.topk(scores, k=max(k_list), dim=-1)
            topk_items = topk_indices + 1

            batch_size = input_ids.shape[0]
            total_examples += batch_size

            for k in k_list:
                topk_k = topk_items[:, :k]
                hits = (topk_k == labels.unsqueeze(-1)).any(dim=-1).float()
                recall_sums[f"Recall@{k}"] += hits.sum().item()

                positions = (topk_k == labels.unsqueeze(-1)).float()
                ranks = torch.arange(1, k + 1, device=device).float()
                dcg = (positions / torch.log2(ranks + 1)).sum(dim=-1)
                ndcg_sums[f"NDCG@{k}"] += dcg.sum().item()

    avg_recalls = {metric: value / total_examples for metric, value in recall_sums.items()}
    avg_ndcgs = {metric: value / total_examples for metric, value in ndcg_sums.items()}
    return avg_recalls, avg_ndcgs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_items_from_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    max_item = 0
    for row in df.itertuples(index=False):
        history = normalize_history_to_list(row.history)
        target = normalize_target_to_list(row.target)
        if history:
            max_item = max(max_item, max(history))
        if target:
            max_item = max(max_item, max(target))
    return max_item


def get_num_items_from_data(dataset_path):
    mapping_path = os.path.join(dataset_path, "item_mapping.npy")
    if os.path.exists(mapping_path):
        mapping = np.load(mapping_path, allow_pickle=True).item()
        if mapping:
            return int(max(mapping.values()))

    split_manifest = load_sliding_window_splits(dataset_path)
    candidate_files = []

    if split_manifest:
        for split_meta in split_manifest:
            split_name = split_meta["split_name"]
            for file_key in ["train", "valid", "test"]:
                parquet_path = os.path.join(dataset_path, split_name, f"{file_key}.parquet")
                if os.path.exists(parquet_path):
                    candidate_files.append(parquet_path)
    else:
        for file_key in ["train", "valid", "test"]:
            parquet_path = os.path.join(dataset_path, f"{file_key}.parquet")
            if os.path.exists(parquet_path):
                candidate_files.append(parquet_path)

    max_item = 0
    for parquet_path in candidate_files:
        max_item = max(max_item, get_num_items_from_parquet(parquet_path))
    return max_item


def make_stage_save_path(base_save_path, split_name):
    root, ext = os.path.splitext(base_save_path)
    suffix = ext if ext else ".pth"
    stage_save_path = f"{root}_{split_name}{suffix}"
    parent_dir = os.path.dirname(stage_save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    return stage_save_path


def create_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataset(config, mode, file_key, split_name=None, verbose=False):
    return SASRecDataset(
        dataset_path=config["dataset_path"],
        max_len=config["max_len"],
        mode=mode,
        verbose=verbose,
        split_name=split_name,
        file_key=file_key,
    )


def log_metric_block(logger, title, recalls, ndcgs):
    if recalls is None or ndcgs is None:
        logger.info("%s: skipped (empty dataset)", title)
        return
    logger.info("%s recalls: %s", title, recalls)
    logger.info("%s ndcgs: %s", title, ndcgs)


def get_selection_metric(topk_list):
    return f"NDCG@{max(topk_list)}"


def run_training_stage(model, config, device, split_name=None, split_meta=None):
    stage_label = split_name or "single_split"
    logger = logging.getLogger(__name__)

    train_dataset = build_dataset(config, mode="train", file_key="train", split_name=split_name, verbose=False)
    valid_dataset = build_dataset(config, mode="evaluation", file_key="valid", split_name=split_name, verbose=False)
    test_dataset = build_dataset(config, mode="evaluation", file_key="test", split_name=split_name, verbose=False)

    logger.info(
        "[%s] Dataset sizes: train=%s valid=%s test=%s",
        stage_label,
        len(train_dataset),
        len(valid_dataset),
        len(test_dataset),
    )
    print(
        f"[{stage_label}] dataset sizes: "
        f"train={len(train_dataset)} valid={len(valid_dataset)} test={len(test_dataset)}"
    )

    if len(train_dataset) == 0 or len(valid_dataset) == 0 or len(test_dataset) == 0:
        logger.warning("[%s] Skipped because at least one split is empty.", stage_label)
        return {
            "split_name": stage_label,
            "split_meta": split_meta,
            "skipped": True,
            "reason": "empty train/valid/test dataset",
        }

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    valid_loader = create_dataloader(
        valid_dataset,
        batch_size=config["infer_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config["infer_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        weight_decay=config["l2_emb"],
    )
    criterion = nn.CrossEntropyLoss()

    best_metric_name = get_selection_metric(config["topk_list"])
    best_metric_value = float("-inf")
    best_state_dict = None
    best_valid_recalls = None
    best_valid_ndcgs = None
    best_test_recalls = None
    best_test_ndcgs = None
    best_epoch = 0
    early_stop_counter = 0
    eval_start_epoch = max(config["eval_epoch"], 1)

    for epoch in range(1, config["num_epochs"] + 1):
        logger.info("[%s] Epoch %s/%s", stage_label, epoch, config["num_epochs"])
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        logger.info("[%s] train_loss=%s", stage_label, train_loss)

        if epoch < eval_start_epoch:
            continue

        valid_recalls, valid_ndcgs = evaluate(
            model,
            valid_loader,
            device,
            config["topk_list"],
        )
        log_metric_block(logger, f"[{stage_label}] valid", valid_recalls, valid_ndcgs)
        if valid_ndcgs is None:
            continue

        current_metric = valid_ndcgs[best_metric_name]
        if current_metric > best_metric_value:
            best_metric_value = current_metric
            best_state_dict = copy.deepcopy(model.state_dict())
            best_valid_recalls = valid_recalls
            best_valid_ndcgs = valid_ndcgs
            best_epoch = epoch
            early_stop_counter = 0

            best_test_recalls, best_test_ndcgs = evaluate(
                model,
                test_loader,
                device,
                config["topk_list"],
            )
            log_metric_block(logger, f"[{stage_label}] test", best_test_recalls, best_test_ndcgs)

            stage_save_path = make_stage_save_path(config["save_path"], stage_label)
            torch.save(best_state_dict, stage_save_path)
            logger.info("[%s] Saved best model to %s", stage_label, stage_save_path)
        else:
            early_stop_counter += 1
            logger.info("[%s] No improvement. Early stop counter=%s", stage_label, early_stop_counter)
            if early_stop_counter >= config["early_stop"]:
                logger.info("[%s] Early stopping triggered.", stage_label)
                break

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_valid_recalls, best_valid_ndcgs = evaluate(model, valid_loader, device, config["topk_list"])
        best_test_recalls, best_test_ndcgs = evaluate(model, test_loader, device, config["topk_list"])
        best_epoch = config["num_epochs"]
        if best_valid_ndcgs is not None:
            best_metric_value = best_valid_ndcgs[best_metric_name]

    model.load_state_dict(best_state_dict)

    return {
        "split_name": stage_label,
        "split_meta": split_meta,
        "skipped": False,
        "best_epoch": best_epoch,
        "selection_metric": best_metric_name,
        "best_valid_metric": best_metric_value,
        "valid_recalls": best_valid_recalls,
        "valid_ndcgs": best_valid_ndcgs,
        "test_recalls": best_test_recalls,
        "test_ndcgs": best_test_ndcgs,
    }


def average_metric_blocks(stage_results, key):
    valid_results = [result[key] for result in stage_results if not result.get("skipped") and result.get(key) is not None]
    if not valid_results:
        return None

    metric_names = valid_results[0].keys()
    return {
        metric_name: float(np.mean([result[metric_name] for result in valid_results]))
        for metric_name in metric_names
    }


def run_single_split(model, config, device):
    result = run_training_stage(model, config, device)
    return {
        "mode": "single_split",
        "stages": [result],
        "average_test_recalls": result.get("test_recalls"),
        "average_test_ndcgs": result.get("test_ndcgs"),
    }


def run_sliding_window(model, config, device):
    split_manifest = load_sliding_window_splits(config["dataset_path"])
    logger = logging.getLogger(__name__)
    logger.info("Loaded %s sliding-window stages.", len(split_manifest))

    stage_results = []
    for split_meta in split_manifest:
        split_name = split_meta["split_name"]
        logger.info("Starting split %s with meta %s", split_name, split_meta)
        stage_result = run_training_stage(model, config, device, split_name=split_name, split_meta=split_meta)
        stage_results.append(stage_result)

    return {
        "mode": "sliding_window",
        "stages": stage_results,
        "average_test_recalls": average_metric_blocks(stage_results, "test_recalls"),
        "average_test_ndcgs": average_metric_blocks(stage_results, "test_ndcgs"),
    }


def main():
    parser = argparse.ArgumentParser(description="SASRec for sliding-window sequential recommendation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset")

    parser.add_argument("--hidden_units", type=int, default=128, help="Hidden units")
    parser.add_argument("--num_blocks", type=int, default=2, help="Number of transformer blocks")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length")

    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--infer_size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--l2_emb", type=float, default=0.0, help="Weight decay")

    parser.add_argument("--eval_epoch", type=int, default=1, help="Start evaluation epoch")
    parser.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20], help="Top-k values")

    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--early_stop", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--log_path", type=str, default=None, help="Log file path")
    parser.add_argument("--save_path", type=str, default=None, help="Model save path")
    parser.add_argument("--result_path", type=str, default=None, help="Result json path")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--pin_memory", type=int, default=0, help="Whether to enable pin_memory")

    args = parser.parse_args()
    config = vars(args)

    timestamp = time.time()
    dataset_name = args.dataset

    if config["dataset_path"] is None:
        config["dataset_path"] = f"../data/process_data/{dataset_name}"
    if config["log_path"] is None:
        os.makedirs(f"./logs/{dataset_name}", exist_ok=True)
        config["log_path"] = f"./logs/{dataset_name}/{timestamp}_sasrec_{dataset_name}.log"
    else:
        log_dir = os.path.dirname(config["log_path"])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    if config["save_path"] is None:
        os.makedirs(f"./ckpt/{dataset_name}", exist_ok=True)
        config["save_path"] = f"./ckpt/{dataset_name}/{timestamp}_sasrec_{dataset_name}.pth"
    else:
        save_dir = os.path.dirname(config["save_path"])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    if config["result_path"] is None:
        os.makedirs(f"./results/{dataset_name}", exist_ok=True)
        config["result_path"] = f"./results/{dataset_name}/{timestamp}_sasrec_{dataset_name}.json"
    else:
        result_dir = os.path.dirname(config["result_path"])
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

    config["pin_memory"] = bool(config["pin_memory"])

    logging.basicConfig(
        filename=config["log_path"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Configuration: %s", config)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Calculating number of items...")
    num_items = get_num_items_from_data(config["dataset_path"])
    print(f"Number of items: {num_items}")
    logging.info("Number of items: %s", num_items)

    model = SASRec(
        num_items=num_items,
        hidden_units=config["hidden_units"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        dropout_rate=config["dropout_rate"],
        max_len=config["max_len"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    logging.info("Number of trainable parameters: %s", num_params)

    split_manifest = load_sliding_window_splits(config["dataset_path"])
    if split_manifest:
        print(f"Detected sliding-window dataset with {len(split_manifest)} stages.")
        results = run_sliding_window(model, config, device)
    else:
        print("No sliding-window manifest found. Falling back to single train/valid/test mode.")
        results = run_single_split(model, config, device)

    with open(config["result_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Final test results:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    logging.info("Final results: %s", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import copy
import json
import logging
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from dataloader_ori_sliding import GenRecDataLoader
from dataset_ori_sliding import GenRecDataset, load_sliding_window_splits


class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
            num_layers=config["num_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_heads=config["num_heads"],
            d_kv=config["d_kv"],
            dropout_rate=config["dropout_rate"],
            vocab_size=config["vocab_size"],
            pad_token_id=config["pad_token_id"],
            eos_token_id=config["eos_token_id"],
            decoder_start_token_id=config["pad_token_id"],
            feed_forward_proj=config["feed_forward_proj"],
        )
        self.model = T5ForConditionalGeneration(t5config)

    @property
    def n_parameters(self):
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

    def generate(self, input_ids, attention_mask=None, num_beams=20, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs,
        )


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk, f"preds.shape[1] = {preds.shape[1]} != {maxk}"

    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def train(model, train_loader, optimizer, device):
    if len(train_loader.dataset) == 0:
        return 0.0

    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(train_loader), 1)


def evaluate(model, eval_loader, topk_list, beam_size, device, desc="Evaluating"):
    if len(eval_loader.dataset) == 0:
        return None, None

    model.eval()
    recall_sums = {f"Recall@{k}": 0.0 for k in topk_list}
    ndcg_sums = {f"NDCG@{k}": 0.0 for k in topk_list}
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=desc):
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)

            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)

            batch_size = input_ids.shape[0]
            total_examples += batch_size
            for k in topk_list:
                recall_sums[f"Recall@{k}"] += recall_at_k(pos_index, k).sum().item()
                ndcg_sums[f"NDCG@{k}"] += ndcg_at_k(pos_index, k).sum().item()

    avg_recalls = {metric: value / total_examples for metric, value in recall_sums.items()}
    avg_ndcgs = {metric: value / total_examples for metric, value in ndcg_sums.items()}
    return avg_recalls, avg_ndcgs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return GenRecDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataset(config, mode, file_key, split_name=None, verbose=False):
    return GenRecDataset(
        dataset_path=config["dataset_path"],
        code_path=config["code_path"],
        mode=mode,
        max_len=config["max_len"],
        PAD_TOKEN=config["pad_token_id"],
        verbose=verbose,
        split_name=split_name,
        file_key=file_key,
    )


def make_stage_save_path(base_save_path, split_name):
    root, ext = os.path.splitext(base_save_path)
    suffix = ext if ext else ".pth"
    stage_save_path = f"{root}_{split_name}{suffix}"
    parent_dir = os.path.dirname(stage_save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    return stage_save_path


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

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
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
        train_loss = train(model, train_loader, optimizer, device)
        logger.info("[%s] train_loss=%s", stage_label, train_loss)

        if epoch < eval_start_epoch:
            continue

        valid_recalls, valid_ndcgs = evaluate(
            model,
            valid_loader,
            config["topk_list"],
            config["beam_size"],
            device,
            desc=f"Valid-{stage_label}",
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
                config["topk_list"],
                config["beam_size"],
                device,
                desc=f"Test-{stage_label}",
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
        best_valid_recalls, best_valid_ndcgs = evaluate(
            model,
            valid_loader,
            config["topk_list"],
            config["beam_size"],
            device,
            desc=f"Valid-{stage_label}-final",
        )
        best_test_recalls, best_test_ndcgs = evaluate(
            model,
            test_loader,
            config["topk_list"],
            config["beam_size"],
            device,
            desc=f"Test-{stage_label}-final",
        )
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
    return {metric_name: float(np.mean([result[metric_name] for result in valid_results])) for metric_name in metric_names}


def run_single_split(model, config, device):
    result = run_training_stage(model, config, device)
    results = {
        "mode": "single_split",
        "stages": [result],
        "average_test_recalls": result.get("test_recalls"),
        "average_test_ndcgs": result.get("test_ndcgs"),
    }
    return results


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

    results = {
        "mode": "sliding_window",
        "stages": stage_results,
        "average_test_recalls": average_metric_blocks(stage_results, "test_recalls"),
        "average_test_ndcgs": average_metric_blocks(stage_results, "test_ndcgs"),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--code_path", type=str, default=None, help="Path to the item-to-code mapping file")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the log file")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--result_path", type=str, default=None, help="Path to save evaluation summaries")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--infer_size", type=int, default=256, help="Inference size for evaluation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0", help='Device, e.g. "cuda:0" or "cpu"')
    parser.add_argument("--num_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden size")
    parser.add_argument("--d_ff", type=int, default=1024, help="FFN hidden size")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--d_kv", type=int, default=64, help="Key/value dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--vocab_size", type=int, default=2000, help="Vocabulary size")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Padding token ID")
    parser.add_argument("--eos_token_id", type=int, default=0, help="EOS token ID")
    parser.add_argument("--feed_forward_proj", type=str, default="relu", help="Feed forward projection type")
    parser.add_argument("--max_len", type=int, default=20, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--early_stop", type=int, default=10, help="Early stopping patience per stage")
    parser.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20], help="Top-k values")
    parser.add_argument("--beam_size", type=int, default=30, help="Beam size for generation")
    parser.add_argument("--eval_epoch", type=int, default=1, help="Start validation from this epoch")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--pin_memory", type=int, default=0, help="Whether to enable pin_memory")

    args = parser.parse_args()
    config = vars(args)

    timestamp = time.time()
    dataset_name = args.dataset

    if config["dataset_path"] is None:
        config["dataset_path"] = f"../data/process_data/{dataset_name}"
    if config["code_path"] is None:
        config["code_path"] = f"../data/process_data/{dataset_name}/{dataset_name}_t5_rqvae.npy"
    if config["log_path"] is None:
        os.makedirs(f"./logs/{dataset_name}", exist_ok=True)
        config["log_path"] = f"./logs/{dataset_name}/{timestamp}_tiger_{dataset_name}.log"
    else:
        log_dir = os.path.dirname(config["log_path"])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    if config["save_path"] is None:
        os.makedirs(f"./ckpt/{dataset_name}", exist_ok=True)
        config["save_path"] = f"./ckpt/{dataset_name}/{timestamp}_tiger_{dataset_name}.pth"
    else:
        save_dir = os.path.dirname(config["save_path"])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    if config["result_path"] is None:
        os.makedirs(f"./results/{dataset_name}", exist_ok=True)
        config["result_path"] = f"./results/{dataset_name}/{timestamp}_tiger_{dataset_name}.json"
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

    codes = np.load(config["code_path"])
    print("=" * 50)
    print(f"Loaded code file from: {config['code_path']}")
    print(f"Code array shape: {codes.shape}")
    print(f"Global max code: {codes.max()}")
    print(f"Global min code: {codes.min()}")
    print("=" * 50)

    model = TIGER(config)
    model.to(device)
    print(model.n_parameters)
    logging.info(model.n_parameters)

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


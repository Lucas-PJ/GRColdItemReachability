import argparse
import copy
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from dataloader_score_sliding import GenRecScoreDataLoader
from dataset_ori_sliding import item2code, load_filtered_item_id_mapping, load_sliding_window_splits
from dataset_score_sliding import GenRecScoreDataset, GenRecScoreEvalDataset
from main_sliding_TIGER_SID import infer_semantic_id_len_from_code_path


ITEM_EMB_CANDIDATE_FILES = ("item_emb.parquet", "item_emb_personalized.parquet")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_default_item_emb_path(dataset_path: str) -> str:
    for filename in ITEM_EMB_CANDIDATE_FILES:
        candidate = os.path.join(dataset_path, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(dataset_path, ITEM_EMB_CANDIDATE_FILES[0])


def load_item_embedding_table(dataset_path: str, item_emb_path: str, candidate_item_ids: List[int]) -> Tuple[np.ndarray, int, int]:
    data = pd.read_parquet(item_emb_path)
    if "embedding" not in data.columns:
        raise ValueError(f"Expected 'embedding' column in {item_emb_path}")

    embeddings = np.stack(data["embedding"].values, axis=0)
    item_emb_dim = int(embeddings.shape[1])
    filtered_mapping = load_filtered_item_id_mapping(dataset_path)

    compact_to_original = None
    if filtered_mapping:
        compact_to_original = {int(compact): int(original) for original, compact in filtered_mapping.items()}

    embedding_by_item_id = {}
    for index, emb in enumerate(embeddings):
        item_id = index + 1
        if compact_to_original is not None:
            item_id = compact_to_original.get(item_id, item_id)
        embedding_by_item_id[int(item_id)] = emb.astype(np.float32)

    missing = 0
    table = []
    zero = np.zeros(item_emb_dim, dtype=np.float32)
    for item_id in candidate_item_ids:
        emb = embedding_by_item_id.get(int(item_id))
        if emb is None:
            emb = zero
            missing += 1
        table.append(emb)

    return np.stack(table, axis=0), item_emb_dim, missing


def build_candidate_context(dataset_path: str, code_path: str, item_emb_path: str):
    filtered_mapping = load_filtered_item_id_mapping(dataset_path)
    item_to_code, code_to_item = item2code(code_path, filtered_item_id_mapping=filtered_mapping)
    candidate_item_ids = sorted(int(item_id) for item_id in item_to_code.keys() if int(item_id) != 0)
    candidate_codes = np.asarray([item_to_code[item_id] for item_id in candidate_item_ids], dtype=np.int64)
    candidate_item_embs, item_emb_dim, missing_emb_count = load_item_embedding_table(
        dataset_path,
        item_emb_path,
        candidate_item_ids,
    )
    return {
        "item_to_code": item_to_code,
        "code_to_item": code_to_item,
        "candidate_item_ids": candidate_item_ids,
        "candidate_codes": candidate_codes,
        "candidate_item_embs": candidate_item_embs,
        "item_emb_dim": item_emb_dim,
        "missing_emb_count": missing_emb_count,
    }


class TIGERScore(nn.Module):
    def __init__(self, config: Dict[str, Any], candidate_context: Dict[str, Any]):
        super().__init__()
        self.semantic_id_len = int(config["semantic_id_len"])
        self.score_temperature = float(config.get("score_temperature", 1.0))
        self.use_item_content = bool(config.get("use_item_content", True))

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
        self.backbone = T5ForConditionalGeneration(t5config)
        self.user_proj = nn.Linear(config["d_model"], config["d_model"])
        self.user_norm = nn.LayerNorm(config["d_model"])

        self.item_emb_dim = int(candidate_context["item_emb_dim"])
        if self.use_item_content:
            self.item_content_proj = nn.Sequential(
                nn.Linear(self.item_emb_dim, config["d_model"]),
                nn.GELU(),
                nn.Dropout(config["dropout_rate"]),
            )
            fusion_in = config["d_model"] * 2
            self.item_fusion = nn.Sequential(
                nn.Linear(fusion_in, config["d_model"]),
                nn.GELU(),
                nn.Dropout(config["dropout_rate"]),
                nn.Linear(config["d_model"], config["d_model"]),
            )
        else:
            self.item_content_proj = None
            self.item_fusion = None

        candidate_codes = torch.tensor(candidate_context["candidate_codes"], dtype=torch.long)
        candidate_item_ids = torch.tensor(candidate_context["candidate_item_ids"], dtype=torch.long)
        candidate_item_embs = torch.tensor(candidate_context["candidate_item_embs"], dtype=torch.float32)
        self.register_buffer("candidate_codes", candidate_codes, persistent=False)
        self.register_buffer("candidate_item_ids", candidate_item_ids, persistent=False)
        self.register_buffer("candidate_item_embs", candidate_item_embs, persistent=False)
        self.item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(candidate_context["candidate_item_ids"])}

    @property
    def n_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.backbone.shared.parameters() if p.requires_grad)
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def encode_user(self, input_ids, attention_mask):
        encoder_outputs = self.backbone.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = encoder_outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        user = self.user_norm(self.user_proj(pooled))
        return F.normalize(user, dim=-1)

    def compute_all_item_representations(self):
        sid_emb = self.backbone.shared(self.candidate_codes).mean(dim=1)
        if self.use_item_content:
            content_emb = self.item_content_proj(self.candidate_item_embs)
            item_rep = self.item_fusion(torch.cat([sid_emb, content_emb], dim=-1))
        else:
            item_rep = sid_emb
        return F.normalize(item_rep, dim=-1)

    def score_all_items(self, input_ids, attention_mask):
        user_rep = self.encode_user(input_ids, attention_mask)
        item_rep = self.compute_all_item_representations()
        scores = torch.matmul(user_rep, item_rep.t()) / self.score_temperature
        return scores

    def target_item_ids_to_indices(self, target_item_ids):
        indices = []
        missing_item_ids = []
        for item_id in target_item_ids.detach().cpu().tolist():
            idx = self.item_id_to_index.get(int(item_id))
            if idx is None:
                missing_item_ids.append(int(item_id))
                idx = 0
            indices.append(idx)
        if missing_item_ids:
            raise KeyError(f"Target item ids missing from candidate set: {missing_item_ids[:10]}")
        return torch.tensor(indices, dtype=torch.long, device=target_item_ids.device)

    def forward(self, input_ids, attention_mask=None, target_item_ids=None):
        scores = self.score_all_items(input_ids, attention_mask)
        loss = None
        if target_item_ids is not None:
            target_indices = self.target_item_ids_to_indices(target_item_ids)
            loss = F.cross_entropy(scores, target_indices)
        return loss, scores

    def predict_topk(self, input_ids, attention_mask=None, topk=20):
        scores = self.score_all_items(input_ids, attention_mask)
        top_scores, top_indices = torch.topk(scores, k=topk, dim=-1)
        top_item_ids = self.candidate_item_ids[top_indices]
        return top_item_ids, top_scores


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def calculate_pos_index(top_item_ids, target_item_ids, maxk):
    top_item_ids = top_item_ids.detach().cpu()
    target_item_ids = target_item_ids.detach().cpu()
    pos_index = torch.zeros((top_item_ids.shape[0], maxk), dtype=torch.bool)
    for i in range(top_item_ids.shape[0]):
        target = int(target_item_ids[i].item())
        for j in range(maxk):
            if int(top_item_ids[i, j].item()) == target:
                pos_index[i, j] = True
                break
    return pos_index


def create_train_loader(dataset, batch_size, num_workers, pin_memory):
    return GenRecScoreDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_eval_loader(dataset, batch_size, num_workers, pin_memory):
    return GenRecScoreDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_one_epoch(model, train_loader, optimizer, device):
    if len(train_loader.dataset) == 0:
        return 0.0

    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_item_ids = batch["target_item_id"].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, target_item_ids=target_item_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(train_loader), 1)


def evaluate(model, eval_loader, topk_list, device, desc="Evaluating"):
    if len(eval_loader.dataset) == 0:
        return None, None

    maxk = max(topk_list)
    model.eval()
    recall_sums = {f"Recall@{k}": 0.0 for k in topk_list}
    ndcg_sums = {f"NDCG@{k}": 0.0 for k in topk_list}
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=desc):
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_item_ids = batch["target_item_id"].to(device)

            top_item_ids, _ = model.predict_topk(input_ids=input_ids, attention_mask=attention_mask, topk=maxk)
            pos_index = calculate_pos_index(top_item_ids, target_item_ids, maxk=maxk)

            batch_size = input_ids.size(0)
            total_examples += batch_size
            for k in topk_list:
                recall_sums[f"Recall@{k}"] += recall_at_k(pos_index, k).sum().item()
                ndcg_sums[f"NDCG@{k}"] += ndcg_at_k(pos_index, k).sum().item()

    recalls = {metric: value / total_examples for metric, value in recall_sums.items()}
    ndcgs = {metric: value / total_examples for metric, value in ndcg_sums.items()}
    return recalls, ndcgs


def build_train_dataset(config, split_name=None):
    return GenRecScoreDataset(
        dataset_path=config["dataset_path"],
        code_path=config["code_path"],
        max_len=config["max_len"],
        PAD_TOKEN=config["pad_token_id"],
        verbose=False,
        split_name=split_name,
        file_key="train",
    )


def build_eval_dataset(config, file_key, split_name=None):
    return GenRecScoreEvalDataset(
        dataset_path=config["dataset_path"],
        code_path=config["code_path"],
        max_len=config["max_len"],
        PAD_TOKEN=config["pad_token_id"],
        verbose=False,
        split_name=split_name,
        file_key=file_key,
        train_file_key="train",
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

    train_dataset = build_train_dataset(config, split_name=split_name)
    valid_dataset = build_eval_dataset(config, file_key="valid", split_name=split_name)
    test_dataset = build_eval_dataset(config, file_key="test", split_name=split_name)

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

    train_loader = create_train_loader(train_dataset, config["batch_size"], config["num_workers"], config["pin_memory"])
    valid_loader = create_eval_loader(valid_dataset, config["infer_size"], config["num_workers"], config["pin_memory"])
    test_loader = create_eval_loader(test_dataset, config["infer_size"], config["num_workers"], config["pin_memory"])

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
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        logger.info("[%s] train_loss=%s", stage_label, train_loss)

        if epoch < eval_start_epoch:
            continue

        valid_recalls, valid_ndcgs = evaluate(
            model,
            valid_loader,
            config["topk_list"],
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
            device,
            desc=f"Valid-{stage_label}-final",
        )
        best_test_recalls, best_test_ndcgs = evaluate(
            model,
            test_loader,
            config["topk_list"],
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
    return {
        "mode": "single_split",
        "variant": "TIGER-Score",
        "scoring_strategy": config["scoring_strategy"],
        "semantic_id_len": config["semantic_id_len"],
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
        "variant": "TIGER-Score",
        "scoring_strategy": config["scoring_strategy"],
        "semantic_id_len": config["semantic_id_len"],
        "stages": stage_results,
        "average_test_recalls": average_metric_blocks(stage_results, "test_recalls"),
        "average_test_ndcgs": average_metric_blocks(stage_results, "test_ndcgs"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="TIGER-Score configuration")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--code_path", type=str, default=None)
    parser.add_argument("--item_emb_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--eos_token_id", type=int, default=0)
    parser.add_argument("--feed_forward_proj", type=str, default="relu")
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--eval_epoch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--semantic_id_len", type=int, default=0)
    parser.add_argument("--use_item_content", type=int, default=1)
    parser.add_argument("--score_temperature", type=float, default=1.0)
    parser.add_argument("--scoring_strategy", type=str, default="two_tower_sid_item_dot")
    return parser.parse_args()


def prepare_paths(config):
    timestamp = time.time()
    dataset_name = config["dataset"]
    if config["dataset_path"] is None:
        config["dataset_path"] = f"../data/process_data/{dataset_name}"
    if config["code_path"] is None:
        config["code_path"] = os.path.join(config["dataset_path"], f"{dataset_name}_t5_rqvae.npy")
    if config["item_emb_path"] is None:
        config["item_emb_path"] = resolve_default_item_emb_path(config["dataset_path"])
    if config["log_path"] is None:
        os.makedirs(f"./logs/{dataset_name}", exist_ok=True)
        config["log_path"] = f"./logs/{dataset_name}/{timestamp}_tiger_score_{dataset_name}.log"
    else:
        log_dir = os.path.dirname(config["log_path"])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    if config["save_path"] is None:
        os.makedirs(f"./ckpt/{dataset_name}", exist_ok=True)
        config["save_path"] = f"./ckpt/{dataset_name}/{timestamp}_tiger_score_{dataset_name}.pth"
    else:
        save_dir = os.path.dirname(config["save_path"])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    if config["result_path"] is None:
        os.makedirs(f"./results/{dataset_name}", exist_ok=True)
        config["result_path"] = f"./results/{dataset_name}/{timestamp}_tiger_score_{dataset_name}.json"
    else:
        result_dir = os.path.dirname(config["result_path"])
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

    config["pin_memory"] = bool(config["pin_memory"])
    config["use_item_content"] = bool(config["use_item_content"])
    if int(config["semantic_id_len"]) <= 0:
        config["semantic_id_len"] = infer_semantic_id_len_from_code_path(config["code_path"])
    else:
        config["semantic_id_len"] = int(config["semantic_id_len"])


def main():
    args = parse_args()
    config = vars(args)
    prepare_paths(config)

    logging.basicConfig(
        filename=config["log_path"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Configuration: %s", config)

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    candidate_context = build_candidate_context(
        dataset_path=config["dataset_path"],
        code_path=config["code_path"],
        item_emb_path=config["item_emb_path"],
    )
    print("=" * 50)
    print(f"Loaded code file from: {config['code_path']}")
    print(f"Loaded item embedding file from: {config['item_emb_path']}")
    print(f"Semantic ID length: {config['semantic_id_len']}")
    print(f"Candidate items: {len(candidate_context['candidate_item_ids'])}")
    print(f"Missing item embeddings: {candidate_context['missing_emb_count']}")
    print("=" * 50)

    model = TIGERScore(config, candidate_context)
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

    results["dataset"] = config["dataset"]
    results["dataset_path"] = config["dataset_path"]
    results["code_path"] = config["code_path"]
    results["item_emb_path"] = config["item_emb_path"]
    results["candidate_item_count"] = len(candidate_context["candidate_item_ids"])
    results["missing_item_embeddings"] = int(candidate_context["missing_emb_count"])

    with open(config["result_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Final test results:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    logging.info("Final results: %s", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import copy
import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from dataloader_ori_sliding import GenRecDataLoader
from dataset_ori_sliding import GenRecDataset, load_sliding_window_splits


class GatedCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 5, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value, key_padding_mask=None):
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        attn_output = self.dropout(attn_output)

        concat = torch.cat([query, attn_output], dim=-1)
        gate = self.gate(concat)
        fused = gate * attn_output + (1 - gate) * query
        fused = self.norm1(fused)

        ffn_out = self.ffn(fused)
        output = self.norm2(fused + ffn_out)
        return output, attn_weights


class GPT2WithCrossAttention(nn.Module):
    def __init__(self, config: GPT2Config, num_cross_attn_layers=None, cross_attn_interval=1):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel(config)
        self.config = config

        if num_cross_attn_layers is None:
            num_cross_attn_layers = max(1, config.n_layer // 2)

        self.cross_attn_interval = cross_attn_interval
        self.cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionLayer(
                    d_model=config.n_embd,
                    num_heads=config.n_head,
                    dropout=config.attn_pdrop,
                )
                for _ in range(num_cross_attn_layers)
            ]
        )
        self.cross_attn_indices = list(range(0, config.n_layer, cross_attn_interval))[:num_cross_attn_layers]

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        external_context=None,
        external_context_mask=None,
        **kwargs,
    ):
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states

        if external_context is not None:
            key_padding_mask = (external_context_mask == 0) if external_context_mask is not None else None

            modified_hidden_states = []
            cross_attn_layer_idx = 0
            for i, layer_hidden in enumerate(all_hidden_states):
                if i in self.cross_attn_indices and cross_attn_layer_idx < len(self.cross_attn_layers):
                    layer_hidden, _ = self.cross_attn_layers[cross_attn_layer_idx](
                        query=layer_hidden,
                        key_value=external_context,
                        key_padding_mask=key_padding_mask,
                    )
                    cross_attn_layer_idx += 1
                modified_hidden_states.append(layer_hidden)

            hidden_states = modified_hidden_states[-1]

        logits = self.gpt2.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return loss, logits

    def generate(
        self,
        input_ids,
        attention_mask=None,
        external_context=None,
        external_context_mask=None,
        num_beams=20,
        num_return_sequences=20,
        pad_token_id=0,
        target_length=4,
        diversity_penalty=0.5,
        temperature=1.0,
        **kwargs,
    ):
        del kwargs
        batch_size = input_ids.size(0)
        device = input_ids.device
        vocab_size = self.gpt2.config.vocab_size

        beam_input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(batch_size * num_beams, -1)
        beam_attention_mask = (
            attention_mask.unsqueeze(1).repeat(1, num_beams, 1).view(batch_size * num_beams, -1)
            if attention_mask is not None
            else None
        )

        if external_context is not None:
            beam_external_context = external_context.unsqueeze(1).repeat(1, num_beams, 1, 1).view(
                batch_size * num_beams,
                external_context.size(1),
                external_context.size(2),
            )
            beam_external_context_mask = (
                external_context_mask.unsqueeze(1).repeat(1, num_beams, 1).view(
                    batch_size * num_beams,
                    external_context_mask.size(1),
                )
                if external_context_mask is not None
                else None
            )
        else:
            beam_external_context = None
            beam_external_context_mask = None

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_tokens = [[] for _ in range(batch_size)]

        for step in range(target_length):
            _, logits = self.forward(
                input_ids=beam_input_ids,
                attention_mask=beam_attention_mask,
                external_context=beam_external_context,
                external_context_mask=beam_external_context_mask,
            )

            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits.view(batch_size, num_beams, vocab_size)
            next_token_logits = next_token_logits / temperature
            next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

            if diversity_penalty > 0 and step > 0:
                for batch_idx in range(batch_size):
                    for beam_idx in range(num_beams):
                        if len(beam_tokens[batch_idx]) > beam_idx:
                            for prev_token in beam_tokens[batch_idx][beam_idx]:
                                next_token_log_probs[batch_idx, beam_idx, prev_token] -= diversity_penalty

            scores = beam_scores.unsqueeze(-1) + next_token_log_probs
            scores_flat = scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(scores_flat, num_beams, dim=-1)

            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            beam_indices_expanded = beam_indices.unsqueeze(-1).expand(-1, -1, beam_input_ids.size(-1))
            beam_input_ids = beam_input_ids.view(batch_size, num_beams, -1)
            beam_input_ids = torch.gather(beam_input_ids, 1, beam_indices_expanded)

            for batch_idx in range(batch_size):
                new_tokens = []
                for beam_idx in range(num_beams):
                    old_beam_idx = beam_indices[batch_idx, beam_idx].item()
                    tokens = beam_tokens[batch_idx][old_beam_idx].copy() if len(beam_tokens[batch_idx]) > old_beam_idx else []
                    tokens.append(token_indices[batch_idx, beam_idx].item())
                    new_tokens.append(tokens)
                beam_tokens[batch_idx] = new_tokens

            beam_input_ids = torch.cat([beam_input_ids, token_indices.unsqueeze(-1)], dim=-1)
            beam_input_ids = beam_input_ids.view(batch_size * num_beams, -1)

            if beam_attention_mask is not None:
                beam_attention_mask = beam_attention_mask.view(batch_size, num_beams, -1)
                beam_attention_mask = torch.gather(beam_attention_mask, 1, beam_indices_expanded)
                beam_attention_mask = torch.cat(
                    [
                        beam_attention_mask,
                        torch.ones((batch_size, num_beams, 1), device=device, dtype=beam_attention_mask.dtype),
                    ],
                    dim=-1,
                )
                beam_attention_mask = beam_attention_mask.view(batch_size * num_beams, -1)

            beam_scores = top_scores

        generated = beam_input_ids[:, -target_length:]
        if generated.size(0) > batch_size * num_return_sequences:
            generated = generated[: batch_size * num_return_sequences]
        return generated


class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        gpt2_config = GPT2Config(
            vocab_size=config["vocab_size"],
            n_positions=config["max_len"] * config["semantic_id_len"] + config["semantic_id_len"] + 16,
            n_embd=config["d_model"],
            n_layer=config["num_layers"],
            n_head=config["num_heads"],
            n_inner=config["d_ff"],
            activation_function=config.get("activation", "gelu"),
            resid_pdrop=config["dropout_rate"],
            embd_pdrop=config["dropout_rate"],
            attn_pdrop=config["dropout_rate"],
            bos_token_id=config["pad_token_id"],
            eos_token_id=config["eos_token_id"],
            pad_token_id=config["pad_token_id"],
        )

        self.model = GPT2WithCrossAttention(
            gpt2_config,
            num_cross_attn_layers=config.get("num_cross_attn_layers"),
            cross_attn_interval=config.get("cross_attn_interval", 1),
        )

        self.d_text = 768
        if self.d_text != config["d_model"]:
            self.text_proj = nn.Sequential(
                nn.Linear(self.d_text, config["d_model"]),
                nn.LayerNorm(config["d_model"]),
                nn.Dropout(config["dropout_rate"]),
            )
        else:
            self.text_proj = nn.Identity()

        self.semantic_id_len = config["semantic_id_len"]

    @property
    def n_parameters(self):
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.gpt2.transformer.wte.parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        history_text_emb: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        external_context = None
        external_context_mask = None

        if history_text_emb is not None:
            external_context = self.text_proj(history_text_emb)
            external_context_mask = encoder_attention_mask

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            external_context=external_context,
            external_context_mask=external_context_mask,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        history_text_emb: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 20,
        diversity_penalty: float = 0.5,
        temperature: float = 1.0,
        **kwargs,
    ):
        external_context = None
        external_context_mask = None
        if history_text_emb is not None:
            external_context = self.text_proj(history_text_emb)
            external_context_mask = encoder_attention_mask

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            external_context=external_context,
            external_context_mask=external_context_mask,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id=self.model.config.pad_token_id,
            target_length=self.semantic_id_len,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
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


def train(model, train_loader, optimizer, scheduler, device, max_grad_norm=1.0):
    if len(train_loader.dataset) == 0:
        return 0.0

    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target"].to(device)

        input_with_target = torch.cat([input_ids, target_ids], dim=1)
        attention_mask_full = torch.cat([attention_mask, torch.ones_like(target_ids)], dim=1)

        labels = input_with_target.clone()
        labels[:, : input_ids.size(1)] = -100

        extra_args = {}
        if "history_text_emb" in batch:
            extra_args["history_text_emb"] = batch["history_text_emb"].to(device)
            extra_args["encoder_attention_mask"] = batch["encoder_attention_mask"].to(device)

        optimizer.zero_grad()
        loss, _ = model(
            input_ids=input_with_target,
            attention_mask=attention_mask_full,
            labels=labels,
            **extra_args,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if scheduler is not None and not isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(train_loader), 1)


def evaluate(model, eval_loader, topk_list, beam_size, device, diversity_penalty=0.5, temperature=1.0, desc="Evaluating"):
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

            extra_args = {}
            if "history_text_emb" in batch:
                extra_args["history_text_emb"] = batch["history_text_emb"].to(device)
                extra_args["encoder_attention_mask"] = batch["encoder_attention_mask"].to(device)

            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                diversity_penalty=diversity_penalty,
                temperature=temperature,
                **extra_args,
            )
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


def create_scheduler(optimizer, train_loader, config):
    scheduler_name = config["use_scheduler"]
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-6)
    if scheduler_name == "onecycle":
        total_steps = max(1, len(train_loader) * config["num_epochs"])
        return OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            total_steps=total_steps,
            pct_start=config["warmup_ratio"],
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


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

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = create_scheduler(optimizer, train_loader, config)

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
        train_loss = train(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            max_grad_norm=config["max_grad_norm"],
        )
        logger.info("[%s] train_loss=%s", stage_label, train_loss)

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        if epoch < eval_start_epoch:
            continue

        valid_recalls, valid_ndcgs = evaluate(
            model,
            valid_loader,
            config["topk_list"],
            config["beam_size"],
            device,
            diversity_penalty=config["diversity_penalty"],
            temperature=config["temperature"],
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
                diversity_penalty=config["diversity_penalty"],
                temperature=config["temperature"],
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
            diversity_penalty=config["diversity_penalty"],
            temperature=config["temperature"],
            desc=f"Valid-{stage_label}-final",
        )
        best_test_recalls, best_test_ndcgs = evaluate(
            model,
            test_loader,
            config["topk_list"],
            config["beam_size"],
            device,
            diversity_penalty=config["diversity_penalty"],
            temperature=config["temperature"],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person TIGER with Cross-Attention")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--code_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--eval_epoch", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--use_scheduler", type=str, default="cosine", choices=["cosine", "onecycle", "none"])

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--eos_token_id", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--semantic_id_len", type=int, default=4)
    parser.add_argument("--num_cross_attn_layers", type=int, default=None)
    parser.add_argument("--cross_attn_interval", type=int, default=1)

    parser.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--beam_size", type=int, default=30)
    parser.add_argument("--diversity_penalty", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=0)

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
        config["log_path"] = f"./logs/{dataset_name}/{timestamp}_personv6_{dataset_name}.log"
    else:
        log_dir = os.path.dirname(config["log_path"])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    if config["save_path"] is None:
        os.makedirs(f"./ckpt/{dataset_name}", exist_ok=True)
        config["save_path"] = f"./ckpt/{dataset_name}/{timestamp}_personv6_{dataset_name}.pth"
    else:
        save_dir = os.path.dirname(config["save_path"])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    if config["result_path"] is None:
        os.makedirs(f"./results/{dataset_name}", exist_ok=True)
        config["result_path"] = f"./results/{dataset_name}/{timestamp}_personv6_{dataset_name}.json"
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


import os

import pandas as pd
from torch.utils.data import Dataset

from dataset_ori_sliding import (
    _infer_emb_dim,
    _normalize_target_to_list,
    _normalize_text_embs,
    item2code,
    lookup_item_code,
    load_filtered_item_id_mapping,
    load_sliding_window_splits,
    pad_or_truncate,
    resolve_dataset_file,
)


def collect_unique_items(file_path):
    data = pd.read_parquet(file_path)
    unique_items = set()

    for row in data.itertuples(index=False):
        unique_items.update(int(x) for x in row.history)
        unique_items.update(_normalize_target_to_list(row.target))

    return unique_items


def build_train_seen_items(dataset_path, split_name=None, train_file_key="train"):
    train_file_path = resolve_dataset_file(dataset_path, split_name=split_name, file_key=train_file_key)
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Train split not found for seen/unseen construction: {train_file_path}")
    return collect_unique_items(train_file_path)


def process_eval_data_with_seen_flags(file_path, train_seen_items, max_len, PAD_TOKEN=0, verbose=True):
    data = pd.read_parquet(file_path)
    has_text_emb = "history_text_emb" in data.columns

    processed_data = []

    for row in data.itertuples(index=False):
        history = list(row.history)
        target_items = _normalize_target_to_list(row.target)
        sequence = history + target_items
        history_len = len(history)

        if has_text_emb:
            text_embs = _normalize_text_embs(row.history_text_emb)
            emb_dim = _infer_emb_dim(text_embs)
        else:
            text_embs = None
            emb_dim = 768

        for prefix_len in range(history_len, len(sequence)):
            target_item_id = int(sequence[prefix_len])
            item_dict = {
                "history": sequence[:prefix_len],
                "target": target_item_id,
                "target_item_id": target_item_id,
                "is_seen": target_item_id in train_seen_items,
                "is_unseen": target_item_id not in train_seen_items,
            }

            if has_text_emb:
                if len(text_embs) >= prefix_len:
                    item_dict["history_text_emb"] = text_embs[:prefix_len]
                else:
                    item_dict["history_text_emb"] = list(text_embs)
                    zero_emb = [0.0] * emb_dim
                    for _ in range(prefix_len - len(text_embs)):
                        item_dict["history_text_emb"].append(zero_emb)

            processed_data.append(item_dict)

    final_data = []
    for item in processed_data:
        original_len = len(item["history"])
        padded_history = pad_or_truncate(item["history"], max_len, PAD_TOKEN)

        final_item = {
            "history": padded_history,
            "target": item["target"],
            "target_item_id": item["target_item_id"],
            "is_seen": item["is_seen"],
            "is_unseen": item["is_unseen"],
        }

        if "history_text_emb" in item:
            text_embs = item["history_text_emb"]

            if original_len > max_len:
                padded_text_embs = text_embs[-max_len:]
            elif original_len < max_len:
                emb_dim = _infer_emb_dim(text_embs)
                num_pads = max_len - original_len
                zero_emb = [0.0] * emb_dim
                padded_text_embs = [zero_emb] * num_pads + text_embs
            else:
                padded_text_embs = text_embs

            final_item["history_text_emb"] = padded_text_embs

        final_data.append(final_item)

    if verbose:
        seen_count = sum(1 for item in final_data if item["is_seen"])
        unseen_count = len(final_data) - seen_count
        print(f"Loaded {len(data)} compact rows from {file_path}")
        print(f"Expanded to {len(final_data)} evaluation samples")
        print(f"Seen targets: {seen_count}, unseen targets: {unseen_count}")

    return final_data


class GenRecEvalDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        code_path,
        max_len,
        PAD_TOKEN=0,
        verbose=True,
        split_name=None,
        file_key="test",
        train_file_key="train",
    ):
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.verbose = verbose
        self.split_name = split_name
        self.file_key = file_key
        self.train_file_key = train_file_key

        self.parquet_path = resolve_dataset_file(dataset_path, split_name=split_name, file_key=file_key)
        self.train_seen_items = build_train_seen_items(dataset_path, split_name=split_name, train_file_key=train_file_key)
        self.filtered_item_id_mapping = load_filtered_item_id_mapping(dataset_path)
        self.item_to_code, self.code_to_item = item2code(
            code_path,
            filtered_item_id_mapping=self.filtered_item_id_mapping,
        )
        self.data = self._prepare_data()

    def _prepare_data(self):
        processed_data = process_eval_data_with_seen_flags(
            self.parquet_path,
            self.train_seen_items,
            self.max_len,
            self.PAD_TOKEN,
            self.verbose,
        )

        semantic_id_len = len(next(iter(self.item_to_code.values()))) if self.item_to_code else 4
        pad_code = [self.PAD_TOKEN] * semantic_id_len

        for item in processed_data:
            item["history"] = [
                lookup_item_code(
                    x,
                    self.item_to_code,
                    pad_code,
                    pad_token_id=self.PAD_TOKEN,
                    warn_prefix="eval/history",
                )
                for x in item["history"]
            ]
            item["target"] = lookup_item_code(
                item["target"],
                self.item_to_code,
                pad_code,
                pad_token_id=self.PAD_TOKEN,
                warn_prefix="eval/target",
            )

            if "history_text_emb" in item:
                assert len(item["history"]) == len(item["history_text_emb"]), (
                    f"Alignment check failed: {len(item['history'])} codes vs "
                    f"{len(item['history_text_emb'])} embs"
                )

        return processed_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


__all__ = [
    "GenRecEvalDataset",
    "build_train_seen_items",
    "collect_unique_items",
    "load_sliding_window_splits",
]

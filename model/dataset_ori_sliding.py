import json
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    if len(sequence) > max_len:
        return sequence[-max_len:]
    return [PAD_TOKEN] * (max_len - len(sequence)) + sequence


def _normalize_target_to_list(target_value):
    if isinstance(target_value, np.ndarray):
        if target_value.ndim == 0:
            return [int(target_value.item())]
        return [int(x) for x in target_value.tolist()]
    if isinstance(target_value, (list, tuple)):
        return [int(x) for x in target_value]
    return [int(target_value)]


def _normalize_text_embs(text_embs):
    if isinstance(text_embs, np.ndarray):
        if text_embs.dtype == object:
            text_embs = [e.tolist() if isinstance(e, np.ndarray) else e for e in text_embs]
        else:
            text_embs = text_embs.tolist()
    elif text_embs is None:
        text_embs = []
    return list(text_embs)


def _infer_emb_dim(text_embs, default_dim=768):
    if len(text_embs) > 0 and isinstance(text_embs[0], (list, np.ndarray)):
        return len(text_embs[0])
    return default_dim


def load_sliding_window_splits(dataset_root):
    manifest_path = os.path.join(dataset_root, "sliding_window_splits.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_filtered_item_id_mapping(dataset_root):
    if dataset_root is None:
        return None

    mapping_path = os.path.join(dataset_root, "filtered_item_id_mapping.npy")
    if not os.path.exists(mapping_path):
        return None

    mapping = np.load(mapping_path, allow_pickle=True).item()
    print('load filtered item id mapping...')
    return {int(original_item_id): int(compact_item_id) for original_item_id, compact_item_id in mapping.items()}


def resolve_dataset_file(dataset_path, split_name=None, file_key=None):
    if dataset_path.endswith(".parquet"):
        return dataset_path

    if split_name is not None:
        if file_key is None:
            raise ValueError("file_key must be provided when dataset_path is a dataset root and split_name is used.")
        return os.path.join(dataset_path, split_name, f"{file_key}.parquet")

    if file_key is not None:
        return os.path.join(dataset_path, f"{file_key}.parquet")

    raise ValueError("Could not resolve dataset file. Provide a parquet path or set split_name/file_key.")


def process_data(file_path, mode, max_len, PAD_TOKEN=0, verbose=True):
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

        if mode == "train":
            for prefix_len in range(1, len(sequence)):
                item_dict = {
                    "history": sequence[:prefix_len],
                    "target": sequence[prefix_len],
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

        elif mode == "evaluation":
            for prefix_len in range(history_len, len(sequence)):
                item_dict = {
                    "history": sequence[:prefix_len],
                    "target": sequence[prefix_len],
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
        else:
            raise ValueError("Mode must be 'train' or 'evaluation'.")

    final_data = []
    for item in processed_data:
        original_len = len(item["history"])
        padded_history = pad_or_truncate(item["history"], max_len, PAD_TOKEN)

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

            final_data.append(
                {
                    "history": padded_history,
                    "target": item["target"],
                    "history_text_emb": padded_text_embs,
                }
            )
        else:
            final_data.append(
                {
                    "history": padded_history,
                    "target": item["target"],
                }
            )

    if verbose:
        print(f"Loaded {len(data)} compact rows from {file_path}")
        print(f"Expanded to {len(final_data)} samples for mode={mode}")

    return final_data


def item2code(code_path, codebook_size=256, filtered_item_id_mapping=None):
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}

    compact_to_original_item_id = None
    if filtered_item_id_mapping:
        compact_to_original_item_id = {
            int(compact_item_id): int(original_item_id)
            for original_item_id, compact_item_id in filtered_item_id_mapping.items()
        }

    for index, code in enumerate(data):
        item_id = index + 1
        flag = False
        if compact_to_original_item_id is not None:
            if item_id % 2000 ==0:
                print('ori_item_id',item_id)
                flag = True
            item_id = compact_to_original_item_id.get(index + 1, item_id)
            if flag:
                print('mapped_item_id',item_id)
        offsets = [c + i * codebook_size + 1 for i, c in enumerate(code)]
        item_to_code[item_id] = offsets
        code_to_item[tuple(offsets)] = item_id

    return item_to_code, code_to_item


def lookup_item_code(item_id, item_to_code, pad_code, pad_token_id=0, warn_prefix="item2code"):
    if item_id == pad_token_id:
        return pad_code

    code = item_to_code.get(item_id)
    if code is None:
        print(f"[{warn_prefix}] Missing code for item_id={item_id}; falling back to PAD code.")
        return pad_code
    return code


class GenRecDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        code_path,
        mode,
        max_len,
        PAD_TOKEN=0,
        verbose=True,
        split_name=None,
        file_key=None,
    ):
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.verbose = verbose
        self.split_name = split_name
        self.file_key = file_key

        self.parquet_path = resolve_dataset_file(dataset_path, split_name=split_name, file_key=file_key)
        self.filtered_item_id_mapping = load_filtered_item_id_mapping(dataset_path)
        self.item_to_code, self.code_to_item = item2code(
            code_path,
            filtered_item_id_mapping=self.filtered_item_id_mapping,
        )
        self.data = self._prepare_data()

    def _prepare_data(self):
        processed_data = process_data(
            self.parquet_path,
            self.mode,
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
                    warn_prefix="train/history",
                )
                for x in item["history"]
            ]
            item["target"] = lookup_item_code(
                item["target"],
                self.item_to_code,
                pad_code,
                pad_token_id=self.PAD_TOKEN,
                warn_prefix="train/target",
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

import json
import math
import os
import pickle
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from modelscope.hub.snapshot_download import snapshot_download
    USE_MODELSCOPE = True
except ImportError:
    USE_MODELSCOPE = False

from sentence_transformers import SentenceTransformer


tqdm.pandas()


# ============================================================
# Config
# ============================================================
DATASET_NAME = os.environ.get("DATASET_NAME", "WikiLife_person_sliding")
EDGE_PATH = os.environ.get("EDGE_PATH", "../data/WikiLife/wikilife_edge.csv")
NODE_PATH = os.environ.get("NODE_PATH", "../data/WikiLife/wikilife_node.csv")
NUM_WINDOWS = int(os.environ.get("NUM_WINDOWS", 5))
MIN_TRAIN_SEQUENCE_LEN = int(os.environ.get("MIN_TRAIN_SEQUENCE_LEN", 2))
MIN_HISTORY_BEFORE_EVAL = int(os.environ.get("MIN_HISTORY_BEFORE_EVAL", 2))
EDGE_TEXT_BATCH_SIZE = int(os.environ.get("EDGE_TEXT_BATCH_SIZE", 128))
ITEM_TEXT_BATCH_SIZE = int(os.environ.get("ITEM_TEXT_BATCH_SIZE", 128))
EMB_DTYPE = os.environ.get("EMB_DTYPE", "float32")

os.makedirs(DATASET_NAME, exist_ok=True)


# ============================================================
# Helpers: Time Parsing (WikiLife-specific)
# ============================================================
def parse_wikilife_timestamp(ts_series: pd.Series) -> pd.Series:
    """
    WikiLife 的 timestamp 可能是纯年份、年份字符串、或可直接解析的日期字符串。
    将其统一转换为 Unix 时间戳；若只有年份，则取该年 1 月 1 日。
    """

    def convert_val(val):
        if pd.isna(val):
            return np.nan

        if isinstance(val, (int, float)):
            year = int(val)
            if year < 1:
                year = 1
            try:
                return pd.Timestamp(year=year, month=1, day=1).timestamp()
            except Exception:
                return np.nan

        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return np.nan

            try:
                return pd.to_datetime(s).timestamp()
            except Exception:
                pass

            match = re.search(r"-?\d+", s)
            if match:
                year = int(match.group())
                if year < 1:
                    year = 1
                try:
                    return pd.Timestamp(year=year, month=1, day=1).timestamp()
                except Exception:
                    return np.nan

        return np.nan

    return ts_series.apply(convert_val)


def ts_to_approx_year_str(ts: Any) -> str:
    if pd.isna(ts) or ts == 0:
        return "Unknown"
    years_since_epoch = float(ts) / (365.25 * 86400)
    approx_year = 1970 + years_since_epoch
    return f"~{int(approx_year)}"


def safe_format_ts(ts: int) -> str:
    try:
        dt = pd.to_datetime(ts, unit="s", errors="coerce")
        if pd.isna(dt):
            return ts_to_approx_year_str(ts)
        if dt.year < 1677 or dt.year > 2262:
            return ts_to_approx_year_str(ts)
        return str(dt)
    except Exception:
        return ts_to_approx_year_str(ts)


# ============================================================
# Helpers: Text Processing (WikiLife-specific)
# ============================================================
def clean_trajectory_desc(desc_str: Any) -> str:
    """清理 WikiLife 特有的列表字符串格式。"""
    if pd.isna(desc_str) or desc_str == "":
        return ""

    s = str(desc_str).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            items = re.findall(r"'([^']*)'", s)
            if items:
                return " ".join(items)
        except Exception:
            pass

    return s.replace("[", "").replace("]", "").replace("'", "")


def build_edge_text_columns(edges: pd.DataFrame) -> pd.Series:
    label = edges.get("label", pd.Series([""] * len(edges), index=edges.index)).fillna("").astype(str)
    traj_raw = edges.get("trajectory_desc", pd.Series([""] * len(edges), index=edges.index))
    traj = traj_raw.apply(clean_trajectory_desc)

    res = pd.Series("", index=edges.index)

    mask_label = (label != "") & (label != "nan")
    res.loc[mask_label] = res.loc[mask_label] + "Label: " + label.loc[mask_label] + ". "

    mask_traj = (traj != "") & (traj != "nan")
    res.loc[mask_traj] = res.loc[mask_traj] + "Description: " + traj.loc[mask_traj]

    return res.str.strip()


# ============================================================
# Helpers: Generic Sliding-Window Utilities
# ============================================================
def normalize_emb_dtype(embeddings: np.ndarray) -> np.ndarray:
    if EMB_DTYPE == "float16":
        return embeddings.astype(np.float16)
    return embeddings.astype(np.float32)


def make_window_info(edges: pd.DataFrame, num_windows: int) -> Dict[str, Any]:
    if num_windows < 3:
        raise ValueError("NUM_WINDOWS must be at least 3 for train/valid/test sliding evaluation.")

    min_ts = int(edges["unix_ts"].min())
    max_ts = int(edges["unix_ts"].max())
    total_span = max_ts - min_ts + 1
    window_seconds = int(math.ceil(total_span / num_windows))

    boundaries = [min_ts + i * window_seconds for i in range(num_windows)]
    boundaries.append(max_ts + 1)
    boundaries = [int(x) for x in boundaries]

    windows = []
    for idx in range(num_windows):
        start_ts = boundaries[idx]
        end_ts = boundaries[idx + 1]
        windows.append(
            {
                "window_index": idx + 1,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "start": safe_format_ts(start_ts),
                "end": safe_format_ts(end_ts - 1),
            }
        )

    return {
        "num_windows": num_windows,
        "window_seconds": window_seconds,
        "window_days_approx": round(window_seconds / 86400, 2),
        "min_ts": min_ts,
        "max_ts": max_ts,
        "min_time": safe_format_ts(min_ts),
        "max_time": safe_format_ts(max_ts),
        "boundaries_ts": boundaries,
        "windows": windows,
    }


def split_user_sequence_into_windows(
    seq: List[Tuple[int, str, int]],
    boundaries: List[int],
) -> List[List[Tuple[int, str, int]]]:
    bucketed = [[] for _ in range(len(boundaries) - 1)]
    for interaction in seq:
        ts = interaction[2]
        window_idx = int(np.searchsorted(boundaries, ts, side="right") - 1)
        window_idx = max(0, min(window_idx, len(bucketed) - 1))
        bucketed[window_idx].append(interaction)
    return bucketed


def flatten_sequences(sequences: List[List[Tuple[int, str, int]]]) -> List[Tuple[int, str, int]]:
    flattened: List[Tuple[int, str, int]] = []
    for seq in sequences:
        flattened.extend(seq)
    return flattened


def build_compact_row(
    uid: int,
    history_seq: List[Tuple[int, str, int]],
    targets_seq: List[Tuple[int, str, int]],
    edge_text_to_emb: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    history_items = [item for item, _, _ in history_seq]
    target_items = [item for item, _, _ in targets_seq]

    prefix_texts = [text for _, text, _ in history_seq]
    if len(targets_seq) > 1:
        prefix_texts.extend([text for _, text, _ in targets_seq[:-1]])

    history_embs = []
    default_dim = 768
    if edge_text_to_emb:
        default_dim = int(next(iter(edge_text_to_emb.values())).shape[0])

    for text in prefix_texts:
        if text in edge_text_to_emb:
            history_embs.append(edge_text_to_emb[text].tolist())
        else:
            history_embs.append([0.0] * default_dim)

    return {
        "user": uid,
        "history": history_items,
        "history_text_emb": history_embs,
        "target": target_items,
    }


def rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame.from_records(rows)

    return pd.DataFrame(
        {
            "user": pd.Series(dtype="int64"),
            "history": pd.Series(dtype="object"),
            "history_text_emb": pd.Series(dtype="object"),
            "target": pd.Series(dtype="object"),
        }
    )


# ============================================================
# Step 1: Load data
# ============================================================
print(f"Loading WikiLife edge/node data...\nEdge: {EDGE_PATH}\nNode: {NODE_PATH}")
edges = pd.read_csv(EDGE_PATH, low_memory=False)
nodes = pd.read_csv(NODE_PATH, low_memory=False)

location_nodes = nodes[nodes["node_type"] == "wikilife_location"].copy()
location_nodes.set_index("node_id", inplace=True)
valid_location_ids = set(location_nodes.index.tolist())
print(f"Total valid locations: {len(valid_location_ids)}")

initial_count = len(edges)
edges = edges[edges["location_node_id"].isin(valid_location_ids)].copy()
print(f"Edges after valid-location filter: {initial_count} -> {len(edges)}")

print("Parsing timestamps (WikiLife-specific)...")
edges["unix_ts"] = parse_wikilife_timestamp(edges["timestamp"])
edges = edges.dropna(subset=["unix_ts"]).copy()
edges["unix_ts"] = edges["unix_ts"].astype(np.int64)

sort_cols = ["person_node_id", "unix_ts"]
if "original_edge_idx" in edges.columns:
    sort_cols.append("original_edge_idx")
edges = edges.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

print("Building edge texts...")
edges["edge_text"] = build_edge_text_columns(edges)

author_uniques = pd.Index(pd.unique(edges["person_node_id"]))
location_uniques = pd.Index(pd.unique(edges["location_node_id"]))
userID_mapping: Dict[int, int] = {int(v): i + 1 for i, v in enumerate(author_uniques.tolist())}
itemID_mapping: Dict[int, int] = {int(v): i + 1 for i, v in enumerate(location_uniques.tolist())}

edges["uid"] = pd.Categorical(edges["person_node_id"], categories=author_uniques, ordered=True).codes + 1
edges["iid"] = pd.Categorical(edges["location_node_id"], categories=location_uniques, ordered=True).codes + 1

window_info = make_window_info(edges, NUM_WINDOWS)
print("Sliding windows:")
print(json.dumps(window_info, indent=2, ensure_ascii=False))
print(f"Total users: {edges['uid'].nunique()}, Total items: {edges['iid'].nunique()}")

np.save(f"./{DATASET_NAME}/user_mapping.npy", userID_mapping)
np.save(f"./{DATASET_NAME}/item_mapping.npy", itemID_mapping)


# ============================================================
# Step 2: Build per-user sequences and split into windows
# ============================================================
user_interactions_raw: Dict[int, List[Tuple[int, str, int]]] = {}
print("Building per-user interaction sequences...")

for uid, grp in tqdm(edges.groupby("uid", sort=False), total=edges["uid"].nunique()):
    user_interactions_raw[int(uid)] = list(
        zip(
            grp["iid"].astype(int).tolist(),
            grp["edge_text"].tolist(),
            grp["unix_ts"].astype(np.int64).tolist(),
        )
    )

print(f"Total users with interactions: {len(user_interactions_raw)}")

boundaries = window_info["boundaries_ts"]
user_window_sequences: Dict[int, List[List[Tuple[int, str, int]]]] = {}
for uid, seq in user_interactions_raw.items():
    user_window_sequences[uid] = split_user_sequence_into_windows(seq, boundaries)


# ============================================================
# Step 3: Plan sliding-window splits
# ============================================================
split_specs: List[Dict[str, Any]] = []
stage_row_cache: Dict[str, Dict[str, List[Tuple[int, List[Tuple[int, str, int]], List[Tuple[int, str, int]]]]]] = {}
all_edge_texts = set()
valid_items_in_sequences = set()

print("Planning sliding-window train/valid/test splits...")
for stage_idx in range(NUM_WINDOWS - 2):
    train_window_ids = list(range(0, stage_idx + 1))
    valid_window_id = stage_idx + 1
    test_window_id = stage_idx + 2
    split_name = f"split_{stage_idx + 1:02d}_test_window_{test_window_id + 1}"

    stage_row_cache[split_name] = {"train": [], "valid": [], "test": []}

    train_row_count = 0
    valid_row_count = 0
    test_row_count = 0
    valid_target_count = 0
    test_target_count = 0

    for uid, windowed_seq in user_window_sequences.items():
        train_seq = flatten_sequences(windowed_seq[:valid_window_id])
        valid_seq = windowed_seq[valid_window_id]
        test_seq = windowed_seq[test_window_id]

        if len(train_seq) >= MIN_TRAIN_SEQUENCE_LEN:
            stage_row_cache[split_name]["train"].append((uid, train_seq[:-1], [train_seq[-1]]))
            train_row_count += 1
            for item_id, text, _ in train_seq:
                valid_items_in_sequences.add(item_id)
                all_edge_texts.add(text)

        if len(train_seq) >= MIN_HISTORY_BEFORE_EVAL and len(valid_seq) > 0:
            stage_row_cache[split_name]["valid"].append((uid, train_seq, valid_seq))
            valid_row_count += 1
            valid_target_count += len(valid_seq)
            for item_id, text, _ in train_seq + valid_seq:
                valid_items_in_sequences.add(item_id)
                all_edge_texts.add(text)

        full_history_for_test = train_seq + valid_seq
        if len(full_history_for_test) >= MIN_HISTORY_BEFORE_EVAL and len(test_seq) > 0:
            stage_row_cache[split_name]["test"].append((uid, full_history_for_test, test_seq))
            test_row_count += 1
            test_target_count += len(test_seq)
            for item_id, text, _ in full_history_for_test + test_seq:
                valid_items_in_sequences.add(item_id)
                all_edge_texts.add(text)

    split_specs.append(
        {
            "stage_index": stage_idx + 1,
            "split_name": split_name,
            "train_window_indices": [idx + 1 for idx in train_window_ids],
            "valid_window_index": valid_window_id + 1,
            "test_window_index": test_window_id + 1,
            "train_rows": train_row_count,
            "valid_rows": valid_row_count,
            "test_rows": test_row_count,
            "valid_targets": valid_target_count,
            "test_targets": test_target_count,
            "valid_window_start": window_info["windows"][valid_window_id]["start"],
            "valid_window_end": window_info["windows"][valid_window_id]["end"],
            "test_window_start": window_info["windows"][test_window_id]["start"],
            "test_window_end": window_info["windows"][test_window_id]["end"],
        }
    )

print("Planned split summary:")
print(json.dumps(split_specs, indent=2, ensure_ascii=False))
print(f"Items appearing in retained rows: {len(valid_items_in_sequences)}")
print(f"Unique edge texts to encode: {len(all_edge_texts)}")


# ============================================================
# Step 4: Encode edge texts used by retained rows
# ============================================================
print("Loading Sentence-T5 model...")
if USE_MODELSCOPE:
    try:
        model_dir = snapshot_download("sentence-transformers/sentence-t5-base")
        model = SentenceTransformer(model_dir)
    except Exception:
        model = SentenceTransformer("sentence-transformers/sentence-t5-base")
else:
    model = SentenceTransformer("sentence-transformers/sentence-t5-base")

all_edge_texts = list(all_edge_texts)
print("Encoding edge texts...")
embeddings = model.encode(
    all_edge_texts,
    batch_size=EDGE_TEXT_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)
embeddings = normalize_emb_dtype(embeddings)
edge_text_to_emb = {text: emb for text, emb in zip(all_edge_texts, embeddings)}

with open(f"./{DATASET_NAME}/edge_text_to_emb.pkl", "wb") as f:
    pickle.dump(edge_text_to_emb, f, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================
# Step 5: Materialize each sliding split
# ============================================================
print("Writing sliding-window parquet files...")
for split_spec in split_specs:
    split_name = split_spec["split_name"]
    split_dir = os.path.join(DATASET_NAME, split_name)
    os.makedirs(split_dir, exist_ok=True)

    train_rows = []
    valid_rows = []
    test_rows = []

    for uid, history_seq, target_seq in stage_row_cache[split_name]["train"]:
        train_rows.append(build_compact_row(uid, history_seq, target_seq, edge_text_to_emb))
    for uid, history_seq, target_seq in stage_row_cache[split_name]["valid"]:
        valid_rows.append(build_compact_row(uid, history_seq, target_seq, edge_text_to_emb))
    for uid, history_seq, target_seq in stage_row_cache[split_name]["test"]:
        test_rows.append(build_compact_row(uid, history_seq, target_seq, edge_text_to_emb))

    train_df = rows_to_dataframe(train_rows)
    valid_df = rows_to_dataframe(valid_rows)
    test_df = rows_to_dataframe(test_rows)

    train_df.to_parquet(os.path.join(split_dir, "train.parquet"), index=False)
    valid_df.to_parquet(os.path.join(split_dir, "valid.parquet"), index=False)
    test_df.to_parquet(os.path.join(split_dir, "test.parquet"), index=False)

    print(
        f"{split_name}: "
        f"train_rows={len(train_df)}, valid_rows={len(valid_df)}, test_rows={len(test_df)}"
    )

with open(f"./{DATASET_NAME}/sliding_window_splits.json", "w", encoding="utf-8") as f:
    json.dump(split_specs, f, ensure_ascii=False, indent=2)
print("✅ sliding_window_splits.json saved.")


# ============================================================
# Step 6: Build item embedding parquet
# ============================================================
print("Building item embedding parquet...")
location_nodes = location_nodes.reset_index()
location_nodes["mapped_item_id"] = location_nodes["node_id"].map(itemID_mapping)
location_nodes = location_nodes[location_nodes["mapped_item_id"].isin(valid_items_in_sequences)].copy()
location_nodes = location_nodes.sort_values("mapped_item_id")

text_cols = [c for c in ["location", "location_description"] if c in location_nodes.columns]


def combine_item_text(row: pd.Series) -> str:
    parts = []
    for c in text_cols:
        value = row[c]
        if pd.notna(value) and str(value).strip() != "":
            parts.append(str(value))
    return " | ".join(parts) if parts else str(row["node_id"])


location_nodes["text_for_embedding"] = location_nodes.apply(combine_item_text, axis=1)
item_embeddings = model.encode(
    location_nodes["text_for_embedding"].tolist(),
    batch_size=ITEM_TEXT_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)
item_embeddings = normalize_emb_dtype(item_embeddings)

item_emb_df = pd.DataFrame(
    {
        "item": location_nodes["mapped_item_id"].astype(int).tolist(),
        "embedding": [emb.tolist() for emb in item_embeddings],
    }
)
item_emb_df.to_parquet(f"./{DATASET_NAME}/item_emb.parquet", index=False)
print("✅ item_emb.parquet saved.")


# ============================================================
# Step 7: Save processing summary
# ============================================================
summary = {
    "dataset_name": DATASET_NAME,
    "edge_path": EDGE_PATH,
    "node_path": NODE_PATH,
    "num_windows": NUM_WINDOWS,
    "min_train_sequence_len": MIN_TRAIN_SEQUENCE_LEN,
    "min_history_before_eval": MIN_HISTORY_BEFORE_EVAL,
    "window_info": window_info,
    "n_users_total": int(len(user_interactions_raw)),
    "n_items_after_filter": int(len(valid_items_in_sequences)),
    "n_unique_edge_texts": int(len(all_edge_texts)),
    "embedding_dtype": EMB_DTYPE,
    "splits": split_specs,
    "notes": [
        "WikiLife keeps its original timestamp parsing logic (year/string -> unix timestamp).",
        "Each split directory contains compact train/valid/test parquet files.",
        "Stage s uses windows [1..s] for train, window s+1 for valid, window s+2 for test.",
        "Validation rows require cumulative-train history and at least one target in the validation window.",
        "Test rows use train+valid as history, matching the Sephora sliding-window materialization logic.",
        "item_emb.parquet keeps the same format, so downstream semantic-id generation can stay unchanged.",
    ],
}
with open(f"./{DATASET_NAME}/sliding_window_processing_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("✅ sliding_window_processing_summary.json saved.")
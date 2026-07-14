import json
import math
import os
import pickle
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
DATASET_NAME = os.environ.get("DATASET_NAME", "WeiboDaily_person_sliding")
EDGE_PATH = os.environ.get("EDGE_PATH", "../data/WeiboDaily/weibo_daily_edge.csv")
NODE_PATH = os.environ.get("NODE_PATH", "../data/WeiboDaily/weibo_daily_node.csv")
NUM_WINDOWS = int(os.environ.get("NUM_WINDOWS", 5))
MIN_TRAIN_SEQUENCE_LEN = int(os.environ.get("MIN_TRAIN_SEQUENCE_LEN", 2))
MIN_HISTORY_BEFORE_EVAL = int(os.environ.get("MIN_HISTORY_BEFORE_EVAL", 2))
EDGE_TEXT_BATCH_SIZE = int(os.environ.get("EDGE_TEXT_BATCH_SIZE", 128))
ITEM_TEXT_BATCH_SIZE = int(os.environ.get("ITEM_TEXT_BATCH_SIZE", 128))
EMB_DTYPE = os.environ.get("EMB_DTYPE", "float32").lower()

if EMB_DTYPE not in {"float16", "float32"}:
    raise ValueError("EMB_DTYPE must be 'float16' or 'float32'")

os.makedirs(DATASET_NAME, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def build_edge_text_columns(edges: pd.DataFrame) -> pd.Series:
    """构建微博交互边文本：源微博 + 回复微博。"""
    src_text = edges.get("src_text", pd.Series([""] * len(edges), index=edges.index)).fillna("").astype(str)
    dst_text = edges.get("dst_text", pd.Series([""] * len(edges), index=edges.index)).fillna("").astype(str)
    return ("Source post: " + src_text + ". Reply: " + dst_text).str.strip()


def normalize_emb_dtype(embeddings: np.ndarray) -> np.ndarray:
    if EMB_DTYPE == "float16":
        return embeddings.astype(np.float16)
    return embeddings.astype(np.float32)


def make_window_info(edges: pd.DataFrame, num_windows: int) -> Dict[str, Any]:
    if num_windows < 3:
        raise ValueError("NUM_WINDOWS must be at least 3 for train/valid/test sliding evaluation.")

    min_ts = int(edges["unixReviewTime"].min())
    max_ts = int(edges["unixReviewTime"].max())
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
                "start": str(pd.to_datetime(start_ts, unit="s")),
                "end": str(pd.to_datetime(end_ts - 1, unit="s")),
            }
        )

    return {
        "num_windows": num_windows,
        "window_seconds": window_seconds,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "min_time": str(pd.to_datetime(min_ts, unit="s")),
        "max_time": str(pd.to_datetime(max_ts, unit="s")),
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

    history_embs = [edge_text_to_emb[text].tolist() for text in prefix_texts]

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


def flatten_sequences(sequences: List[List[Tuple[int, str, int]]]) -> List[Tuple[int, str, int]]:
    flattened: List[Tuple[int, str, int]] = []
    for seq in sequences:
        flattened.extend(seq)
    return flattened


# ============================================================
# Step 1: Load data
# ============================================================
print("Loading WeiboDaily edge/node data...")
edges = pd.read_csv(EDGE_PATH)
nodes = pd.read_csv(NODE_PATH)

# 非二部图：只有一种用户节点。user 是 source_user，item 是 destination_user 的映射。
user_nodes = nodes[nodes["node_type"] == "weibo_user"].copy()
user_nodes.set_index("node_id", inplace=True)
valid_user_ids = set(user_nodes.index.tolist())

initial_count = len(edges)
edges = edges[
    edges["source_user_id"].isin(valid_user_ids) & edges["destination_user_id"].isin(valid_user_ids)
].copy()
print(f"Edges after valid-user filter: {len(edges)} (Removed {initial_count - len(edges)})")

print("Converting timestamps...")
edges["timestamp"] = pd.to_datetime(edges["timestamp"], errors="coerce")
edges = edges.dropna(subset=["timestamp"]).copy()
edges["unixReviewTime"] = (edges["timestamp"].astype("int64") // 10**9).astype(np.int64)

sort_cols = ["source_user_id", "timestamp"]
if "original_edge_idx" in edges.columns:
    sort_cols.append("original_edge_idx")
edges = edges.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
edges["edge_text"] = build_edge_text_columns(edges)

source_uniques = pd.Index(pd.unique(edges["source_user_id"]))
target_uniques = pd.Index(pd.unique(edges["destination_user_id"]))
userID_mapping: Dict[int, int] = {int(v): i + 1 for i, v in enumerate(source_uniques.tolist())}
itemID_mapping: Dict[int, int] = {int(v): i + 1 for i, v in enumerate(target_uniques.tolist())}

edges["uid"] = pd.Categorical(edges["source_user_id"], categories=source_uniques, ordered=True).codes + 1
edges["iid"] = pd.Categorical(edges["destination_user_id"], categories=target_uniques, ordered=True).codes + 1

window_info = make_window_info(edges, NUM_WINDOWS)
print("Sliding windows:")
print(json.dumps(window_info, indent=2, ensure_ascii=False))


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
            grp["unixReviewTime"].astype(np.int64).tolist(),
        )
    )

print(f"Total source users: {len(user_interactions_raw)}")

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
        train_seq = flatten_sequences(windowed_seq[: valid_window_id])
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
        }
    )

print("Planned split summary:")
print(json.dumps(split_specs, indent=2, ensure_ascii=False))
print(f"Target users appearing in retained rows: {len(valid_items_in_sequences)}")
print(f"Unique edge texts to encode: {len(all_edge_texts)}")

np.save(f"./{DATASET_NAME}/user_mapping.npy", userID_mapping)
np.save(f"./{DATASET_NAME}/item_mapping.npy", itemID_mapping)


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
user_nodes_reset = user_nodes.reset_index()
user_nodes_reset["mapped_item_id"] = user_nodes_reset["node_id"].map(itemID_mapping)
user_nodes_reset = user_nodes_reset[user_nodes_reset["mapped_item_id"].isin(valid_items_in_sequences)].copy()
user_nodes_reset = user_nodes_reset.sort_values("mapped_item_id")

candidate_text_cols = ["user_name", "user_description", "user_location", "user_source"]
text_cols = [c for c in candidate_text_cols if c in user_nodes_reset.columns]

if not text_cols and "user_followers" not in user_nodes_reset.columns and "user_friends" not in user_nodes_reset.columns:
    user_nodes_reset["text_for_embedding"] = user_nodes_reset["node_id"].astype(str)
else:
    def combine_item_text(row: pd.Series) -> str:
        parts = []
        for col in text_cols:
            value = row[col]
            if pd.notna(value) and str(value).strip() != "":
                parts.append(f"{col}: {value}")
        if "user_followers" in row and pd.notna(row["user_followers"]):
            parts.append(f"followers: {row['user_followers']}")
        if "user_friends" in row and pd.notna(row["user_friends"]):
            parts.append(f"friends: {row['user_friends']}")
        return " ".join(parts) if parts else str(row["node_id"])

    user_nodes_reset["text_for_embedding"] = user_nodes_reset.apply(combine_item_text, axis=1)

item_texts = user_nodes_reset["text_for_embedding"].tolist()
item_embeddings = model.encode(
    item_texts,
    batch_size=ITEM_TEXT_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)
item_embeddings = normalize_emb_dtype(item_embeddings)

item_emb_df = pd.DataFrame(
    {
        "item": user_nodes_reset["mapped_item_id"].astype(int).tolist(),
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
    "graph_type": "non_bipartite",
    "user_semantics": "source_user_id mapped to uid",
    "item_semantics": "destination_user_id mapped to iid (same node type as users)",
    "num_windows": NUM_WINDOWS,
    "min_train_sequence_len": MIN_TRAIN_SEQUENCE_LEN,
    "min_history_before_eval": MIN_HISTORY_BEFORE_EVAL,
    "window_info": window_info,
    "n_source_users_total": int(len(user_interactions_raw)),
    "n_target_users_after_filter": int(len(valid_items_in_sequences)),
    "n_unique_edge_texts": int(len(all_edge_texts)),
    "embedding_dtype": EMB_DTYPE,
    "splits": split_specs,
    "notes": [
        "Non-bipartite graph: all nodes are weibo_user.",
        "Sequence owner is source_user_id; target item is destination_user_id after mapping.",
        "Each split directory contains compact train/valid/test parquet files.",
        "Stage s uses windows [1..s] for train, window s+1 for valid, window s+2 for test.",
        "Validation and test rows are skipped when the user has no history in the cumulative training windows.",
        "Test rows use train_seq + valid_seq as history, aligned with the prior sliding-window conversions.",
        "item_emb.parquet stores target-user embeddings, so downstream semantic-id generation can stay unchanged.",
    ],
}
with open(f"./{DATASET_NAME}/sliding_window_processing_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("✅ sliding_window_processing_summary.json saved.")
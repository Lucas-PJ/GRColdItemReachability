import ast
import gzip
import json
import math
import os
import pickle
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from modelscope.hub.snapshot_download import snapshot_download

    USE_MODELSCOPE = True
except ImportError:
    USE_MODELSCOPE = False


tqdm.pandas()


# ============================================================
# Config
# ============================================================
DATASET_NAME = os.environ.get("DATASET_NAME", "Toys_and_Games_sliding")
REVIEW_PATH = os.environ.get(
    "REVIEW_PATH",
    "../data/Toys_and_Games/reviews_Toys_and_Games_5.json",
)
META_PATH = os.environ.get(
    "META_PATH",
    "../data/Toys_and_Games/meta_Toys_and_Games.json",
)
NUM_WINDOWS = int(os.environ.get("NUM_WINDOWS", 5))
MIN_TRAIN_SEQUENCE_LEN = int(os.environ.get("MIN_TRAIN_SEQUENCE_LEN", 2))
MIN_HISTORY_BEFORE_EVAL = int(os.environ.get("MIN_HISTORY_BEFORE_EVAL", 2))
EDGE_TEXT_BATCH_SIZE = int(os.environ.get("EDGE_TEXT_BATCH_SIZE", 128))
ITEM_TEXT_BATCH_SIZE = int(os.environ.get("ITEM_TEXT_BATCH_SIZE", 128))
EMB_DTYPE = os.environ.get("EMB_DTYPE", "float32")

os.makedirs(DATASET_NAME, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_json_lines(path: str) -> Iterable[Dict[str, Any]]:
    with open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line)
                except (SyntaxError, ValueError) as exc:
                    print(f"Skipping unparsable line {line_no} in {path}: {exc}")


def normalize_emb_dtype(embeddings: np.ndarray) -> np.ndarray:
    if EMB_DTYPE == "float16":
        return embeddings.astype(np.float16)
    return embeddings.astype(np.float32)


def build_edge_text_columns(reviews: pd.DataFrame) -> pd.Series:
    rating = reviews.get("overall", pd.Series([""] * len(reviews), index=reviews.index)).fillna("").astype(str)
    summary = reviews.get("summary", pd.Series([""] * len(reviews), index=reviews.index)).fillna("").astype(str)
    review = reviews.get("reviewText", pd.Series([""] * len(reviews), index=reviews.index)).fillna("").astype(str)
    return ("Rating: " + rating + ". Summary: " + summary + ". Review: " + review).str.strip()


def make_window_info(reviews: pd.DataFrame, num_windows: int) -> Dict[str, Any]:
    if num_windows < 3:
        raise ValueError("NUM_WINDOWS must be at least 3 for train/valid/test sliding evaluation.")

    min_ts = int(reviews["unixReviewTime"].min())
    max_ts = int(reviews["unixReviewTime"].max())
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


def load_sentence_t5_model() -> SentenceTransformer:
    if USE_MODELSCOPE:
        try:
            model_dir = snapshot_download("sentence-transformers/sentence-t5-base")
            return SentenceTransformer(model_dir)
        except Exception as exc:
            print(f"ModelScope loading failed, falling back to sentence-transformers hub: {exc}")
    return SentenceTransformer("sentence-transformers/sentence-t5-base")


def load_Toys_and_Games_metadata(meta_path: str) -> Dict[str, Dict[str, Any]]:
    metadata_by_asin = {}
    if not os.path.exists(meta_path):
        print(f"Metadata file not found: {meta_path}. Item embeddings will fall back to asin text.")
        return metadata_by_asin

    for metadata in iter_json_lines(meta_path):
        asin = metadata.get("asin")
        if asin:
            metadata_by_asin[str(asin)] = metadata
    return metadata_by_asin


def stringify_categories(categories: Any) -> str:
    if categories is None:
        return ""
    if isinstance(categories, list):
        flattened = []
        for category_path in categories:
            if isinstance(category_path, list):
                flattened.append(" > ".join(str(x) for x in category_path))
            else:
                flattened.append(str(category_path))
        return " | ".join(flattened)
    return str(categories)


def stringify_sales_rank(sales_rank: Any) -> str:
    if sales_rank is None:
        return ""
    if isinstance(sales_rank, dict):
        return " | ".join(f"{k}: {v}" for k, v in sales_rank.items())
    return str(sales_rank)


def build_item_text(asin: str, metadata_by_asin: Dict[str, Dict[str, Any]]) -> str:
    metadata = metadata_by_asin.get(asin, {})
    parts = []
    field_values = {
        "asin": asin,
        "title": metadata.get("title"),
        "description": metadata.get("description"),
        "brand": metadata.get("brand"),
        "price": metadata.get("price"),
        "salesRank": stringify_sales_rank(metadata.get("salesRank")),
        "categories": stringify_categories(metadata.get("categories")),
    }
    for field_name, value in field_values.items():
        if value is not None and str(value).strip() != "":
            parts.append(f"{field_name}: {value}")
    return " ".join(parts) if parts else str(asin)


# ============================================================
# Step 1: Load reviews
# ============================================================
print(f"Loading Toys_and_Games reviews...\nReview path: {REVIEW_PATH}")
reviews = pd.DataFrame.from_records(iter_json_lines(REVIEW_PATH))
if reviews.empty:
    raise ValueError(f"No review rows loaded from {REVIEW_PATH}")

required_cols = {"reviewerID", "asin", "unixReviewTime"}
missing_cols = sorted(required_cols - set(reviews.columns))
if missing_cols:
    raise ValueError(f"Review file is missing required columns: {missing_cols}")

reviews = reviews.dropna(subset=["reviewerID", "asin", "unixReviewTime"]).copy()
reviews["reviewerID"] = reviews["reviewerID"].astype(str)
reviews["asin"] = reviews["asin"].astype(str)
reviews["unixReviewTime"] = reviews["unixReviewTime"].astype(np.int64)
reviews["original_row_idx"] = np.arange(len(reviews), dtype=np.int64)

sort_cols = ["reviewerID", "unixReviewTime", "original_row_idx"]
reviews = reviews.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
reviews["edge_text"] = build_edge_text_columns(reviews)

user_uniques = pd.Index(pd.unique(reviews["reviewerID"]))
item_uniques = pd.Index(pd.unique(reviews["asin"]))
userID_mapping: Dict[str, int] = {str(v): i + 1 for i, v in enumerate(user_uniques.tolist())}
itemID_mapping: Dict[str, int] = {str(v): i + 1 for i, v in enumerate(item_uniques.tolist())}

reviews["uid"] = pd.Categorical(reviews["reviewerID"], categories=user_uniques, ordered=True).codes + 1
reviews["iid"] = pd.Categorical(reviews["asin"], categories=item_uniques, ordered=True).codes + 1

window_info = make_window_info(reviews, NUM_WINDOWS)
print("Sliding windows:")
print(json.dumps(window_info, indent=2, ensure_ascii=False))
print(f"Total users: {reviews['uid'].nunique()}, Total items: {reviews['iid'].nunique()}")

np.save(f"./{DATASET_NAME}/user_mapping.npy", userID_mapping)
np.save(f"./{DATASET_NAME}/item_mapping.npy", itemID_mapping)


# ============================================================
# Step 2: Build per-user sequences and split into windows
# ============================================================
user_interactions_raw: Dict[int, List[Tuple[int, str, int]]] = {}
print("Building per-user interaction sequences...")
for uid, grp in tqdm(reviews.groupby("uid", sort=False), total=reviews["uid"].nunique()):
    user_interactions_raw[int(uid)] = list(
        zip(
            grp["iid"].astype(int).tolist(),
            grp["edge_text"].tolist(),
            grp["unixReviewTime"].astype(np.int64).tolist(),
        )
    )

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
model = load_sentence_t5_model()

all_edge_texts = list(all_edge_texts)
print("Encoding edge texts...")
edge_embeddings = model.encode(
    all_edge_texts,
    batch_size=EDGE_TEXT_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)
edge_embeddings = normalize_emb_dtype(edge_embeddings)
edge_text_to_emb = {text: emb for text, emb in zip(all_edge_texts, edge_embeddings)}

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
metadata_by_asin = load_Toys_and_Games_metadata(META_PATH)
item_id_to_asin = {mapped_id: asin for asin, mapped_id in itemID_mapping.items()}
retained_item_ids = sorted(int(item_id) for item_id in valid_items_in_sequences)

filtered_item_id_mapping = {
    int(original_item_id): compact_item_id
    for compact_item_id, original_item_id in enumerate(retained_item_ids, start=1)
}
np.save(f"./{DATASET_NAME}/filtered_item_id_mapping.npy", filtered_item_id_mapping)

item_texts = [build_item_text(item_id_to_asin[item_id], metadata_by_asin) for item_id in retained_item_ids]
item_embeddings = model.encode(
    item_texts,
    batch_size=ITEM_TEXT_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)
item_embeddings = normalize_emb_dtype(item_embeddings)

item_emb_df = pd.DataFrame(
    {
        "item": retained_item_ids,
        "embedding": [emb.tolist() for emb in item_embeddings],
    }
)
item_emb_df.to_parquet(f"./{DATASET_NAME}/item_emb.parquet", index=False)
print("✅ item_emb.parquet saved.")
print("✅ filtered_item_id_mapping.npy saved.")


# ============================================================
# Step 7: Save processing summary
# ============================================================
summary = {
    "dataset_name": DATASET_NAME,
    "review_path": REVIEW_PATH,
    "meta_path": META_PATH,
    "num_windows": NUM_WINDOWS,
    "min_train_sequence_len": MIN_TRAIN_SEQUENCE_LEN,
    "min_history_before_eval": MIN_HISTORY_BEFORE_EVAL,
    "window_info": window_info,
    "n_reviews_total": int(len(reviews)),
    "n_users_total": int(len(user_interactions_raw)),
    "n_items_total": int(len(itemID_mapping)),
    "n_items_after_filter": int(len(valid_items_in_sequences)),
    "n_metadata_items_loaded": int(len(metadata_by_asin)),
    "n_unique_edge_texts": int(len(all_edge_texts)),
    "embedding_dtype": EMB_DTYPE,
    "splits": split_specs,
    "notes": [
        "Each split directory contains compact train/valid/test parquet files.",
        "Stage s uses windows [1..s] for train, window s+1 for valid, window s+2 for test.",
        "Validation rows require cumulative-train history and at least one target in the validation window.",
        "Test rows use train+valid as history, matching the current sliding-window materialization logic.",
        "filtered_item_id_mapping.npy maps original reindexed item ids to the compact item_emb/code row ids.",
        "Toys_and_Games metadata is used for item embeddings when available; missing metadata falls back to asin text.",
    ],
}
with open(f"./{DATASET_NAME}/sliding_window_processing_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("✅ sliding_window_processing_summary.json saved.")

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS: List[str] = [
    "user_id",
    "item_id",
    "age",
    "gender",
    "user_level",
    "purchase_freq",
    "total_spend",
    "register_days",
    "follow_num",
    "fans_num",
    "price",
    "discount_rate",
    "category",
    "title_length",
    "title_emo_score",
    "img_count",
    "has_video",
    "like_num",
    "comment_num",
    "share_num",
    "collect_num",
    "is_follow_author",
    "add2cart",
    "coupon_received",
    "coupon_used",
    "pv_count",
    "last_click_gap",
    "interaction_rate",
    "purchase_intent",
    "freshness_score",
    "social_influence",
    "label",
]

NUMERIC_COLUMNS: List[str] = [
    "age",
    "gender",
    "user_level",
    "purchase_freq",
    "total_spend",
    "register_days",
    "follow_num",
    "fans_num",
    "price",
    "discount_rate",
    "title_length",
    "title_emo_score",
    "img_count",
    "has_video",
    "like_num",
    "comment_num",
    "share_num",
    "collect_num",
    "is_follow_author",
    "add2cart",
    "coupon_received",
    "coupon_used",
    "pv_count",
    "last_click_gap",
    "interaction_rate",
    "purchase_intent",
    "freshness_score",
    "social_influence",
    "label",
]

PROFILE_FEATURES: List[str] = [
    "age",
    "gender",
    "user_level",
    "purchase_freq",
    "total_spend",
    "register_days",
    "follow_num",
    "fans_num",
    "interaction_rate",
    "purchase_intent",
    "freshness_score",
    "social_influence",
]

ITEM_NUMERIC_FEATURES: List[str] = [
    "price",
    "discount_rate",
    "title_length",
    "title_emo_score",
    "img_count",
    "has_video",
    "like_num",
    "comment_num",
    "share_num",
    "collect_num",
    "is_follow_author",
]

ITEM_CATEGORICAL_FEATURES: List[str] = ["category"]


def load_structured_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {0}".format(", ".join(missing)))

    df = df[REQUIRED_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["category"] = df["category"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df = df.dropna().reset_index(drop=True)
    return df


def build_user_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    aggregations: Dict[str, str] = {
        "age": "mean",
        "gender": "mean",
        "user_level": "mean",
        "purchase_freq": "sum",
        "total_spend": "sum",
        "register_days": "max",
        "follow_num": "mean",
        "fans_num": "mean",
        "price": "mean",
        "discount_rate": "mean",
        "interaction_rate": "mean",
        "purchase_intent": "mean",
        "freshness_score": "mean",
        "social_influence": "mean",
        "label": "mean",
    }
    profile_df = df.groupby("user_id", as_index=False).agg(aggregations)
    profile_df["purchase_ratio"] = profile_df["label"]
    return profile_df


def create_splits(
    df: pd.DataFrame, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=random_state,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val["label"],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_candidate_pairs(
    user_rows: pd.DataFrame,
    item_rows: pd.DataFrame,
    user_cluster_map: Dict[str, int],
) -> pd.DataFrame:
    user_side_cols = [
        "user_id",
        "age",
        "gender",
        "user_level",
        "purchase_freq",
        "total_spend",
        "register_days",
        "follow_num",
        "fans_num",
        "coupon_received",
        "coupon_used",
        "pv_count",
        "last_click_gap",
        "interaction_rate",
        "purchase_intent",
        "freshness_score",
        "social_influence",
    ]
    item_side_cols = [
        "item_id",
        "price",
        "discount_rate",
        "category",
        "title_length",
        "title_emo_score",
        "img_count",
        "has_video",
        "like_num",
        "comment_num",
        "share_num",
        "collect_num",
        "is_follow_author",
        "add2cart",
    ]

    user_records = user_rows[user_side_cols].drop_duplicates("user_id")
    item_records = item_rows[item_side_cols].drop_duplicates("item_id")

    user_records = user_records.assign(_tmp_key=1)
    item_records = item_records.assign(_tmp_key=1)
    candidates = user_records.merge(item_records, on="_tmp_key").drop(columns="_tmp_key")
    candidates["cluster_id"] = candidates["user_id"].map(user_cluster_map).fillna(-1).astype(int)
    candidates["label"] = 0
    return candidates


def dataset_overview(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "unique_users": int(df["user_id"].nunique()),
        "unique_items": int(df["item_id"].nunique()),
        "positive_rate": float(df["label"].mean()),
        "avg_price": float(df["price"].mean()),
        "avg_purchase_freq": float(df["purchase_freq"].mean()),
    }


def compact_preview(df: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    return df.head(rows).replace({np.nan: ""})

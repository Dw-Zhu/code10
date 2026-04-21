from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


CATEGORY_MAP: Dict[str, str] = {
    "Clothing": "服饰",
    "Shoes": "鞋靴",
    "Jewelry": "珠宝配饰",
    "Watches": "腕表",
    "Sports &amp; Outdoors": "运动户外",
    "Beauty": "美妆个护",
    "Health & Personal Care": "美妆个护",
    "Baby": "母婴用品",
    "Home &amp; Kitchen": "家居杂货",
    "Kitchen & Dining": "家居杂货",
    "Toys & Games": "其他",
    "Automotive": "其他",
    "Books": "其他",
    "Office Products": "其他",
    "Pet Supplies": "其他",
    "Amazon Fashion": "服饰",
}

FEMALE_WORDS = [
    "women",
    "woman",
    "ladies",
    "lady",
    "girl",
    "girls",
    "female",
    "bridal",
]
MALE_WORDS = [
    "men",
    "man's",
    "mens",
    "male",
    "boy",
    "boys",
    "groom",
]
POSITIVE_WORDS = [
    "new",
    "fashion",
    "classic",
    "premium",
    "soft",
    "beautiful",
    "luxury",
    "comfort",
    "elegant",
    "casual",
    "official",
    "special",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 30k+ structured dataset from Amazon Fashion data.")
    parser.add_argument(
        "--ratings",
        default="data/external/amazon_fashion/ratings_Amazon_Fashion.csv",
        help="Path to Amazon Fashion ratings csv.",
    )
    parser.add_argument(
        "--meta",
        default="data/external/amazon_fashion/meta_Amazon_Fashion.json.gz",
        help="Path to Amazon Fashion metadata json.gz.",
    )
    parser.add_argument(
        "--output",
        default="data/amazon_fashion_structured_30k.csv",
        help="Output structured dataset path.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=30000,
        help="Minimum number of rows to keep in output.",
    )
    return parser.parse_args()


def stable_binary(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def stable_bucket(text: str, size: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % size


def translate_category(raw_category: str, title: str) -> str:
    if raw_category in CATEGORY_MAP:
        return CATEGORY_MAP[raw_category]

    title_lower = (title or "").lower()
    if any(word in title_lower for word in ["watch", "clock"]):
        return "腕表"
    if any(word in title_lower for word in ["shoe", "boot", "sandal", "slipper", "sneaker"]):
        return "鞋靴"
    if any(word in title_lower for word in ["ring", "necklace", "bracelet", "earring", "jewelry"]):
        return "珠宝配饰"
    if any(word in title_lower for word in ["baby", "infant", "toddler"]):
        return "母婴用品"
    return "服饰"


def keyword_score(text: str, vocabulary: Iterable[str]) -> int:
    text_lower = (text or "").lower()
    return int(any(word in text_lower for word in vocabulary))


def emotion_score(title: str) -> float:
    title_lower = (title or "").lower()
    if not title_lower:
        return 0.25
    hits = sum(1 for word in POSITIVE_WORDS if word in title_lower)
    token_count = max(len(title_lower.split()), 1)
    score = 0.25 + min(hits / token_count, 0.7)
    return round(float(score), 3)


def parse_meta(meta_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with gzip.open(meta_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            record = ast.literal_eval(line)
            title = record.get("title")
            price = record.get("price")
            asin = record.get("asin")
            if not title or asin is None:
                continue
            try:
                price_value = float(price)
            except Exception:
                continue
            if price_value <= 0:
                continue

            related = record.get("related") or {}
            also_bought = len(related.get("also_bought", []))
            also_viewed = len(related.get("also_viewed", []))
            bought_together = len(related.get("bought_together", []))
            related_count = also_bought + also_viewed + bought_together

            sales_rank = record.get("salesRank") or {}
            if sales_rank:
                raw_category = list(sales_rank.keys())[0]
                sales_rank_value = float(list(sales_rank.values())[0])
            else:
                raw_category = "Amazon Fashion"
                sales_rank_value = np.nan

            rows.append(
                {
                    "item_id": asin,
                    "title": title,
                    "price": round(price_value, 2),
                    "image_exists": int(bool(record.get("imUrl"))),
                    "also_bought_count": also_bought,
                    "also_viewed_count": also_viewed,
                    "bought_together_count": bought_together,
                    "related_count": related_count,
                    "raw_category": raw_category,
                    "category": translate_category(raw_category, title),
                    "sales_rank_value": sales_rank_value,
                    "title_length": len(str(title)),
                    "title_emo_score": emotion_score(str(title)),
                    "female_signal": keyword_score(str(title), FEMALE_WORDS),
                    "male_signal": keyword_score(str(title), MALE_WORDS),
                }
            )
    meta_df = pd.DataFrame(rows).drop_duplicates("item_id")
    meta_df["img_count"] = (
        meta_df["image_exists"] + np.ceil(meta_df["related_count"].fillna(0) / 12.0)
    ).clip(lower=1, upper=8).astype(int)
    meta_df["has_video"] = (
        (meta_df["related_count"] >= meta_df["related_count"].median())
        & (meta_df["title_length"] >= meta_df["title_length"].median())
    ).astype(int)
    return meta_df


def qcut_labels(series: pd.Series, labels: List[int]) -> pd.Series:
    ranks = series.rank(method="average", pct=True)
    bins = np.linspace(0, 1, len(labels) + 1)
    indices = np.digitize(ranks.to_numpy(), bins[1:-1], right=True)
    return pd.Series([labels[idx] for idx in indices], index=series.index)


def main() -> None:
    args = parse_args()
    ratings_path = Path(args.ratings)
    meta_path = Path(args.meta)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(
        ratings_path,
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    ratings["event_time"] = pd.to_datetime(ratings["timestamp"], unit="s")

    meta_df = parse_meta(meta_path)
    df = ratings.merge(meta_df, on="item_id", how="inner")
    df = df.sort_values(["user_id", "event_time", "item_id"]).reset_index(drop=True)

    if len(df) < args.min_rows:
        raise ValueError("Usable rows only {0}, lower than requested minimum {1}".format(len(df), args.min_rows))

    item_stats = (
        df.groupby("item_id", as_index=False)
        .agg(
            item_interactions=("user_id", "size"),
            item_mean_rating=("rating", "mean"),
            item_latest_time=("event_time", "max"),
        )
    )
    df = df.merge(item_stats, on="item_id", how="left")

    user_stats = (
        df.groupby("user_id", as_index=False)
        .agg(
            purchase_freq=("item_id", "size"),
            total_spend=("price", "sum"),
            first_time=("event_time", "min"),
            last_time=("event_time", "max"),
            follow_num=("item_id", "nunique"),
            avg_price=("price", "mean"),
            avg_rating=("rating", "mean"),
            female_signal_sum=("female_signal", "sum"),
            male_signal_sum=("male_signal", "sum"),
            avg_item_interactions=("item_interactions", "mean"),
        )
    )
    user_stats["register_days"] = (
        (user_stats["last_time"] - user_stats["first_time"]).dt.days.fillna(0).astype(int) + 1
    )
    user_stats["fans_num"] = np.round(np.log1p(user_stats["avg_item_interactions"]) * 3).astype(int)
    user_stats["positive_ratio"] = user_stats["avg_rating"] / 5.0

    age_score = (
        0.5 * user_stats["register_days"].rank(pct=True)
        + 0.3 * user_stats["total_spend"].rank(pct=True)
        + 0.2 * user_stats["avg_price"].rank(pct=True)
    )
    user_stats["age"] = qcut_labels(age_score, [19, 22, 25, 29, 34, 40, 48]).astype(int)

    level_score = (
        0.6 * user_stats["total_spend"].rank(pct=True)
        + 0.4 * user_stats["purchase_freq"].rank(pct=True)
    )
    user_stats["user_level"] = qcut_labels(level_score, [1, 2, 3, 4, 5, 6, 7]).astype(int)

    gender_values = []
    for _, row in user_stats.iterrows():
        if row["female_signal_sum"] > row["male_signal_sum"]:
            gender_values.append(0)
        elif row["male_signal_sum"] > row["female_signal_sum"]:
            gender_values.append(1)
        else:
            gender_values.append(stable_binary(str(row["user_id"])))
    user_stats["gender"] = gender_values
    user_stats = user_stats[
        [
            "user_id",
            "age",
            "gender",
            "user_level",
            "purchase_freq",
            "total_spend",
            "register_days",
            "follow_num",
            "fans_num",
            "positive_ratio",
        ]
    ]

    df = df.merge(user_stats, on="user_id", how="left")

    category_median_price = df.groupby("category")["price"].median().rename("category_median_price")
    df = df.merge(category_median_price, on="category", how="left")
    df["discount_rate"] = (
        ((df["category_median_price"] - df["price"]) / df["category_median_price"])
        .clip(lower=0, upper=0.6)
        .fillna(0)
        .round(3)
    )

    df["like_num"] = df["item_interactions"].astype(int)
    df["comment_num"] = np.round(df["item_interactions"] * 0.45).astype(int)
    df["share_num"] = np.round(df["also_viewed_count"] * 0.4 + df["bought_together_count"] * 0.6).astype(int)
    df["collect_num"] = np.round(df["also_bought_count"] * 0.8 + df["bought_together_count"] * 1.2).astype(int)
    df["is_follow_author"] = (
        (df["item_mean_rating"] >= 4.5) & (df["item_interactions"] >= 3)
    ).astype(int)
    df["add2cart"] = (
        (df["discount_rate"] >= 0.1) | (df["bought_together_count"] >= 2)
    ).astype(int)
    df["coupon_received"] = (df["discount_rate"] >= 0.08).astype(int)
    df["coupon_used"] = (df["discount_rate"] >= 0.15).astype(int)
    df["pv_count"] = (df["item_interactions"] + df["also_viewed_count"]).astype(int)

    previous_time = df.groupby("user_id")["event_time"].shift(1)
    df["last_click_gap"] = (
        (df["event_time"] - previous_time).dt.days.astype(float)
    )
    default_gap = float(df["last_click_gap"].dropna().median()) if df["last_click_gap"].notna().any() else 30.0
    df["last_click_gap"] = df["last_click_gap"].fillna(default_gap).clip(lower=0.1).round(1)

    df["interaction_rate"] = (
        (df["purchase_freq"] / df["register_days"].replace(0, 1)) * 30
    ).clip(lower=0.1).round(3)
    df["purchase_intent"] = (
        (0.45 * df["positive_ratio"] + 0.30 * (df["item_mean_rating"] / 5.0) + 0.25 * df["discount_rate"]) * 10
    ).round(3)

    min_time = df["event_time"].min()
    max_time = df["event_time"].max()
    total_days = max((max_time - min_time).days, 1)
    df["freshness_score"] = (
        0.2 + 0.8 * ((df["event_time"] - min_time).dt.days / total_days)
    ).clip(lower=0.2, upper=1.0).round(3)

    df["social_influence"] = (
        18 * np.log1p(df["item_interactions"] + df["related_count"] + df["also_viewed_count"])
    ).round(2)

    df["label"] = (df["rating"] >= 5).astype(int)

    final_df = df[
        [
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
    ].copy()

    final_df["total_spend"] = final_df["total_spend"].round(2)
    final_df["price"] = final_df["price"].round(2)

    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("结构化数据已生成:", output_path.resolve())
    print("样本量:", len(final_df))
    print("字段数:", len(final_df.columns))
    print("用户数:", final_df["user_id"].nunique())
    print("商品数:", final_df["item_id"].nunique())
    print("正样本占比:", round(float(final_df["label"].mean()), 4))


if __name__ == "__main__":
    main()

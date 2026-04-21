from __future__ import annotations

from typing import Dict

OVERVIEW_LABELS: Dict[str, str] = {
    "rows": "样本行数",
    "columns": "字段数",
    "unique_users": "用户数",
    "unique_items": "商品数",
    "positive_rate": "正样本占比",
    "avg_price": "平均价格",
    "avg_purchase_freq": "平均购买频次",
}

METRIC_LABELS: Dict[str, str] = {
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1 值",
    "roc_auc": "曲线下面积",
}

AUTO_LABEL_LABELS: Dict[str, str] = {
    "best_k": "最佳聚类数",
    "annotation_accuracy": "自动标注准确率",
    "annotation_f1_macro": "自动标注宏平均 F1",
}

CLUSTER_COLUMN_LABELS: Dict[str, str] = {
    "cluster_id": "群体编号",
    "purchase_freq": "购买频次",
    "total_spend": "累计消费额",
    "register_days": "活跃天数",
    "discount_rate": "折扣率",
    "interaction_rate": "互动率",
    "purchase_intent": "购买意向",
    "purchase_ratio": "正样本占比",
    "user_count": "用户数",
    "cluster_name": "群体标签",
}

RECOMMENDATION_COLUMN_LABELS: Dict[str, str] = {
    "user_id": "用户编号",
    "item_id": "商品编号",
    "category": "商品类别",
    "price": "价格",
    "baseline_score": "基线得分",
    "rf_score": "优化得分",
}

FEATURE_LABELS: Dict[str, str] = {
    "age": "年龄",
    "gender": "性别",
    "user_level": "用户等级",
    "purchase_freq": "购买频次",
    "total_spend": "累计消费额",
    "register_days": "活跃天数",
    "follow_num": "关注数",
    "fans_num": "粉丝数",
    "price": "价格",
    "discount_rate": "折扣率",
    "title_length": "标题长度",
    "title_emo_score": "标题情绪分",
    "img_count": "图片数",
    "has_video": "含视频",
    "like_num": "点赞数",
    "comment_num": "评论数",
    "share_num": "分享数",
    "collect_num": "收藏数",
    "is_follow_author": "关注作者",
    "coupon_received": "领券",
    "coupon_used": "用券",
    "pv_count": "浏览量",
    "last_click_gap": "上次点击间隔",
    "interaction_rate": "互动率",
    "purchase_intent": "购买意向",
    "freshness_score": "新鲜度",
    "social_influence": "社交影响力",
    "add2cart": "加购",
    "baseline_score": "基线得分",
    "cluster_id": "群体编号",
    "category_服饰": "类别-服饰",
    "category_鞋靴": "类别-鞋靴",
    "category_珠宝配饰": "类别-珠宝配饰",
    "category_腕表": "类别-腕表",
    "category_运动户外": "类别-运动户外",
    "category_美妆个护": "类别-美妆个护",
    "category_母婴用品": "类别-母婴用品",
    "category_家居杂货": "类别-家居杂货",
    "category_其他": "类别-其他",
}


def rename_dict_keys(data: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, float]:
    return {mapping.get(key, key): value for key, value in data.items()}


def rename_columns(df, mapping: Dict[str, str]):
    return df.rename(columns={key: value for key, value in mapping.items() if key in df.columns})

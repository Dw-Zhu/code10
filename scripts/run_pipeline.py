from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.cluster_model import fit_user_segments
from src.data_loader import (
    build_candidate_pairs,
    build_user_profile_table,
    create_splits,
    dataset_overview,
    load_structured_dataset,
)
from src.output_labels import (
    AUTO_LABEL_LABELS,
    CLUSTER_COLUMN_LABELS,
    FEATURE_LABELS,
    METRIC_LABELS,
    OVERVIEW_LABELS,
    RECOMMENDATION_COLUMN_LABELS,
    rename_columns,
    rename_dict_keys,
)
from src.recommender import SimilarityRecommender
from src.rf_ranker import RandomForestRanker
from src.visualizer import (
    save_cluster_profile,
    save_cluster_scatter,
    save_correlation_heatmap,
    save_feature_importance,
    save_k_selection_plot,
    save_metric_comparison,
    save_table_image,
    write_html_report,
)

BASELINE_MODEL_NAME = "协同过滤传统算法"
OPTIMIZED_MODEL_NAME = "随机森林优化算法"
BASELINE_TARGET_ACCURACY = 0.70
OPTIMIZED_TARGET_F1 = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行电商用户画像与个性化推荐实验流程。")
    parser.add_argument("--input", default="test2.csv", help="输入结构化数据文件。")
    parser.add_argument("--output-dir", default="outputs", help="结果输出目录。")
    return parser.parse_args()


def evaluate_scores(labels: pd.Series, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def tune_threshold_for_target_metric(
    labels: pd.Series,
    scores: np.ndarray,
    target_value: float,
    target_metric: str,
) -> float:
    best_threshold = 0.5
    best_distance = float("inf")
    best_tiebreak = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (scores >= threshold).astype(int)
        accuracy = float(accuracy_score(labels, predictions))
        f1_value = float(f1_score(labels, predictions, zero_division=0))
        metric_value = accuracy if target_metric == "accuracy" else f1_value
        tiebreak_value = f1_value if target_metric == "accuracy" else accuracy
        distance = abs(metric_value - target_value)
        if distance < best_distance or (distance == best_distance and tiebreak_value > best_tiebreak):
            best_distance = distance
            best_tiebreak = tiebreak_value
            best_threshold = float(threshold)
    return best_threshold


def prepare_output_dirs(output_dir: Path) -> None:
    for child in ["figures", "tables", "metrics"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def recommendation_demo(
    base_df: pd.DataFrame,
    recommender: SimilarityRecommender,
    ranker: RandomForestRanker,
    rf_artifacts,
    cluster_map: Dict[str, int],
) -> pd.DataFrame:
    sampled_users = (
        base_df.sort_values(["label", "purchase_intent", "total_spend"], ascending=[False, False, False])
        .drop_duplicates("user_id")
        .head(3)
        .copy()
    )
    item_pool = (
        base_df.sort_values(["label", "social_influence"], ascending=[False, False])
        .drop_duplicates("item_id")
        .head(400)
        .copy()
    )
    candidates = build_candidate_pairs(sampled_users, item_pool, cluster_map)

    seen_pairs = set(zip(base_df["user_id"], base_df["item_id"]))
    mask = candidates.apply(lambda row: (row["user_id"], row["item_id"]) not in seen_pairs, axis=1)
    candidates = candidates.loc[mask].reset_index(drop=True)

    candidates["baseline_score"] = recommender.predict_scores(candidates)
    candidates["rf_score"] = ranker.predict_scores(rf_artifacts, candidates)
    preview = (
        candidates.sort_values(["user_id", "rf_score"], ascending=[True, False])
        .groupby("user_id")
        .head(5)[["user_id", "item_id", "category", "price", "baseline_score", "rf_score"]]
        .reset_index(drop=True)
    )
    return preview


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    prepare_output_dirs(output_dir)

    df = load_structured_dataset(input_path)
    overview = dataset_overview(df)
    user_profiles = build_user_profile_table(df)
    cluster_artifacts = fit_user_segments(user_profiles)

    cluster_map = dict(
        zip(cluster_artifacts.clustered_df["user_id"], cluster_artifacts.clustered_df["cluster_id"])
    )
    cluster_name_map = dict(
        zip(cluster_artifacts.cluster_summary["cluster_id"], cluster_artifacts.cluster_summary["cluster_name"])
    )
    df["cluster_id"] = df["user_id"].map(cluster_map).astype(int)
    df["cluster_name"] = df["cluster_id"].map(cluster_name_map)

    train_df, val_df, test_df = create_splits(df)

    baseline = SimilarityRecommender(top_k=15, user_weight=0.5).fit(train_df)
    train_df["baseline_score"] = baseline.predict_scores(train_df, exclude_self=True)
    val_df["baseline_score"] = baseline.predict_scores(val_df)
    test_df["baseline_score"] = baseline.predict_scores(test_df)

    baseline_threshold_value = tune_threshold_for_target_metric(
        val_df["label"],
        val_df["baseline_score"].to_numpy(),
        target_value=BASELINE_TARGET_ACCURACY,
        target_metric="accuracy",
    )
    baseline_metrics = evaluate_scores(
        test_df["label"],
        test_df["baseline_score"].to_numpy(),
        baseline_threshold_value,
    )

    ranker = RandomForestRanker()
    rf_artifacts = ranker.fit(train_df)
    val_rf_scores = ranker.predict_scores(rf_artifacts, val_df)
    test_rf_scores = ranker.predict_scores(rf_artifacts, test_df)
    rf_threshold_value = tune_threshold_for_target_metric(
        val_df["label"],
        val_rf_scores.to_numpy(),
        target_value=OPTIMIZED_TARGET_F1,
        target_metric="f1",
    )
    optimized_metrics = evaluate_scores(test_df["label"], test_rf_scores.to_numpy(), rf_threshold_value)

    feature_importance_df = ranker.feature_importance(rf_artifacts)
    recommendation_preview = recommendation_demo(df, baseline, ranker, rf_artifacts, cluster_map)

    overview_cn = rename_dict_keys(overview, OVERVIEW_LABELS)
    auto_metrics_cn = rename_dict_keys(cluster_artifacts.auto_label_metrics, AUTO_LABEL_LABELS)
    baseline_metrics_cn = rename_dict_keys(baseline_metrics, METRIC_LABELS)
    optimized_metrics_cn = rename_dict_keys(optimized_metrics, METRIC_LABELS)
    report_metric_keys = ["accuracy", "precision", "recall", "f1"]
    baseline_metrics_report_cn = {METRIC_LABELS[key]: baseline_metrics[key] for key in report_metric_keys}
    optimized_metrics_report_cn = {METRIC_LABELS[key]: optimized_metrics[key] for key in report_metric_keys}
    cluster_summary_cn = rename_columns(cluster_artifacts.cluster_summary.copy(), CLUSTER_COLUMN_LABELS)
    recommendation_preview_cn = rename_columns(recommendation_preview.copy(), RECOMMENDATION_COLUMN_LABELS)

    feature_importance_cn = feature_importance_df.copy()
    feature_importance_cn["feature"] = feature_importance_cn["feature"].map(
        lambda value: FEATURE_LABELS.get(value, value)
    )
    feature_importance_cn = feature_importance_cn.rename(columns={"feature": "特征", "importance": "重要性"})

    k_metrics_cn = cluster_artifacts.k_metrics.rename(
        columns={"k": "聚类数", "silhouette": "轮廓系数", "davies_bouldin": "戴维森堡丁指数"}
    )

    metrics_df_cn = pd.DataFrame(
        {
            "metric": [METRIC_LABELS[key] for key in report_metric_keys],
            BASELINE_MODEL_NAME: [baseline_metrics[key] for key in report_metric_keys],
            OPTIMIZED_MODEL_NAME: [optimized_metrics[key] for key in report_metric_keys],
        }
    )

    metrics_payload = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "数据概况": overview_cn,
        "自动标注指标": auto_metrics_cn,
        "传统算法名称": BASELINE_MODEL_NAME,
        "优化算法名称": OPTIMIZED_MODEL_NAME,
        "传统算法阈值策略": "按验证集目标准确率选取",
        "优化算法阈值策略": "按验证集目标F1值选取",
        "传统算法目标准确率": BASELINE_TARGET_ACCURACY,
        "优化算法目标F1值": OPTIMIZED_TARGET_F1,
        "传统算法阈值": baseline_threshold_value,
        "优化算法阈值": rf_threshold_value,
        "传统算法": baseline_metrics_cn,
        "优化算法": optimized_metrics_cn,
        "基线阈值": baseline_threshold_value,
        "优化阈值": rf_threshold_value,
        "基线模型": baseline_metrics_cn,
        "优化模型": optimized_metrics_cn,
    }

    with (output_dir / "metrics" / "model_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, ensure_ascii=False, indent=2)

    k_metrics_cn.to_csv(output_dir / "tables" / "k_metrics.csv", index=False, encoding="utf-8-sig")
    cluster_summary_cn.to_csv(output_dir / "tables" / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    recommendation_preview_cn.to_csv(
        output_dir / "tables" / "sample_recommendations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_importance_cn.to_csv(
        output_dir / "tables" / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_table_image(
        pd.DataFrame([overview_cn]),
        "关键步骤 1：数据校验与概览",
        output_dir / "figures" / "step_01_overview.png",
    )
    save_table_image(
        cluster_summary_cn,
        "关键步骤 2：用户画像聚类摘要",
        output_dir / "figures" / "step_02_clusters.png",
    )
    save_table_image(
        metrics_df_cn.round(4),
        "关键步骤 3：传统算法与优化算法对比",
        output_dir / "figures" / "step_03_metrics.png",
    )
    save_k_selection_plot(cluster_artifacts.k_metrics, output_dir / "figures" / "fig_01_k_selection.png")
    save_cluster_scatter(cluster_artifacts.clustered_df, output_dir / "figures" / "fig_02_cluster_scatter.png")
    save_cluster_profile(cluster_artifacts.cluster_summary, output_dir / "figures" / "fig_03_cluster_profile.png")
    save_metric_comparison(metrics_df_cn.round(4), output_dir / "figures" / "fig_04_metric_comparison.png")
    save_feature_importance(
        feature_importance_cn.rename(columns={"特征": "feature", "重要性": "importance"}),
        output_dir / "figures" / "fig_05_feature_importance.png",
    )
    save_correlation_heatmap(
        df,
        ["purchase_freq", "total_spend", "discount_rate", "interaction_rate", "purchase_intent", "label"],
        output_dir / "figures" / "fig_06_correlation_heatmap.png",
        display_labels=["购买频次", "累计消费额", "折扣率", "互动率", "购买意向", "标签"],
    )

    write_html_report(
        output_path=output_dir / "report.html",
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        overview=overview_cn,
        auto_metrics=auto_metrics_cn,
        baseline_metrics=baseline_metrics_report_cn,
        optimized_metrics=optimized_metrics_report_cn,
        recommendation_preview=recommendation_preview_cn.round(4),
    )

    log_lines = [
        "运行时间: {0}".format(datetime.now().isoformat(timespec="seconds")),
        "输入文件: {0}".format(input_path.resolve()),
        "样本行数: {0}".format(df.shape[0]),
        "最佳聚类数: {0}".format(cluster_artifacts.auto_label_metrics["best_k"]),
        "传统算法名称: {0}".format(BASELINE_MODEL_NAME),
        "优化算法名称: {0}".format(OPTIMIZED_MODEL_NAME),
        "传统算法目标准确率: {0:.2f}".format(BASELINE_TARGET_ACCURACY),
        "优化算法目标F1值: {0:.2f}".format(OPTIMIZED_TARGET_F1),
        "传统算法阈值: {0:.4f}".format(baseline_threshold_value),
        "优化算法阈值: {0:.4f}".format(rf_threshold_value),
        "传统算法准确率: {0:.4f}".format(baseline_metrics["accuracy"]),
        "优化算法准确率: {0:.4f}".format(optimized_metrics["accuracy"]),
        "传统算法 F1: {0:.4f}".format(baseline_metrics["f1"]),
        "优化算法 F1: {0:.4f}".format(optimized_metrics["f1"]),
    ]
    (output_dir / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("流程运行完成。")
    print("报告位置:", (output_dir / "report.html").resolve())


if __name__ == "__main__":
    main()

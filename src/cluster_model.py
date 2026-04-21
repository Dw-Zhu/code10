from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, davies_bouldin_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import PROFILE_FEATURES


@dataclass
class ClusterArtifacts:
    clustered_df: pd.DataFrame
    cluster_summary: pd.DataFrame
    k_metrics: pd.DataFrame
    auto_label_metrics: Dict[str, float]
    profile_feature_columns: List[str]


def _assign_cluster_names(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    top_purchase_idx = summary["purchase_ratio"].idxmax()
    top_value_idx = summary["total_spend"].idxmax()
    top_interaction_idx = summary["interaction_rate"].idxmax()
    top_discount_idx = summary["discount_rate"].idxmax()

    assigned = []
    used = set()
    for idx, row in summary.iterrows():
        if idx == top_purchase_idx and idx == top_value_idx:
            name = "高价值转化"
        elif idx == top_interaction_idx and row["purchase_ratio"] <= summary["purchase_ratio"].median():
            name = "高互动待转化"
        elif idx == top_discount_idx:
            name = "价格敏感"
        else:
            name = "稳态常规"

        if name in used:
            name = "{0}-{1}".format(name, int(row["cluster_id"]))
        used.add(name)
        assigned.append(name)

    summary["cluster_name"] = assigned
    return summary


def fit_user_segments(
    user_df: pd.DataFrame,
    k_values: Iterable[int] = range(2, 7),
    random_state: int = 42,
    metric_sample_size: int = 5000,
) -> ClusterArtifacts:
    work_df = user_df.copy()
    feature_df = work_df[PROFILE_FEATURES].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)

    metric_rows = []
    best_score = -1.0
    best_model = None
    best_k = None
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(scaled)
        if len(work_df) > metric_sample_size:
            sample_df = work_df.sample(metric_sample_size, random_state=random_state)
            sample_scaled = scaler.transform(sample_df[PROFILE_FEATURES])
            sample_labels = model.predict(sample_scaled)
            sil = silhouette_score(sample_scaled, sample_labels)
            dbi = davies_bouldin_score(sample_scaled, sample_labels)
        else:
            sil = silhouette_score(scaled, labels)
            dbi = davies_bouldin_score(scaled, labels)
        metric_rows.append({"k": k, "silhouette": sil, "davies_bouldin": dbi})
        if sil > best_score:
            best_score = sil
            best_model = model
            best_k = k

    if best_model is None:
        raise ValueError("No cluster model was selected.")

    work_df["cluster_id"] = best_model.predict(scaled)
    pca = PCA(n_components=2, random_state=random_state)
    reduced = pca.fit_transform(scaled)
    work_df["pca_x"] = reduced[:, 0]
    work_df["pca_y"] = reduced[:, 1]

    summary = (
        work_df.groupby("cluster_id", as_index=False)[
            [
                "purchase_freq",
                "total_spend",
                "register_days",
                "discount_rate",
                "interaction_rate",
                "purchase_intent",
                "label",
            ]
        ]
        .mean()
        .rename(columns={"label": "purchase_ratio"})
    )
    counts = work_df.groupby("cluster_id").size().rename("user_count").reset_index()
    summary = summary.merge(counts, on="cluster_id", how="left")
    summary = _assign_cluster_names(summary)
    work_df = work_df.merge(summary[["cluster_id", "cluster_name"]], on="cluster_id", how="left")

    x_train, x_test, y_train, y_test = train_test_split(
        feature_df,
        work_df["cluster_id"],
        test_size=0.25,
        stratify=work_df["cluster_id"],
        random_state=random_state,
    )
    annotation_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, multi_class="auto")),
        ]
    )
    annotation_model.fit(x_train, y_train)
    y_pred = annotation_model.predict(x_test)
    auto_label_metrics = {
        "best_k": float(best_k),
        "annotation_accuracy": float(accuracy_score(y_test, y_pred)),
        "annotation_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    return ClusterArtifacts(
        clustered_df=work_df,
        cluster_summary=summary.sort_values("cluster_id").reset_index(drop=True),
        k_metrics=pd.DataFrame(metric_rows),
        auto_label_metrics=auto_label_metrics,
        profile_feature_columns=PROFILE_FEATURES,
    )

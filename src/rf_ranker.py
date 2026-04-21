from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import ITEM_CATEGORICAL_FEATURES, ITEM_NUMERIC_FEATURES, PROFILE_FEATURES


@dataclass
class RankerArtifacts:
    pipeline: Pipeline
    feature_names: List[str]


class RandomForestRanker:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        ordered_columns = PROFILE_FEATURES + ITEM_NUMERIC_FEATURES + ITEM_CATEGORICAL_FEATURES + [
            "coupon_received",
            "coupon_used",
            "pv_count",
            "last_click_gap",
            "interaction_rate",
            "purchase_intent",
            "freshness_score",
            "social_influence",
            "add2cart",
            "baseline_score",
            "cluster_id",
        ]
        self.feature_columns = []
        for column in ordered_columns:
            if column not in self.feature_columns:
                self.feature_columns.append(column)

    def fit(self, df: pd.DataFrame) -> RankerArtifacts:
        numeric_columns = [
            column for column in self.feature_columns if column not in ITEM_CATEGORICAL_FEATURES
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ITEM_CATEGORICAL_FEATURES),
            ]
        )
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            random_state=self.random_state,
        )
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(df[self.feature_columns], df["label"])

        pre = pipeline.named_steps["preprocess"]
        cat_transformer = pre.named_transformers_["cat"]
        if hasattr(cat_transformer, "get_feature_names_out"):
            cat_names = cat_transformer.get_feature_names_out(ITEM_CATEGORICAL_FEATURES).tolist()
        else:
            cat_names = cat_transformer.get_feature_names(ITEM_CATEGORICAL_FEATURES).tolist()
        feature_names = numeric_columns + cat_names
        return RankerArtifacts(pipeline=pipeline, feature_names=feature_names)

    def predict_scores(self, artifacts: RankerArtifacts, df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            artifacts.pipeline.predict_proba(df[self.feature_columns])[:, 1],
            index=df.index,
            name="rf_score",
        )

    @staticmethod
    def feature_importance(artifacts: RankerArtifacts) -> pd.DataFrame:
        importance_df = pd.DataFrame(
            {
                "feature": artifacts.feature_names,
                "importance": artifacts.pipeline.named_steps["model"].feature_importances_,
            }
        )
        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

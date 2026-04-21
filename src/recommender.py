from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

from .data_loader import ITEM_CATEGORICAL_FEATURES, ITEM_NUMERIC_FEATURES, PROFILE_FEATURES


@dataclass
class ThresholdResult:
    threshold: float
    best_f1: float


class SimilarityRecommender:
    def __init__(self, top_k: int = 15, user_weight: float = 0.6) -> None:
        self.top_k = top_k
        self.user_weight = user_weight
        self.item_weight = 1.0 - user_weight
        self._train_df: Optional[pd.DataFrame] = None
        self._user_columns: List[str] = []
        self._item_columns: List[str] = []
        self._user_means: Optional[pd.Series] = None
        self._user_stds: Optional[pd.Series] = None
        self._item_means: Optional[pd.Series] = None
        self._item_stds: Optional[pd.Series] = None
        self._train_matrix: Optional[np.ndarray] = None
        self._nn_index: Optional[NearestNeighbors] = None
        self._global_mean: float = 0.5

    def fit(self, df: pd.DataFrame) -> "SimilarityRecommender":
        self._train_df = df.reset_index(drop=True).copy()
        self._global_mean = float(self._train_df["label"].mean())

        user_matrix, self._user_columns = self._design_matrix(self._train_df, PROFILE_FEATURES, [])
        item_matrix, self._item_columns = self._design_matrix(
            self._train_df, ITEM_NUMERIC_FEATURES, ITEM_CATEGORICAL_FEATURES
        )

        self._user_means = user_matrix.mean(axis=0)
        self._user_stds = user_matrix.std(axis=0).replace(0, 1)
        self._item_means = item_matrix.mean(axis=0)
        self._item_stds = item_matrix.std(axis=0).replace(0, 1)

        train_user = self._standardize(user_matrix, self._user_means, self._user_stds)
        train_item = self._standardize(item_matrix, self._item_means, self._item_stds)
        self._train_matrix = self._combine_features(train_user, train_item)
        self._nn_index = NearestNeighbors(
            n_neighbors=min(self.top_k + 2, len(self._train_df)),
            metric="cosine",
            algorithm="brute",
        )
        self._nn_index.fit(self._train_matrix)
        return self

    def predict_scores(self, df: pd.DataFrame, exclude_self: bool = False) -> np.ndarray:
        if self._train_df is None or self._nn_index is None or self._train_matrix is None:
            raise ValueError("SimilarityRecommender must be fitted before predicting.")

        user_matrix, _ = self._design_matrix(df, PROFILE_FEATURES, [], self._user_columns)
        item_matrix, _ = self._design_matrix(
            df, ITEM_NUMERIC_FEATURES, ITEM_CATEGORICAL_FEATURES, self._item_columns
        )
        query_user = self._standardize(user_matrix, self._user_means, self._user_stds)
        query_item = self._standardize(item_matrix, self._item_means, self._item_stds)
        query_matrix = self._combine_features(query_user, query_item)
        distances, indices = self._nn_index.kneighbors(query_matrix)

        scores = []
        train_labels = self._train_df["label"].to_numpy(dtype=float)
        for row_idx in range(len(df)):
            neighbor_indices = indices[row_idx]
            neighbor_scores = 1.0 - distances[row_idx]
            filtered_indices = []
            filtered_scores = []
            for neighbor_idx, similarity in zip(neighbor_indices, neighbor_scores):
                if exclude_self and len(df) == len(self._train_df):
                    same_user = df.iloc[row_idx]["user_id"] == self._train_df.iloc[neighbor_idx]["user_id"]
                    same_item = df.iloc[row_idx]["item_id"] == self._train_df.iloc[neighbor_idx]["item_id"]
                    if same_user and same_item:
                        continue
                filtered_indices.append(neighbor_idx)
                filtered_scores.append(max(float(similarity), 0.0))
                if len(filtered_indices) >= self.top_k:
                    break

            if not filtered_scores or sum(filtered_scores) == 0:
                scores.append(self._global_mean)
            else:
                scores.append(float(np.average(train_labels[filtered_indices], weights=filtered_scores)))
        return np.array(scores, dtype=float)

    @staticmethod
    def tune_threshold(labels: pd.Series, scores: np.ndarray) -> ThresholdResult:
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.linspace(0.2, 0.8, 25):
            predictions = (scores >= threshold).astype(int)
            score = f1_score(labels, predictions)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        return ThresholdResult(threshold=best_threshold, best_f1=best_f1)

    @staticmethod
    def _design_matrix(
        df: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        expected_columns: Optional[List[str]] = None,
    ) -> tuple[pd.DataFrame, List[str]]:
        work_df = df[numeric_columns + categorical_columns].copy()
        if categorical_columns:
            work_df = pd.get_dummies(work_df, columns=categorical_columns, dtype=float)
        if expected_columns is not None:
            work_df = work_df.reindex(columns=expected_columns, fill_value=0.0)
            return work_df, expected_columns
        return work_df, work_df.columns.tolist()

    @staticmethod
    def _standardize(matrix: pd.DataFrame, means: pd.Series, stds: pd.Series) -> np.ndarray:
        standardized = (matrix - means) / stds
        return standardized.to_numpy(dtype=float)

    def _combine_features(self, user_matrix: np.ndarray, item_matrix: np.ndarray) -> np.ndarray:
        user_part = user_matrix * np.sqrt(max(self.user_weight, 1e-8))
        item_part = item_matrix * np.sqrt(max(self.item_weight, 1e-8))
        return np.hstack([user_part, item_part])

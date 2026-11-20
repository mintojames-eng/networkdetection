"""
Feature Selection Engine for CIC-IDS2017
Reduces high-dimensional flow features (67+) to the top most discriminative,
low-latency features using Tree Importance and Collinear Filtering.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class CICFeatureSelector:
    def __init__(self, top_k=10, correlation_threshold=0.92, random_state=42):
        self.top_k = top_k
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.selected_features_ = None
        self.feature_importances_ = None

    def fit(self, X_df: pd.DataFrame, y: np.ndarray):
        """
        1. Filters out highly collinear features (Pearson correlation > threshold).
        2. Fits a Random Forest to rank remaining features by Gini importance.
        3. Selects the top_k most discriminative features.
        """
        # Step 1: Remove Collinear Features
        corr_matrix = X_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.correlation_threshold)]
        
        reduced_X = X_df.drop(columns=to_drop)
        
        # Step 2: Random Forest Feature Importance Ranking
        rf = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=self.random_state, n_jobs=-1)
        rf.fit(reduced_X, y)
        
        importances = pd.Series(rf.feature_importances_, index=reduced_X.columns).sort_values(ascending=False)
        self.feature_importances_ = importances
        
        # Step 3: Pick Top K
        self.selected_features_ = list(importances.head(self.top_k).index)
        return self

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Projects dataframe onto selected top-K features."""
        return X_df[self.selected_features_]

    def fit_transform(self, X_df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit(X_df, y)
        return self.transform(X_df)

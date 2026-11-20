"""
Data Preprocessing & Hierarchical Label Encoding
Prepares telemetry features and partitions into Level 1 (Binary) and Level 2 (Multi-Class).
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

class NIDSPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.level2_label_encoder = LabelEncoder()
        self.feature_names = None
        self.level2_classes_ = None

    def fit_transform(self, df: pd.DataFrame, test_size=0.25, random_state=42):
        self.feature_names = [col for col in df.columns if col != "label"]
        X = np.asarray(df[self.feature_names].values, dtype=np.float64)
        y_raw = np.array(df["label"].tolist(), dtype=str)
        
        # Level 1: Binary (0 = Normal/BENIGN, 1 = Attack)
        is_benign = np.isin(np.char.upper(y_raw.astype(str)), ["NORMAL", "BENIGN"])
        y_l1 = (~is_benign).astype(int)
        
        # Split train & test stratified on raw classes so distribution is preserved
        (
            X_train, X_test,
            y_l1_train, y_l1_test,
            y_raw_train, y_raw_test
        ) = train_test_split(
            X, y_l1, y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw
        )
        
        # Fit scaler on training features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Level 2: Filter out "Normal/BENIGN", train only on attack categories
        attack_train_mask = ~np.isin(np.char.upper(y_raw_train.astype(str)), ["NORMAL", "BENIGN"])
        X_l2_train = X_train_scaled[attack_train_mask]
        y_l2_train_raw = y_raw_train[attack_train_mask]
        
        # Fit Level 2 label encoder (e.g. ['BruteForce', 'DDoS', 'DoS', 'Probe'])
        y_l2_train = self.level2_label_encoder.fit_transform(y_l2_train_raw)
        self.level2_classes_ = list(self.level2_label_encoder.classes_)
        
        # Level 2 test sets (for attacks only)
        attack_test_mask = ~np.isin(np.char.upper(y_raw_test.astype(str)), ["NORMAL", "BENIGN"])
        X_l2_test = X_test_scaled[attack_test_mask]
        y_l2_test_raw = y_raw_test[attack_test_mask]
        y_l2_test = self.level2_label_encoder.transform(y_l2_test_raw)
        
        return {
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_l1_train": y_l1_train,
            "y_l1_test": y_l1_test,
            "y_raw_train": y_raw_train,
            "y_raw_test": y_raw_test,
            "X_l2_train": X_l2_train,
            "y_l2_train": y_l2_train,
            "X_l2_test": X_l2_test,
            "y_l2_test": y_l2_test
        }

    def transform_single(self, feature_vector):
        """Preprocesses a single incoming flow vector for inference."""
        arr = np.array(feature_vector, dtype=float).reshape(1, -1)
        return self.scaler.transform(arr)

    def decode_l2(self, class_idx):
        """Converts integer class back to attack name."""
        return self.level2_label_encoder.inverse_transform([class_idx])[0]

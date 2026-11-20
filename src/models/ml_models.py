"""
Traditional Machine Learning Models for Network Intrusion Detection:
1. Logistic Regression (Baseline)
2. Random Forest (Tabular workhorse)
3. Support Vector Machine (SVM)
4. XGBoost / Gradient Boosting (State-of-the-art tabular ensemble)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

# Check if xgboost is installed, otherwise fallback to HistGradientBoosting (same tree-boosting algorithm)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def get_traditional_models(random_state=42):
    """
    Returns a dictionary of configured traditional machine learning models.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "SVM (RBF Kernel)": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        ) if HAS_XGBOOST else HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )
    }
    return models

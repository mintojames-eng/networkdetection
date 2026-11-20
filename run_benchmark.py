"""
Master Execution Pipeline: AI Network Intrusion Detection
Directly executes the complete 8-stage architecture:
1. CIC-IDS2017 Ingestion
2. Data Preprocessing & Scaling
3. Feature Selection & Collinear Elimination
4. Traditional ML Training (Logistic Regression + Random Forest + XGBoost)
5. Deep Learning Training (ANN + 1D-CNN)
6. Model Comparison & Metrics Leaderboard
7. SHAP Explainability (Local & Global Attribution)
8. Streamlit Security Dashboard Integration
"""

import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.cic_ids2017 import load_or_generate_cic_ids2017
from src.preprocessing import NIDSPreprocessor
from src.feature_selection import CICFeatureSelector
from src.models.ml_models import get_traditional_models
from src.models.ann_model import get_ann_model
from src.models.cnn_1d import get_1dcnn_model
from src.explainability import NIDSShapExplainer

def run_master_pipeline():
    print("=" * 95)
    print("  AI-BASED NETWORK INTRUSION DETECTION & ATTACK CLASSIFICATION PIPELINE")
    print("  Workflow: CIC-IDS2017 -> Preprocessing -> Feature Selection -> ML -> DL -> Comparison -> SHAP")
    print("=" * 95)
    
    # -------------------------------------------------------------
    # STAGE 1: CIC-IDS2017 Ingestion
    # -------------------------------------------------------------
    print("\n[STAGE 1/8] Ingesting CIC-IDS2017 Dataset...")
    raw_df = load_or_generate_cic_ids2017(n_samples=8000, random_state=42)
    feature_cols = [c for c in raw_df.columns if c != "Label"]
    y_raw = raw_df["Label"].values
    y_binary = (y_raw != "BENIGN").astype(int)
    print(f"            Loaded {len(raw_df):,} flows across {len(feature_cols)} raw flow features.")
    print(f"            Class Distribution: {pd.Series(y_raw).value_counts().to_dict()}")

    # -------------------------------------------------------------
    # STAGE 2 & 3: Preprocessing & Feature Selection
    # -------------------------------------------------------------
    print("\n[STAGE 2/8 & 3/8] Running Feature Selection & Preprocessing...")
    selector = CICFeatureSelector(top_k=8, correlation_threshold=0.90, random_state=42)
    selector.fit(raw_df[feature_cols], y_binary)
    
    selected_features = selector.selected_features_
    print(f"            Selected Top {len(selected_features)} Discriminative Flow Features:")
    for idx, f_name in enumerate(selected_features, 1):
        score = selector.feature_importances_.get(f_name, 0.0)
        print(f"              {idx}. {f_name:<30} (Gini Importance: {score:.4f})")
        
    selected_df = selector.transform(raw_df[feature_cols]).copy()
    selected_df["label"] = y_raw
    
    preprocessor = NIDSPreprocessor()
    data = preprocessor.fit_transform(selected_df, test_size=0.25, random_state=42)
    X_train = data["X_train_scaled"]
    y_train = data["y_l1_train"]
    X_test = data["X_test_scaled"]
    y_test = data["y_l1_test"]
    print(f"            Standardized Telemetry: {len(X_train)} Train Flows | {len(X_test)} Test Flows")

    # -------------------------------------------------------------
    # STAGE 4 & 5: Model Training (ML + DL)
    # -------------------------------------------------------------
    print("\n[STAGE 4/8 & 5/8] Training ML (LR, RF, XGB) & DL (ANN, 1D-CNN) Models...")
    models = get_traditional_models(random_state=42)
    # Filter to the requested models
    requested_ml = {k: v for k, v in models.items() if any(m in k for m in ["Logistic", "Random Forest", "XGBoost"])}
    
    # Add Deep Learning Models
    requested_ml["ANN (MLP: 64->32)"] = get_ann_model(random_state=42)
    requested_ml["1D-CNN (Conv1D)"] = get_1dcnn_model(in_features=X_train.shape[1], num_classes=2, random_state=42)

    benchmark_records = []
    trained_models = {}

    for name, model in requested_ml.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t_infer = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t_infer
        latency_us = (infer_time / len(X_test)) * 1_000_000
        
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, zero_division=0) * 100
        rec = recall_score(y_test, y_pred, zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, zero_division=0) * 100
        
        benchmark_records.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "train_time_sec": train_time,
            "latency_us": latency_us
        })
        trained_models[name] = model

    # -------------------------------------------------------------
    # STAGE 6: Model Comparison Leaderboard
    # -------------------------------------------------------------
    print("\n[STAGE 6/8] MODEL COMPARISON LEADERBOARD")
    print("=" * 95)
    header = f"{'Model':<26} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Latency':<9}"
    print(header)
    print("=" * 95)
    for r in benchmark_records:
        print(f"{r['model']:<26} | {r['accuracy']:>7.2f}% | {r['precision']:>7.2f}% | {r['recall']:>7.2f}% | {r['f1_score']:>7.2f}% | {r['latency_us']:>6.1f} µs")
    print("=" * 95)

    # -------------------------------------------------------------
    # STAGE 7: SHAP Explainability (XAI)
    # -------------------------------------------------------------
    print("\n[STAGE 7/8] Computing SHAP Explainability for Inbound Flow...")
    champion_model = trained_models.get("Random Forest", list(trained_models.values())[0])
    
    np.random.seed(42)
    bg_sample = X_train[np.random.choice(len(X_train), size=min(100, len(X_train)), replace=False)]
    explainer = NIDSShapExplainer(champion_model, bg_sample, selected_features)
    
    # Explain an attack sample
    attack_indices = np.where(y_test == 1)[0]
    sample_idx = attack_indices[0] if len(attack_indices) > 0 else 0
    test_flow = X_test[sample_idx]
    
    exp = explainer.explain_flow(test_flow)
    print(f"            Base Attack Expectation (Prior): {exp['base_value']:.4f}")
    print(f"            Model Output Prediction:        {exp['prediction']:.4f}")
    print("\n            Local SHAP Attribution Breakdown:")
    for fname, val in sorted(zip(exp['feature_names'], exp['shap_values']), key=lambda x: abs(x[1]), reverse=True):
        sign = "+" if val > 0 else "-"
        direction = "Push toward ATTACK" if val > 0 else "Push toward NORMAL"
        print(f"              {sign} {fname:<30}: {val:>+7.4f}  ({direction})")

    # -------------------------------------------------------------
    # STAGE 8: Dashboard Notification
    # -------------------------------------------------------------
    print("\n[STAGE 8/8] STREAMLIT SECURITY DASHBOARD READY")
    print("            Launch Command: streamlit run app.py")
    print("=" * 95)

if __name__ == "__main__":
    run_master_pipeline()

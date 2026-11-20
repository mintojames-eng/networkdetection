"""
SHAP (SHapley Additive exPlanations) & XAI Module for Network Intrusion Detection
Provides both Global Explainability (macro dataset patterns) and Local Explainability
(explaining why a specific flow #1254 or #1255 was classified as Benign or Malicious).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    HAS_OFFICIAL_SHAP = True
except ImportError:
    HAS_OFFICIAL_SHAP = False

class NIDSShapExplainer:
    def __init__(self, model, background_data, feature_names):
        """
        Args:
            model: Trained classifier (e.g. Random Forest or XGBoost)
            background_data: Representative background sample (e.g., 100 scaled flows)
            feature_names: List of feature names
        """
        self.model = model
        self.background_data = np.asarray(background_data)
        self.feature_names = feature_names
        
        # Calculate base expected value E[f(x)]
        if hasattr(self.model, "predict_proba"):
            bg_probs = self.model.predict_proba(self.background_data)
            bg_preds = bg_probs[:, 1] if bg_probs.shape[1] > 1 else bg_probs[:, 0]
        else:
            bg_preds = self.model.predict(self.background_data)
        self.expected_value = float(np.mean(bg_preds))

    def explain_flow(self, flow_vector, n_samples=150):
        """
        Calculates local Shapley values for an individual incoming flow vector.
        Decomposes the prediction into baseline + sum of feature attributions:
            f(x) = E[f(x)] + sum(phi_i)
        """
        flow = np.asarray(flow_vector).flatten()
        num_features = len(flow)
        
        # If official shap package is installed
        if HAS_OFFICIAL_SHAP:
            explainer = shap.TreeExplainer(self.model, self.background_data[:50])
            shap_values = explainer.shap_values(flow.reshape(1, -1))
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]
            return {
                "base_value": self.expected_value,
                "feature_names": self.feature_names,
                "feature_values": flow,
                "shap_values": vals
            }

        # High-performance Monte-Carlo Sampling Shapley Algorithm
        # Approximates marginal contributions across feature subsets
        np.random.seed(42)
        shap_values = np.zeros(num_features)
        
        # Predict on actual instance
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(flow.reshape(1, -1))
            inst_pred = p[0, 1] if p.shape[1] > 1 else p[0, 0]
        else:
            inst_pred = float(self.model.predict(flow.reshape(1, -1))[0])
            
        total_diff = inst_pred - self.expected_value
        
        # Sample permutations to compute marginal contributions
        for _ in range(n_samples):
            # Pick a random background instance
            bg_idx = np.random.randint(0, len(self.background_data))
            bg_inst = self.background_data[bg_idx].copy()
            
            # Random feature permutation
            perm = np.random.permutation(num_features)
            
            prev_pred = self.expected_value
            current_hybrid = bg_inst.copy()
            
            for feat_idx in perm:
                current_hybrid[feat_idx] = flow[feat_idx]
                if hasattr(self.model, "predict_proba"):
                    np_p = self.model.predict_proba(current_hybrid.reshape(1, -1))
                    new_pred = np_p[0, 1] if np_p.shape[1] > 1 else np_p[0, 0]
                else:
                    new_pred = float(self.model.predict(current_hybrid.reshape(1, -1))[0])
                marginal_contribution = new_pred - prev_pred
                shap_values[feat_idx] += marginal_contribution
                prev_pred = new_pred
                
        shap_values /= n_samples
        
        # Normalize sum of shap values to match exact efficiency property: sum(phi_i) == f(x) - E[f(x)]
        if np.sum(np.abs(shap_values)) > 1e-6:
            scaling_factor = total_diff / np.sum(shap_values)
            shap_values *= scaling_factor
            
        return {
            "base_value": self.expected_value,
            "prediction": inst_pred,
            "feature_names": self.feature_names,
            "feature_values": flow,
            "shap_values": shap_values
        }

    def plot_waterfall(self, explanation):
        """Generates a clean horizontal bar chart showing positive/negative SHAP pushes."""
        names = explanation["feature_names"]
        vals = explanation["shap_values"]
        
        # Sort by absolute impact
        sorted_indices = np.argsort(np.abs(vals))[::-1][:8]
        top_names = [names[i] for i in sorted_indices][::-1]
        top_vals = [vals[i] for i in sorted_indices][::-1]
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_vals]
        ax.barh(top_names, top_vals, color=colors)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("SHAP Value (Impact on Attack Probability)")
        ax.set_title(f"SHAP Flow Explanation (Base Value: {explanation['base_value']:.2f})")
        plt.tight_layout()
        return fig

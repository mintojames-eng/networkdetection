"""
Two-Level Hierarchical Intrusion Detection Pipeline
Level 1: High-Speed Binary Filter (Normal vs Attack)
Level 2: Fine-Grained Attack Classifier (DoS, DDoS, Probe, BruteForce) -> Alert Engine
"""

import time
from .alert_engine import AlertEngine

class TwoLevelNIDS:
    def __init__(self, level1_model, level2_model, preprocessor, l1_threshold=0.35, alert_engine=None):
        """
        Args:
            level1_model: Trained binary classifier (Normal vs Attack)
            level2_model: Trained multi-class attack classifier
            preprocessor: NIDSPreprocessor instance
            l1_threshold: Safety cutoff probability for flagging an attack (default 0.35 for high recall)
            alert_engine: AlertEngine instance
        """
        self.level1_model = level1_model
        self.level2_model = level2_model
        self.preprocessor = preprocessor
        self.l1_threshold = l1_threshold
        self.alert_engine = alert_engine or AlertEngine()

    def inspect(self, traffic_id, raw_features, telemetry=None):
        """
        Executes hierarchical inspection on a single network flow.
        
        Flow:
            Raw Traffic 
                 ↓
            Level 1 (Binary Filter)
                 ├── Normal → Log & Allow
                 └── Attack → Level 2 (Classifier) → Identify Attack → Alert
        """
        telemetry = telemetry or {"src_ip": "192.168.1.100", "dst_ip": "10.0.0.1", "dst_port": 80}
        start_time = time.perf_counter()
        
        # 1. Scale feature vector
        scaled_x = self.preprocessor.transform_single(raw_features)
        
        # 2. Level 1 Inspection (Binary)
        if hasattr(self.level1_model, "predict_proba"):
            attack_prob = self.level1_model.predict_proba(scaled_x)[0][1]
        else:
            attack_prob = float(self.level1_model.predict(scaled_x)[0])
            
        is_attack = attack_prob >= self.l1_threshold
        
        if not is_attack:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return self.alert_engine.generate_normal_log(traffic_id, elapsed_ms, telemetry)
            
        # 3. Level 2 Inspection (Multi-Class Attack Diagnostic)
        l2_raw_pred = self.level2_model.predict(scaled_x)[0]
        
        # Handle string or integer encoded outputs
        if isinstance(l2_raw_pred, (int, np.integer)):
            attack_type = self.preprocessor.decode_l2(l2_raw_pred)
        else:
            attack_type = str(l2_raw_pred)
            
        # Extract prediction confidence
        confidence_pct = 95.0
        if hasattr(self.level2_model, "predict_proba"):
            probs = self.level2_model.predict_proba(scaled_x)[0]
            confidence_pct = float(np.max(probs)) * 100
            
        total_elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # 4. Generate Security Alert
        return self.alert_engine.generate_attack_alert(
            traffic_id=traffic_id,
            attack_type=attack_type,
            confidence_pct=confidence_pct,
            latency_ms=total_elapsed_ms,
            telemetry=telemetry
        )

import numpy as np

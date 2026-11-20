"""
Simulation Script: Real-Time Network Traffic Inspection (Two-Level NIDS)

Demonstrates the exact two-level workflow:
Traffic #1254 -> Level 1 -> Normal -> Allowed & Logged
Traffic #1255 -> Level 1 -> Malicious -> Level 2 -> DDoS -> Critical Alert + Mitigation
Traffic #1256 -> Level 1 -> Malicious -> Level 2 -> Probe -> PortScan Alert
Traffic #1257 -> Level 1 -> Malicious -> Level 2 -> BruteForce -> SSH Ban Alert
"""

import json
from data.dataset import generate_network_flows
from src.preprocessing import NIDSPreprocessor
from src.pipeline import TwoLevelNIDS
from src.alert_engine import AlertEngine
from src.models.ml_models import RandomForestClassifier
from src.models.cnn_1d import CNN1DClassifier

def main():
    print("=" * 80)
    print("      INITIALIZING REAL-TIME TWO-LEVEL INTRUSION DETECTION SYSTEM")
    print("=" * 80)
    
    # 1. Prepare Data & Preprocessor
    print("[*] Training Level 1 and Level 2 Models on network telemetry...")
    df = generate_network_flows(n_samples=8000, random_state=42)
    preprocessor = NIDSPreprocessor()
    data = preprocessor.fit_transform(df, test_size=0.2, random_state=42)
    
    # 2. Train Level 1: High-Speed Binary Champion (Random Forest / Gradient Boosting)
    level1_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    level1_model.fit(data["X_train_scaled"], data["y_l1_train"])
    print("    [+] Level 1 (Binary Filter) trained successfully.")
    
    # 3. Train Level 2: Multi-Class Attack Diagnostic (1D-CNN)
    level2_model = CNN1DClassifier(in_features=data["X_l2_train"].shape[1], num_classes=len(data["y_l2_train"]), random_state=42)
    level2_model.fit(data["X_l2_train"], data["y_l2_train"])
    print("    [+] Level 2 (1D-CNN Attack Classifier) trained successfully.")
    
    # 4. Instantiate Hierarchical Pipeline
    nids = TwoLevelNIDS(
        level1_model=level1_model,
        level2_model=level2_model,
        preprocessor=preprocessor,
        l1_threshold=0.35,  # High sensitivity cutoff for attacks
        alert_engine=AlertEngine(log_to_file=True, log_path="traffic_audit.log")
    )
    print("    [+] Two-Level NIDS Pipeline online. Ready for packet inspection.\n")
    
    # 5. Simulate Stream of Real Network Traffic
    # Features: [flow_duration, tot_fwd_pkts, tot_bwd_pkts, flow_byts_s, syn_flag_cnt, rst_flag_cnt, pkt_len_mean, dst_port_entropy]
    simulated_traffic = [
        {
            "id": "#1254",
            "features": [14.5, 12, 16, 2200.0, 0, 0, 520.0, 0.15],
            "telemetry": {"src_ip": "192.168.1.45", "dst_ip": "142.250.190.46", "dst_port": 443, "protocol": "HTTPS"}
        },
        {
            "id": "#1255",
            "features": [1.2, 1450, 0, 280000.0, 1200, 0, 128.0, 0.02],
            "telemetry": {"src_ip": "45.134.20.12", "dst_ip": "10.0.0.5", "dst_port": 80, "protocol": "HTTP-FLOOD"}
        },
        {
            "id": "#1256",
            "features": [24.0, 92, 4, 620.0, 80, 18, 48.0, 0.96],
            "telemetry": {"src_ip": "185.220.101.5", "dst_ip": "10.0.0.5", "dst_port": 22, "protocol": "TCP-SYN"}
        },
        {
            "id": "#1257",
            "features": [48.0, 210, 195, 4500.0, 18, 14, 96.0, 0.02],
            "telemetry": {"src_ip": "103.21.244.2", "dst_ip": "10.0.0.5", "dst_port": 22, "protocol": "SSH"}
        }
    ]
    
    print("=" * 80)
    print("                      STREAMING NETWORK FLOW AUDIT")
    print("=" * 80)
    
    for packet in simulated_traffic:
        t_id = packet["id"]
        features = packet["features"]
        telemetry = packet["telemetry"]
        
        result = nids.inspect(t_id, features, telemetry)
        
        if result["status"] == "NORMAL":
            print(f"\n[OK] Traffic {t_id} -> NORMAL (Allowed) in {result['inspection_latency_ms']} ms")
            print(f"     Src: {telemetry['src_ip']} -> Dst: {telemetry['dst_ip']}:{telemetry['dst_port']} ({telemetry['protocol']})")
        else:
            print(f"\n[!] ALERT: Traffic {t_id} -> {result['level_2_attack_type'].upper()} ({result['severity']}) in {result['inspection_latency_ms']} ms")
            print(f"     Confidence: {result['confidence']}")
            print(f"     Src IP: {telemetry['src_ip']} -> Dst Port: {telemetry['dst_port']}")
            print(f"     Automated Action: {result['automated_mitigation']['firewall_command']}")
            
        print("-" * 80)

if __name__ == "__main__":
    main()

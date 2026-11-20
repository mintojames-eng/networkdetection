"""
Dataset Generator & Loader for Network Intrusion Detection
Simulates realistic NetFlow / IPFIX flow features matching CIC-IDS2017 & NSL-KDD benchmarks.
"""

import numpy as np
import pandas as pd

def generate_network_flows(n_samples=12000, random_state=42):
    """
    Generates realistic network flow telemetry with statistical distributions
    tailored to benign traffic and 4 major attack families:
      - Normal: balanced packet lengths, steady flow rates, low SYN/RST ratios.
      - DoS (Denial of Service): high volume SYN/ICMP from a single source, short flow duration.
      - DDoS (Distributed DoS): massive packet bursts, extreme byte rates, zero backward responses.
      - Probe (Port Scanning / Recon): high destination port entropy, low payload bytes, repeated SYN.
      - BruteForce (SSH/FTP): persistent connection duration, repeated identical small packets.
    """
    np.random.seed(random_state)
    
    classes = ["Normal", "DoS", "DDoS", "Probe", "BruteForce"]
    class_probs = [0.70, 0.12, 0.08, 0.07, 0.03]
    labels = np.random.choice(classes, size=n_samples, p=class_probs)
    
    records = []
    for label in labels:
        if label == "Normal":
            duration = np.random.exponential(12.0)
            fwd_pkts = int(np.random.poisson(15)) + 1
            bwd_pkts = int(np.random.poisson(18)) + 1
            byte_rate = np.random.exponential(2500.0)
            syn_cnt = np.random.binomial(1, 0.08)
            rst_cnt = np.random.binomial(1, 0.02)
            pkt_len_mean = np.random.normal(512.0, 120.0)
            port_entropy = np.random.uniform(0.05, 0.25)
            
        elif label == "DoS":
            duration = np.random.exponential(3.0)
            fwd_pkts = int(np.random.poisson(450)) + 50
            bwd_pkts = int(np.random.poisson(2))
            byte_rate = np.random.exponential(95000.0)
            syn_cnt = int(np.random.poisson(400))
            rst_cnt = np.random.binomial(1, 0.05)
            pkt_len_mean = np.random.normal(64.0, 10.0)
            port_entropy = np.random.uniform(0.01, 0.08)
            
        elif label == "DDoS":
            duration = np.random.exponential(1.5)
            fwd_pkts = int(np.random.poisson(1200)) + 100
            bwd_pkts = 0
            byte_rate = np.random.exponential(250000.0)
            syn_cnt = int(np.random.poisson(1000))
            rst_cnt = 0
            pkt_len_mean = np.random.normal(128.0, 20.0)
            port_entropy = np.random.uniform(0.01, 0.05)
            
        elif label == "Probe":
            duration = np.random.exponential(25.0)
            fwd_pkts = int(np.random.poisson(85)) + 10
            bwd_pkts = int(np.random.poisson(5))
            byte_rate = np.random.exponential(600.0)
            syn_cnt = int(np.random.poisson(75))
            rst_cnt = int(np.random.poisson(20))
            pkt_len_mean = np.random.normal(48.0, 8.0)
            port_entropy = np.random.uniform(0.75, 0.99)  # Scanning many ports
            
        elif label == "BruteForce":
            duration = np.random.exponential(45.0)
            fwd_pkts = int(np.random.poisson(180)) + 20
            bwd_pkts = int(np.random.poisson(170)) + 20
            byte_rate = np.random.exponential(4200.0)
            syn_cnt = int(np.random.poisson(15))
            rst_cnt = int(np.random.poisson(12))
            pkt_len_mean = np.random.normal(95.0, 15.0)
            port_entropy = np.random.uniform(0.01, 0.05)
            
        # Introduce realistic telemetry jitter / feature overlap
        jitter = np.random.normal(0, 0.15, size=8)
        feature_vals = np.array([
            max(0.001, duration),
            max(1, fwd_pkts),
            max(0, bwd_pkts),
            max(1.0, byte_rate),
            max(0, syn_cnt),
            max(0, rst_cnt),
            max(20.0, pkt_len_mean),
            min(1.0, max(0.0, port_entropy))
        ]) * (1.0 + jitter)
        
        # 3% label noise to simulate real-world packet capture artifacts
        if np.random.rand() < 0.03 and label != "Normal":
            feature_vals[3] *= 0.1  # dampened byte rate
            feature_vals[4] = 0     # missing syn flag
            
        records.append(feature_vals.tolist())
        
    feature_names = [
        "flow_duration",
        "tot_fwd_pkts",
        "tot_bwd_pkts",
        "flow_byts_s",
        "syn_flag_cnt",
        "rst_flag_cnt",
        "pkt_len_mean",
        "dst_port_entropy"
    ]
    
    df = pd.DataFrame(records, columns=feature_names)
    df["label"] = labels
    return df

if __name__ == "__main__":
    df = generate_network_flows(1000)
    print("Dataset generated successfully:")
    print(df["label"].value_counts())
    print("\nSample records:")
    print(df.head())

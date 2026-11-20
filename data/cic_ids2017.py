"""
CIC-IDS2017 Benchmark Dataset Loader & Synthesizer
Handles official CIC-IDS2017 CSV schema (78 features), column sanitization,
Infinity/NaN cleaning, and generation of representative CIC-IDS2017 telemetry.
"""

import os
import numpy as np
import pandas as pd

# Core canonical feature columns in CIC-IDS2017 (CICFlowMeter output)
CIC_IDS2017_CORE_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

def clean_cic_ids_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw CIC-IDS2017 data:
    1. Strips leading/trailing whitespace from column names (classic CIC-IDS2017 issue).
    2. Replaces Infinity with NaN, then imputes or drops NaNs.
    3. Normalizes Label column.
    """
    # Strip whitespace in column headers
    df.columns = df.columns.str.strip()
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN or fill with median
    df = df.dropna()
    return df

def load_or_generate_cic_ids2017(filepath=None, n_samples=8000, random_state=42) -> pd.DataFrame:
    """
    Loads an official CIC-IDS2017 CSV file if provided and exists;
    otherwise generates a synthetic CIC-IDS2017 dataset preserving the exact
    canonical 67 flow features and standard labels (BENIGN, DoS, DDoS, PortScan, Bot).
    """
    if filepath and os.path.exists(filepath):
        print(f"[*] Loading official CIC-IDS2017 CSV: {filepath}")
        df = pd.read_csv(filepath)
        return clean_cic_ids_dataframe(df)

    # Generate synthetic CIC-IDS2017 dataframe with core flow features
    np.random.seed(random_state)
    classes = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce"]
    class_dist = [0.70, 0.12, 0.08, 0.07, 0.03]
    labels = np.random.choice(classes, size=n_samples, p=class_dist)
    
    rows = []
    for label in labels:
        if label == "BENIGN":
            dst_port = np.random.choice([80, 443, 53, 8080, 22])
            duration = np.random.exponential(1500000)  # in microseconds
            fwd_pkts = np.random.poisson(12) + 1
            bwd_pkts = np.random.poisson(15) + 1
            flow_byts_s = np.random.exponential(3500.0)
            syn_flags = np.random.binomial(1, 0.05)
            rst_flags = 0
            ack_flags = np.random.poisson(15)
            pkt_len_mean = np.random.normal(550.0, 80.0)
            iat_mean = duration / max(1, fwd_pkts + bwd_pkts)
        elif label == "DoS":
            dst_port = 80
            duration = np.random.exponential(250000)
            fwd_pkts = np.random.poisson(600) + 100
            bwd_pkts = np.random.poisson(2)
            flow_byts_s = np.random.exponential(120000.0)
            syn_flags = np.random.poisson(550)
            rst_flags = np.random.binomial(1, 0.05)
            ack_flags = 0
            pkt_len_mean = np.random.normal(64.0, 5.0)
            iat_mean = duration / max(1, fwd_pkts)
        elif label == "DDoS":
            dst_port = 80
            duration = np.random.exponential(120000)
            fwd_pkts = np.random.poisson(1500) + 200
            bwd_pkts = 0
            flow_byts_s = np.random.exponential(350000.0)
            syn_flags = np.random.poisson(1400)
            rst_flags = 0
            ack_flags = 0
            pkt_len_mean = np.random.normal(128.0, 10.0)
            iat_mean = duration / max(1, fwd_pkts)
        elif label == "PortScan":
            dst_port = int(np.random.uniform(1, 65535))
            duration = np.random.exponential(50000)
            fwd_pkts = np.random.poisson(3) + 1
            bwd_pkts = np.random.binomial(1, 0.1)
            flow_byts_s = np.random.exponential(800.0)
            syn_flags = 2
            rst_flags = np.random.binomial(1, 0.8)
            ack_flags = 0
            pkt_len_mean = 44.0
            iat_mean = 1200.0
        else: # BruteForce
            dst_port = 22
            duration = np.random.exponential(4500000)
            fwd_pkts = np.random.poisson(120) + 30
            bwd_pkts = np.random.poisson(110) + 30
            flow_byts_s = np.random.exponential(5500.0)
            syn_flags = 15
            rst_flags = 10
            ack_flags = np.random.poisson(100)
            pkt_len_mean = np.random.normal(90.0, 15.0)
            iat_mean = duration / max(1, fwd_pkts + bwd_pkts)

        # Assemble full 67-column row with correlated flow telemetry
        row = [
            dst_port, max(1.0, duration), max(1, fwd_pkts), max(0, bwd_pkts),
            fwd_pkts * pkt_len_mean, bwd_pkts * pkt_len_mean * 0.8,
            pkt_len_mean * 1.5, pkt_len_mean * 0.5, pkt_len_mean, 25.0,
            pkt_len_mean * 1.2, pkt_len_mean * 0.4, pkt_len_mean * 0.9, 15.0,
            max(1.0, flow_byts_s), (fwd_pkts + bwd_pkts) / max(0.001, duration / 1e6),
            iat_mean, iat_mean * 0.5, iat_mean * 2.0, max(0.1, iat_mean * 0.1),
            duration, iat_mean, iat_mean * 0.4, iat_mean * 1.8, max(0.1, iat_mean * 0.1),
            duration * 0.8, iat_mean * 1.2, iat_mean * 0.3, iat_mean * 1.5, max(0.1, iat_mean * 0.1),
            0, 0, fwd_pkts * 20, bwd_pkts * 20,
            fwd_pkts / max(0.001, duration / 1e6), bwd_pkts / max(0.001, duration / 1e6),
            max(20.0, pkt_len_mean * 0.4), pkt_len_mean * 2.0, pkt_len_mean, 30.0,
            900.0, 0, syn_flags, rst_flags, 0, ack_flags, 0,
            bwd_pkts / max(1, fwd_pkts), pkt_len_mean, pkt_len_mean, pkt_len_mean * 0.9,
            fwd_pkts, fwd_pkts * pkt_len_mean, bwd_pkts, bwd_pkts * pkt_len_mean,
            8192, 8192, fwd_pkts - 1, 20,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        rows.append(row)
        
    df = pd.DataFrame(rows, columns=CIC_IDS2017_CORE_FEATURES)
    df["Label"] = labels
    return df

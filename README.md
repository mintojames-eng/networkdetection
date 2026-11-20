# AI-Based Network Intrusion Detection & Attack Classification with SHAP XAI

A two-level hierarchical intrusion detection system spanning:
`CIC-IDS2017` $\rightarrow$ `Preprocessing` $\rightarrow$ `Feature Selection` $\rightarrow$ `ML (LR + RF + XGB)` $\rightarrow$ `DL (ANN + 1D-CNN)` $\rightarrow$ `Model Comparison` $\rightarrow$ `SHAP Explainability` $\rightarrow$ `Streamlit Security Dashboard`

---

## 🏛️ End-to-End Pipeline

```
CIC-IDS2017 (67+ Raw Features)
       ↓
Data Preprocessing (StandardScaler, NaN/Inf Imputation, Stratified Split)
       ↓
Feature Selection (Collinear Filtering + Random Forest Gini Importance -> Top Features)
       ↓
Level 1: Binary Filter (Logistic Regression, Random Forest, XGBoost) -> Normal (Allow) vs Attack
       ↓ (If Attack)
Level 2: Deep Attack Classification (ANN, 1D-CNN) -> DoS, DDoS, PortScan, BruteForce
       ↓
SHAP Explainability (Calculates Shapley values to explain WHY an alert was triggered)
       ↓
Incident Response & Dashboard (Dynamic Severity + iptables / Blackhole Firewall Rules)
```

```
                   Incoming Network Traffic
                             │
                             ▼
               ┌───────────────────────────┐
               │   Level 1: Binary Filter  │
               │  (High-Throughput / Recall)│
               └─────────────┬─────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
         [ Normal ]                    [ Attack ]
      Traffic #1254                 Traffic #1255
      (Allow & Log)                         │
                                            ▼
                              ┌───────────────────────────┐
                              │ Level 2: Attack Classifier│
                              │ (Fine-Grained Diagnosis)  │
                              └─────────────┬─────────────┘
                                            │
                     ┌──────────────┬───────┴──────┬──────────────┐
                     ▼              ▼              ▼              ▼
                   [DoS]         [DDoS]         [Probe]    [Brute Force]
                     │              │              │              │
                     └──────────────┴───────┬──────┴──────────────┘
                                            ▼
                              ┌───────────────────────────┐
                              │  Level 3: Alert Engine    │
                              │  Severity + IP Quarantine │
                              └───────────────────────────┘
```

- **Level 1 (Binary Filter)**: Inspects 100% of network flows in sub-microseconds. Flags malicious traffic while instantly clearing benign connections.
- **Level 2 (Attack Diagnostic)**: Activated only on malicious traffic to categorize specific attack families: **DoS, DDoS, Probe (PortScan), BruteForce**.
- **Alert & Response Engine**: Emits structured JSON alerts and auto-generates firewall quarantine rules (Linux `iptables` / Null-route commands).

---

## 🧪 Experimental Model Matrix

The project benchmarks 6 distinct models spanning Classical ML and Deep Learning:

### Traditional Machine Learning
1. **Logistic Regression**: Linear baseline; sub-microsecond latency.
2. **Support Vector Machine (SVM)**: Maximum-margin hyperplane classification with RBF kernel.
3. **Random Forest**: Ensemble bagging over decision trees for tabular telemetry.
4. **XGBoost / Gradient Boosting**: State-of-the-art gradient-boosted decision trees.

### Deep Learning
5. **Artificial Neural Network (ANN)**: Multi-layer perceptron (Input $\rightarrow$ Dense 64 $\rightarrow$ Dense 32 $\rightarrow$ Output).
6. **1D-CNN (Convolutional Neural Network)**: 1D convolutional feature extraction sliding across flow attributes to capture spatial and local cross-feature correlations.

---

## 📁 Project Structure

```
network_intrusion_detection/
├── data/
│   └── dataset.py             # Telemetry generator matching CIC-IDS2017 & NSL-KDD
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # StandardScaler, Level-1 & Level-2 label partitioning
│   ├── alert_engine.py         # Severity scoring & firewall rule generator
│   ├── pipeline.py             # TwoLevelNIDS hierarchical execution pipeline
│   └── models/
│       ├── __init__.py
│       ├── ml_models.py        # Logistic Regression, SVM, Random Forest, XGBoost
│       ├── ann_model.py        # Artificial Neural Network (MLP)
│       └── cnn_1d.py           # 1D-CNN (PyTorch + NumPy fallback)
├── run_benchmark.py            # Runs the full 6-model benchmark suite
├── run_simulation.py           # Simulates live traffic stream (#1254, #1255, etc.)
├── app.py                      # Interactive Streamlit Web Dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 How to Run

### 1. Run the 6-Model Benchmark Suite
Evaluates Accuracy, Precision, Recall (Detection Rate), F1-Score, Training Time, and Latency per flow:
```bash
python run_benchmark.py
```

### 2. Run the Live Traffic Simulation
Demonstrates the exact two-level workflow with test packets (`#1254`, `#1255`, etc.):
```bash
python run_simulation.py
```

### 3. Launch the Interactive Web Dashboard
Launches the Streamlit UI to test custom network flows and view real-time firewall mitigation:
```bash
streamlit run app.py
```

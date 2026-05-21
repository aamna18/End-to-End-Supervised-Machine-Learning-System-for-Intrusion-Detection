#  End-to-End Supervised Machine Learning System for Intrusion Detection

A complete machine learning pipeline for binary network intrusion detection. Raw packet captures are transformed into flow-level behavioral features, passed through a rigorous preprocessing and noise-injection pipeline, then used to train and compare **18 model configurations** across three algorithm families. Developed for **CS-324 Machine Learning** at FAST-NUCES.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset and Traffic Capture](#dataset-and-traffic-capture)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Model Configurations](#model-configurations)
6. [Notebook Structure](#notebook-structure)
7. [Visualizations Generated](#visualizations-generated)
8. [Performance Results](#performance-results)
9. [Model Saving](#model-saving)
10. [How to Run](#how-to-run)
11. [Project Structure](#project-structure)
12. [Contributors](#contributors)
13. [Limitations and Future Work](#limitations-and-future-work)
14. [AI Use Declaration](#ai-use-declaration)

---

## Project Overview

| Property | Detail |
|----------|--------|
| Task | Binary classification — Normal (0) vs Attack (1) |
| Dataset | 11,051 flow-level samples, 9 engineered features |
| Models trained | 18 (3 variants × 3 families × 2 split ratios) |
| Best model | NN-Aggressive \[80/10/10\] — F1: 0.9092, AUC: 0.9377 |
| Random seed | 42 (fixed across numpy, TensorFlow, sklearn) |
| Notebook | `ML_FINAL_ENHANCED.ipynb` |

**Key design priorities:**
- High recall — missed attacks (False Negatives) are treated as more costly than false alarms
- No data leakage — scaler fitted on training split only, applied to val/test
- Realistic noise injection to simulate sensor uncertainty in real IDS deployments
- Full interpretability — coefficient plots, tree visualisations, and feature commentary alongside every model

---

## Dataset and Traffic Capture

Traffic was captured using Wireshark/Tshark in a controlled lab environment. Both normal and attack traffic were generated:

**Normal traffic:** Google Meet/WebRTC, HTTPS/TLS, QUIC/HTTP3, general browsing  
**Attack traffic:** SYN Flood, UDP Flood, ICMP Flood, SYN Port Scan, UDP Scan, Aggressive Nmap Scan

Attacks were generated on Kali Linux using `hping3` and `nmap` against an isolated target VM.

Raw PCAP files were parsed with Tshark, time-window aggregated, and engineered into the following feature set:

| Feature | Description |
|---------|-------------|
| `total_packets` | Total packets in the flow window |
| `flow_duration_sec` | Duration of the aggregated flow |
| `pkt_len_mean` | Mean packet length |
| `pkt_len_min` | Minimum packet length |
| `iat_min` | Minimum inter-arrival time between packets |
| `fin_count` | Count of TCP FIN flags |
| `ttl_mean` | Mean Time-To-Live value |
| `retransmission_count` | TCP retransmission count |
| `proto_icmp` | Binary indicator for ICMP protocol |

**Class distribution:** ~47.2% Normal · ~52.8% Attack (near-balanced)

---

## Preprocessing Pipeline

All steps are implemented in Section 1 of the notebook (`ML_FINAL_ENHANCED.ipynb`):

### 1A — Load & Clean
- Drop rows with missing values
- Explicitly remove leakage-prone columns: `proto_other`, `syn_ack_ratio`, `window_size_mean`
- Drop highly correlated features at threshold **r > 0.85** (Pearson, upper triangle)

### 1B — MI-Proportional Noise Injection
To simulate realistic sensor uncertainty in IDS traffic:
- Mutual Information scores are computed for each feature against the label
- Gaussian noise is injected proportional to each feature's MI rank: `σ_factor = 0.20 + 0.60 × (MI / MI_max)`, ranging from **20% to 80%** of each feature's standard deviation
- An additional **5% random label flip** is applied to simulate mislabelled traffic samples

### 1C — Train/Val/Test Splits
Two split strategies are evaluated:

| Split | Train | Validation | Test |
|-------|-------|-----------|------|
| 70/15/15 | 70% | 15% | 15% |
| 80/10/10 | 80% | 10% | 10% |

Both use **stratified splitting** to preserve class ratios. `StandardScaler` is fitted **only on the training set** and applied to validation and test — no leakage.

---

## Exploratory Data Analysis

Section 1.5 of the notebook provides a full EDA on the raw dataset (before noise injection):

| EDA Step | What it shows |
|----------|--------------|
| **Class Distribution** | Bar chart + pie chart of Normal vs Attack counts with percentages |
| **Missing Values Check** | Per-column missing counts and %, plus dtype summary and `describe()` statistics |
| **Correlation Heatmap** | Lower-triangle Pearson heatmap (coolwarm, annotated) with top features ranked by `\|r\|` against the label |
| **Feature Histograms** | Per-feature density histograms with Normal (blue) and Attack (red) overlaid, one subplot per feature |

---

## Model Configurations

### Logistic Regression

All variants use `class_weight='balanced'` and `max_iter=5000`.

| Variant | Penalty | C | Solver | Effect |
|---------|---------|---|--------|--------|
| LR-L1 | L1 (LASSO) | 0.1 | liblinear | Sparse weights → automatic feature selection |
| LR-L2 | L2 (Ridge) | 0.1 | lbfgs | Uniform weight shrinkage |
| LR-EN | ElasticNet | 0.05 | saga | Combined L1+L2, `l1_ratio=0.6` |

**Why LR for IDS:** Coefficients directly map to auditable detection reasons. L1 suppresses noisy features. Probabilistic output enables FPR/FNR threshold tuning. Trains in seconds.

### Decision Tree

| Variant | Depth | Criterion | `min_samples_split` | Character |
|---------|-------|-----------|---------------------|-----------|
| DT-Shallow | 3 | Gini | 2 | Maximally interpretable, extracts human-readable rules |
| DT-Balanced | 7 | Entropy | 5 | Bias-variance sweet spot |
| DT-Deep | None | Gini | 2 | Full tree — reveals overfitting risk |

**Why DT for IDS:** Each root-to-leaf path is a human-readable detection rule (e.g., `IF packet_rate > 500 AND protocol = UDP → ATTACK`). Built-in feature importance. O(log n) inference for high-throughput inspection.

### Neural Network

All variants use Adam optimiser, binary cross-entropy loss, batch size 32, and EarlyStopping on `val_loss` with `restore_best_weights=True`.

| Variant | Architecture | Regularisation | Dropout | Max Epochs | Patience |
|---------|-------------|----------------|---------|------------|---------|
| NN-Conservative | 16 → 8 → 1 | L2 (1e-2) | 0.6 / 0.5 | 80 | 6 |
| NN-Balanced | 32 → 16 → 1 | L2 (1e-3) | 0.5 / 0.4 | 100 | 8 |
| NN-Aggressive | 64 → 32 → 16 → 1 | L1 (1e-4) | 0.3 / 0.2 | 120 | 10 |

All hidden layers use ReLU activation; the output layer uses sigmoid.

**Why NN for IDS:** Hidden layers automatically learn latent traffic representations. Handles high-dimensional, correlated, noisy features naturally. Dropout + weight penalties prevent overfitting under the MI-proportional noise regime.

---

## Notebook Structure

```
ML_FINAL_ENHANCED.ipynb
│
├── § 1    Setup, Imports & Data Preprocessing
│          1A Load & Clean → 1B MI Noise → 1C Splits → 1D Metric Helpers
│
├── § 1.5  Exploratory Data Analysis
│          EDA-1 Class Distribution · EDA-2 Missing Values
│          EDA-3 Correlation Heatmap · EDA-4 Feature Histograms
│
├── § 2    Algorithm Implementations (definitions only)
│          2A Logistic Regression · 2B Decision Tree · 2C Neural Network
│          (each preceded by a "Why this algorithm for IDS?" markdown cell)
│
├── § 3    Model Training & Evaluation
│          3A LR Train → Coefficient Plot → Feature Commentary → 5-Fold CV
│             → Confusion Matrix → ROC & PR Curves
│          3B DT Train → Tree Visualisation → Learning Curves
│             → Split Ratio Effect → Confusion Matrix → ROC & PR Curves
│          3C NN Train → Training Curves (Loss + Accuracy)
│             → Confusion Matrix → ROC & PR Curves
│
├── § 4    Comparative Analysis — All 18 Models
│          4A Master Results Table · 4B F1 & AUC Bar Charts
│          4C Performance Heatmap · 4D FPR vs FNR
│          4E Accuracy vs F1 Scatter · 4F Category Averages
│          4G Generalisation Gap · 4H Split Ratio Effect
│          4I Training Time Comparison · 4J Final Summary
│
├── § 5    Best Model & Save
│          Formatted best-model report + joblib / Keras save
│
└── § 6    Interactive Prediction Interface
           9-model ensemble with per-model attack probability + ensemble vote
```

---

## Visualizations Generated

The notebook produces the following plots inline:

**EDA**
- Target class distribution (bar + pie)
- Feature correlation heatmap (lower triangle, coolwarm)
- Per-feature density histograms: Normal vs Attack overlay

**Logistic Regression**
- Top-12 feature coefficient bar chart (LR-L1, 70/15/15)
- Confusion matrix — counts and normalised (LR-L1, 70/15/15)
- ROC curves for all 3 LR variants (70/15/15 test set)
- Precision-Recall curves for all 3 LR variants

**Decision Tree**
- Full tree diagram — DT-Shallow (depth=3), 70/15/15
- Learning curves for all 3 DT variants (5-fold, 70/15/15)
- Split ratio effect table (DT-Balanced)
- Confusion matrix — counts and normalised (DT-Balanced, 70/15/15)
- ROC curves for all 3 DT variants
- Precision-Recall curves for all 3 DT variants

**Neural Network**
- Training loss and accuracy curves — 3 variants × 70/15/15 split
- Confusion matrix — counts and normalised (NN-Aggressive, 80/10/10)
- ROC curves for all 3 NN variants (80/10/10 test set)
- Precision-Recall curves for all 3 NN variants

**Comparative (18 Models)**
- F1 Score bar chart — all 18 models, colour-coded by algorithm family
- ROC-AUC bar chart — all 18 models
- Performance heatmap — 18 models × 7 metrics (RdYlGn)
- False Positive Rate bar chart
- False Negative Rate bar chart
- Accuracy vs F1 scatter plot with annotations
- Category-average grouped bar chart
- Generalisation gap (Train F1 − Val F1) bar chart with overfit threshold
- Split ratio effect (F1 by model, 70/15/15 vs 80/10/10, per algorithm)
- Training time comparison bar chart

---

## Performance Results

The Neural Network family consistently outperformed linear and tree-based models. The best single configuration was **NN-Aggressive \[80/10/10\]**:

| Metric | Value |
|--------|-------|
| Accuracy | 0.9033 |
| Precision | 0.8933 |
| Recall | 0.9257 |
| F1-Score | 0.9092 |
| ROC-AUC | 0.9377 |
| False Positive Rate | — |
| False Negative Rate | 0.0743 |

**Category-level summary:**

| Family | Avg F1 | Avg AUC | Notes |
|--------|--------|---------|-------|
| Neural Network | Best | Best | Captured non-linear attack patterns; all variants generalised well (Train-Val F1 gap < 0.05) |
| Logistic Regression | 0.8984 | — | Stable, interpretable baseline; ElasticNet most regularisation-consistent |
| Decision Tree | — | — | DT-Balanced was the best DT; DT-Deep overfits on 70/15/15, partially recovers on 80/10/10 |

**Key observations:**
- Increased training data (80/10/10) improves LR and NN performance but does not resolve DT-Deep's memorisation issues
- FPR is lowest for NN-Aggressive; FNR is lowest for LR variants (high recall prioritisation)
- NNs are the slowest to train but provide the best accuracy-to-FPR tradeoff

---

## Model Saving

After identifying the best model, Section 5 saves both the model and its scaler:

```python
# sklearn models (LR / DT)
joblib.dump(best_model, "best_model.pkl")
joblib.dump(best_scaler, "best_scaler.pkl")

# Keras / TF models (NN)
best_model.save("best_model_nn.keras")
joblib.dump(best_scaler, "best_scaler.pkl")
```

To reload and use:
```python
import joblib
model  = joblib.load("best_model.pkl")      # or load_model("best_model_nn.keras")
scaler = joblib.load("best_scaler.pkl")

sample_scaled = scaler.transform(new_sample)
prediction    = model.predict(sample_scaled)
```

---

## How to Run

### Requirements
```
Python 3.8+
tensorflow >= 2.10
scikit-learn >= 1.2
pandas
numpy
matplotlib
seaborn
joblib
jupyter
```

### Installation
```bash
pip install -r requirements.txt
```

### Running the Notebook
```bash
jupyter notebook ML_FINAL_ENHANCED.ipynb
```

Run cells **sequentially from top to bottom**. Each section depends on variables set in the previous ones. The full run (including all NN training) typically takes 3–8 minutes on CPU depending on hardware.

### Interactive Prediction Interface (Section 6)
At the end of the notebook, an interactive cell prompts for values of each feature. Press **Enter** to accept the training-set median as a default for any feature.

Output per model:
```
  Model                  Prediction   Atk Prob   Nrm Prob
  ─────────────────────  ──────────   ────────   ────────
  LR-L1                  🔴 ATTACK      87.3%      12.7%
  NN-Aggressive          🔴 ATTACK      96.1%       3.9%
  ...
  ─────────────────────────────────────────────────────────
  ENSEMBLE VOTE (7/9):   🔴 ATTACK
  Average Attack Probability : 91.4%
```

---

## Project Structure
```
.
├── ML_FINAL_ENHANCED.ipynb    # Main notebook — full pipeline, training, evaluation
├── ids_dataset_ready_for_ml.csv   # Preprocessed flow-level dataset
├── best_model.pkl             # Saved best sklearn model (generated on run)
├── best_model_nn.keras        # Saved best Keras model if NN wins (generated on run)
├── best_scaler.pkl            # Matching StandardScaler (generated on run)
├── data/                      # Raw PCAP and intermediate CSV files
├── src/                       # Tshark extraction and feature engineering scripts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Contributors

| Name | Student ID |
|------|-----------|
| Amna Ahmed | CS-23016 |
| Waleed Ahmed Khan | CS-23044 |
| Abdul Wahab | CS-23035 |

**Instructors:** Dr. Maria Waqas · Miss Mahnoor Malik  
**Course:** CS-324 Machine Learning — FAST-NUCES  

> A separate technical report accompanies this notebook with full methodology, results analysis, and theoretical justification for all design decisions.

---

## Limitations and Future Work

**Current limitations:**
- Dataset represents a controlled lab environment and may not capture the full diversity of real-world intrusion patterns
- Feature masking edge cases — attack traffic with benign-looking feature values can occasionally be misclassified
- Offline batch classification only; no live stream processing capability
- The noise injection pipeline (MI-proportional Gaussian + 5% label flip) is a simulation of sensor uncertainty, not a validated real-world noise model

**Planned future work:**
- Ensemble methods: Random Forest, XGBoost, stacking
- Online / incremental learning for concept drift in evolving attack patterns
- Multi-class attack categorisation (SYN Flood, Port Scan, UDP Flood, etc. as separate classes)
- Transformer-based temporal models over packet sequences
- Real-time deployment dashboard with live FPR/FNR monitoring
- SHAP-based global feature attribution across all model families

---

## AI Use Declaration

Generative AI (ChatGPT) was used for debugging assistance, visualisation suggestions, and report refinement. All implementation, experimentation, hyperparameter decisions, analysis, and architectural choices were performed and verified by the student team.

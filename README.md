# Predictive Maintenance + RAG: End-to-End System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Model](https://img.shields.io/badge/Model-XGBoost%20%2B%20TreeSHAP-orange.svg)](https://xgboost.readthedocs.io/)
[![Retrieval](https://img.shields.io/badge/Retrieval-TF--IDF%20%2B%20Cosine%20Similarity-lightgrey.svg)](https://scikit-learn.org/)

An end-to-end industrial machine failure prediction and retrieval-augmented generation (RAG) system built on the **UCI AI4I 2020 Predictive Maintenance Dataset**.

The system implements the mandatory operational flow:
$$\text{Sensor Row} + \text{Question} \longrightarrow \text{ML Failure Probability} \longrightarrow \text{Top SHAP Factors} \longrightarrow \text{Targeted Retrieval} \longrightarrow \text{Grounded Maintenance Answer + Citations}$$

---

## 1. System Overview & Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Sensor & Operational Data                │
                    │   Type, Air Temp, Process Temp, RPM, Torque, Tool Wear │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Machine Learning Pipeline (XGBoost + scale_pos_weight)                                  │
│    • Predicts failure probability P(failure)                                                │
│    • Applies decision threshold (tau = 0.50, selected on Validation F1)                     │
│    • Confusion Matrix on Test: TP=29, FP=15, TN=951, FN=5 (Recall=85.3%, Precision=65.9%)   │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Local Explainability (TreeSHAP)                                                          │
│    • Computes signed log-odds contributions for all 6 features                              │
│    • Identifies primary drivers pushing toward or away from failure                         │
│    • Screens mechanical conditions (Heat Dissipation, Power Curve, Overstrain, Tool Wear)   │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Indexed Knowledge Retrieval (TF-IDF Vector Index, 454 Passages)                          │
│    • 7 indexed technical sources (NIST, NASA, Fluke, SKF)                                   │
│    • Query formulated dynamically from SHAP drivers + abnormal sensor screens               │
│    • Cosine similarity ranking with source diversity constraints                            │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Grounded Maintenance Guidance & Citation Synthesis                                       │
│    • Strictly distinguishes statistical prediction from confirmed physical diagnosis       │
│    • Synthesizes only claims supported by retrieved passages with explicit citations        │
│    • Evaluates claim support verification (100% supported across 20 test cases)             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
RAG-Maintenance/
├── README.md                           # Setup, run commands, architecture & declarations
├── REPORT.md                           # Comprehensive 3-page technical report
├── requirements.txt                    # Python dependencies
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv                # Raw UCI AI4I 2020 dataset (10,000 records)
│   ├── processed/
│   │   ├── train.csv                   # Stratified training split (8,000 rows, 80%)
│   │   ├── validation.csv              # Stratified validation split (1,000 rows, 10%)
│   │   └── test.csv                    # Untouched test split (1,000 rows, 10%)
│   └── knowledge_base/
│       ├── sources.json                # Metadata for 7 public technical sources
│       ├── documents/                  # Raw extracted texts from PDFs and HTMLs
│       ├── passages.json               # 454 chunked knowledge passages (180 words, 40 overlap)
│       └── index/                      # Serialized TF-IDF vectorizer and matrix
├── models/
│   ├── logistic_regression.joblib      # Baseline linear classifier
│   ├── xgboost.joblib                  # Tuned XGBoost classifier
│   ├── selected_model.json             # Selected model metadata and chosen threshold
│   └── training_info.json              # Training hyperparameters and sample counts
├── results/
│   ├── model_comparison.csv            # Validation comparison (Logistic Regression vs XGBoost)
│   ├── threshold_comparison.csv        # Validation metric trajectory across thresholds
│   ├── final_metrics.json              # Final test set metrics & confusion matrix
│   ├── final_predictions.csv           # Test set predictions & probabilities
│   ├── explanations.csv                # Full test set SHAP explanations
│   └── end_to_end_evaluation.csv       # 20-case end-to-end evaluation table
└── src/
    ├── preprocessing.py                # Train/Val/Test stratified splitting & leakage removal
    ├── validate_data.py                # Schema, range, and split consistency checks
    ├── train.py                        # Model training (Baseline + XGBoost with imbalance weighting)
    ├── evaluate.py                     # Validation evaluation & threshold tuning
    ├── final_evaluation.py             # Final evaluation on untouched test set
    ├── build_kb.py                     # Source scraping, PDF parsing, chunking
    ├── retrieve.py                     # TF-IDF indexing and passage retrieval engine
    ├── explainability.py               # TreeSHAP computation, query synthesis & RAG logic
    └── end_to_end_evaluation.py        # 20-case end-to-end evaluation runner
```

---

## 3. Setup and Installation

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Virtual environment tool (`venv` or `conda`)

### Step 1: Create and Activate Virtual Environment
```bash
# Using venv
python -m venv venv

# On Windows (Command Prompt / PowerShell)
.\venv\Scripts\activate

# On Linux / macOS
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Execution & Reproduction Pipeline

To reproduce the entire pipeline from scratch, execute the following commands in sequence:

```bash
# 1. Preprocess data and create stratified splits (leakage fields excluded)
python src/preprocessing.py

# 2. Validate data types, ranges, class balances, and split integrity
python src/validate_data.py

# 3. Train models (Logistic Regression baseline and XGBoost with scale_pos_weight)
python src/train.py

# 4. Evaluate models on validation set and select optimal decision threshold
python src/evaluate.py

# 5. Evaluate selected model on the untouched test set
python src/final_evaluation.py

# 6. Fetch external technical sources and build chunked knowledge base
python src/build_kb.py

# 7. Test knowledge base indexing and retrieval functionality
python src/retrieve.py

# 8. Run full End-to-End Pipeline on 20 test cases (SHAP -> Retrieval -> Grounded Answer)
python src/end_to_end_evaluation.py
```

---

## 5. Leakage Control Rationale

Strict data leakage boundaries were enforced during preprocessing. The following fields from `ai4i2020.csv` are **strictly excluded** from model inputs:

| Excluded Field | Nature | Technical Rationale for Exclusion |
| :--- | :--- | :--- |
| **`UDI`** | Record Index | Sequential integer identifier ($1 \dots 10000$). Arbitrary row index with no physical or mechanical relation to machine degradation; trees could memorize row ranges. |
| **`Product ID`** | Identifier | Machine quality variant prefix (`L`/`M`/`H`) combined with a sequential serial number (e.g., `L47181`). Quality variant is already represented by `Type`. Serial digits represent manufacturing sequence numbers that would cause spurious temporal/batch leakage. |
| **`Machine failure`** | Ground Truth | Target variable. Retaining it in feature space is direct label leakage. |
| **`TWF`** (Tool Wear Failure) | Failure Mode Flag | Consequence flag indicating machine failure due to tool wear ($200 \le \text{wear} \le 240$ min). In operation, failure modes are only diagnosed *after* failure occurs; using failure mode flags as inputs represents downstream symptom leakage ($Machine\ failure = TWF \lor HDF \lor PWF \lor OSF \lor RNF$). |
| **`HDF`** (Heat Dissipation Failure) | Failure Mode Flag | Consequence flag indicating heat dissipation collapse ($\Delta T < 8.6\text{ K}$ and $\text{speed} < 1380\text{ rpm}$). Direct indicator of failure occurrence. |
| **`PWF`** (Power Failure) | Failure Mode Flag | Consequence flag indicating power out of bounds ($P < 3500\text{ W}$ or $P > 9000\text{ W}$). Direct indicator of failure occurrence. |
| **`OSF`** (Overstrain Failure) | Failure Mode Flag | Consequence flag indicating product of tool wear and torque exceeded threshold. Direct indicator of failure occurrence. |
| **`RNF`** (Random Failure) | Failure Mode Flag | Consequence flag indicating unpredicted random failure ($0.1\%$ probability). Direct indicator of failure occurrence. |

---

## 6. Model Evaluation Summary

### Validation Set Comparison (Threshold = 0.50)
| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.1613 | 0.8824 | 0.2727 | 0.9098 | 0.3586 |
| **XGBoost (Selected)** | **0.6667** | **0.7059** | **0.6857** | **0.9732** | **0.7689** |

### Threshold Tuning on Validation Set (XGBoost)
Validation tuning across thresholds $\tau \in [0.10, 0.50]$ showed that $\tau = 0.50$ maximized the $F_1$-score ($0.6857$), balancing false alarm suppression against critical failure detection.

### Final Untouched Test Set Evaluation ($\tau = 0.50$)
* **Precision**: 65.91% ($0.6591$)
* **Recall**: 85.29% ($0.8529$)
* **F1-Score**: 74.36% ($0.7436$)
* **ROC-AUC**: 97.05% ($0.9705$)
* **PR-AUC**: 87.86% ($0.8786$)

#### Test Confusion Matrix ($N = 1,000$ rows, 34 actual failures)
| | Predicted Negative (0) | Predicted Positive (1) |
| :--- | :---: | :---: |
| **Actual Non-Failure (0)** | **951** (TN) | **15** (FP) |
| **Actual Failure (1)** | **5** (FN) | **29** (TP) |

---

## 7. RAG Knowledge Base Sources

The knowledge base indexes 7 public technical publications (454 distinct passages):

1. **NIST** – *Comprehensive evaluations of condition monitoring-based technologies in industrial maintenance: A systematic review* ([URL](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=959739))
2. **NIST** – *Manufacturing Machinery Maintenance* ([URL](https://www.nist.gov/el/applied-economics-office/manufacturing/topics-manufacturing/manufacturing-machinery-maintenance))
3. **NIST** – *Monitoring, Diagnostics and Prognostics for Manufacturing Operations* ([URL](https://www.nist.gov/programs-projects/monitoring-diagnostics-and-prognostics-manufacturing-operations))
4. **NASA** – *NPR 8831.2F - Maintenance* ([URL](https://nodis3.gsfc.nasa.gov/displayDir.cfm?Internal_ID=N_PR_8831_002F_&page_name=Chapter7))
5. **Fluke** – *When and how to use and set up vibration monitoring on assets* ([URL](https://www.fluke.com/en/learn/blog/condition-monitoring/when-and-how-to-use-and-set-up-vibration-monitoring-on-assets))
6. **Fluke** – *Using Thermal Imaging to Monitor Motors and Gearboxes* ([URL](https://www.fluke.com/en-au/learn/blog/motors-drives-pumps-compressors/using-thermal-imaging-to-monitor-motors-and-gearboxes))
7. **SKF** – *Machine health monitor* ([URL](https://evolution.skf.com/machine-health-monitor/))

---

## 8. Declarations

* **Programming Language**: Python 3.10+
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost` (v2.1.4), `joblib`, `shap` (v0.46.0), `requests`, `beautifulsoup4`, `pypdf`.
* **ML Algorithms**: `LogisticRegression` (balanced class weighting), `XGBClassifier` (`scale_pos_weight` weighting), `TreeSHAP` (`shap.TreeExplainer`).
* **Retrieval Model**: Scikit-Learn `TfidfVectorizer` (sub-linear TF scaling, n-gram range (1,2), English stop-words removal) + Cosine Similarity ranking.
* **Datasets & Documents**: UCI Machine Learning Repository (AI4I 2020 Predictive Maintenance Dataset, CC BY 4.0); Public engineering guidelines from NIST, NASA, Fluke, and SKF.
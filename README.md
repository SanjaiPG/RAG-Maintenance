# Predictive Maintenance with RAG

This project predicts machine failures from sensor readings and then explains each prediction with passages from public maintenance literature. It is built on the UCI AI4I 2020 Predictive Maintenance dataset (10,000 records, CC BY 4.0).

Given a sensor row and a question, the system works in this order: an XGBoost model estimates the failure probability, TreeSHAP picks out the features driving that estimate, those drivers are turned into a retrieval query over a TF-IDF index of technical documents, and the answer is written only from the retrieved passages, with citations.

## How it works

The classifier takes six inputs: machine type, air temperature, process temperature, rotational speed, torque and tool wear. It outputs a failure probability, and a threshold of 0.50 (chosen on the validation set by F1) turns that into a yes/no flag. On the test set this gave 29 true positives, 15 false positives, 951 true negatives and 5 false negatives.

SHAP then gives each feature a signed contribution in log-odds, which shows what pushed the prediction toward or away from failure. The code also checks four mechanical conditions (heat dissipation, power curve, overstrain and tool wear) against the raw readings.

The strongest SHAP drivers and any abnormal conditions are combined into a query. It runs against 454 passages from 7 public sources (NIST, NASA, Fluke, SKF), ranked by cosine similarity with a limit on how many passages may come from one source.

The final answer presents the model output as a statistical prediction and makes no claim of a confirmed diagnosis. Every claim in it has to be supported by a retrieved passage and carries a citation. Across the 20 end-to-end test cases, all claims were supported.

## Setup

You need Python 3.10, 3.11 or 3.12. Create an environment with either venv or conda, then install the dependencies.
 
With venv:
 
```bash
python -m venv venv
 
# Windows
.\venv\Scripts\activate
 
# Linux / macOS
source venv/bin/activate
 
pip install -r requirements.txt
```
 
With conda:
 
```bash
conda create -n rag-maintenance python=3.11
conda activate rag-maintenance
 
pip install -r requirements.txt
```

## Running the pipeline

Run the scripts in this order to reproduce everything from the raw data:

```bash
python src/preprocessing.py          # stratified train/validation/test split, leakage fields dropped
python src/validate_data.py          # schema, range and split checks
python src/train.py                  # logistic regression baseline and XGBoost
python src/evaluate.py               # validation metrics and threshold selection
python src/final_evaluation.py       # one evaluation on the untouched test set
python src/build_kb.py               # download sources and build the passage index
python src/retrieve.py               # sanity check on retrieval
python src/end_to_end_evaluation.py  # 20 cases: SHAP, retrieval, grounded answer
```

The split is 80/10/10 (8,000 train, 1,000 validation, 1,000 test). Passages are 180 words long with a 40-word overlap.

## Leakage control

Several columns in `ai4i2020.csv` are left out of the model inputs because they either identify the row or reveal the outcome.

| Column | Why it is excluded |
| :--- | :--- |
| UDI | Sequential row index. It has no physical meaning, but trees could memorize ranges of it. |
| Product ID | A quality prefix (L/M/H) plus a serial number. The prefix is already captured by Type, and the serial would leak manufacturing order. |
| Machine failure | The target itself. |
| TWF, HDF, PWF, OSF, RNF | Failure mode flags. Machine failure is the OR of these five, and in practice the mode is only known after a failure has happened. |

## Results

### Validation (threshold 0.50)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic regression (baseline) | 0.1613 | 0.8824 | 0.2727 | 0.9098 | 0.3586 |
| XGBoost (selected) | 0.6667 | 0.7059 | 0.6857 | 0.9732 | 0.7689 |

I tried thresholds from 0.10 to 0.50 on the validation set. 0.50 gave the best F1 (0.6857), so it was kept.

### Test set (threshold 0.50, 1,000 rows, 34 actual failures)

| Metric | Value |
| :--- | :---: |
| Precision | 0.6591 |
| Recall | 0.8529 |
| F1 | 0.7436 |
| ROC-AUC | 0.9705 |
| PR-AUC | 0.8786 |

| | Predicted 0 | Predicted 1 |
| :--- | :---: | :---: |
| Actual 0 | 951 | 15 |
| Actual 1 | 5 | 29 |

## Knowledge base sources

1. NIST, [Comprehensive evaluations of condition monitoring-based technologies in industrial maintenance: A systematic review](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=959739)
2. NIST, [Manufacturing Machinery Maintenance](https://www.nist.gov/el/applied-economics-office/manufacturing/topics-manufacturing/manufacturing-machinery-maintenance)
3. NIST, [Monitoring, Diagnostics and Prognostics for Manufacturing Operations](https://www.nist.gov/programs-projects/monitoring-diagnostics-and-prognostics-manufacturing-operations)
4. NASA, [NPR 8831.2F, Maintenance](https://nodis3.gsfc.nasa.gov/displayDir.cfm?Internal_ID=N_PR_8831_002F_&page_name=Chapter7)
5. Fluke, [When and how to use and set up vibration monitoring on assets](https://www.fluke.com/en/learn/blog/condition-monitoring/when-and-how-to-use-and-set-up-vibration-monitoring-on-assets)
6. Fluke, [Using Thermal Imaging to Monitor Motors and Gearboxes](https://www.fluke.com/en-au/learn/blog/motors-drives-pumps-compressors/using-thermal-imaging-to-monitor-motors-and-gearboxes)
7. SKF, [Machine health monitor](https://evolution.skf.com/machine-health-monitor/)

## Tools used

Python 3.10+ with pandas, numpy, scikit-learn, xgboost 2.1.4, joblib, shap 0.46.0, requests, beautifulsoup4 and pypdf.

The models are logistic regression (balanced class weights), XGBoost (`scale_pos_weight` for the class imbalance) and `shap.TreeExplainer`. Retrieval uses scikit-learn's `TfidfVectorizer` with sublinear TF, unigrams and bigrams, and English stop words removed, ranked by cosine similarity.

The data comes from the UCI Machine Learning Repository (AI4I 2020, CC BY 4.0), and the documents come from the NIST, NASA, Fluke and SKF pages listed above.

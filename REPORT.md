# Predictive Maintenance + RAG: System Report

**Dataset**: UCI AI4I 2020 Predictive Maintenance Dataset (DOI: 10.24432/C5HS5C | CC BY 4.0)  
**System Objective**: Predict machine failure probability from multi-sensor operational streams, compute local model explanations, retrieve relevant technical maintenance guidance from indexed public engineering literature, and synthesize a grounded, source-cited recommendation.

---

## 1. System Architecture & Methodology

The system bridges statistical machine learning and structured industrial knowledge retrieval through a 4-stage pipeline:

```
[Sensor Stream: 6 Features] ──> [XGBoost Classifier] ──> P(Failure) ──> [Threshold tau = 0.50]
                                          │
                                          ▼
                               [TreeSHAP Explainer] ──> Top Local Drivers (Signed Log-Odds)
                                          │
                                          ▼
                               [Dynamic Query Builder] ──> (Sensors + SHAP Drivers + Screening Rules)
                                          │
                                          ▼
                               [TF-IDF Passage Index] ──> Top-K Retrieved Passages (NIST/NASA/Fluke/SKF)
                                          │
                                          ▼
                          [Grounded Guidance Synthesizer] ──> Source-Cited Recommendation
                                                              (Strict Epistemic Distinction)
```

### 1.1 Data Preparation & Leakage Control
The dataset contains 10,000 industrial machine records with an severe class imbalance ($3.39\%$ failure rate, 339 positive cases). To avoid data snooping, data was split before any modeling into:
* **Train**: 8,000 rows ($80.0\%$), 271 failures ($3.39\%$)
* **Validation**: 1,000 rows ($10.0\%$), 34 failures ($3.40\%$)
* **Test**: 1,000 rows ($10.0\%$), 34 failures ($3.40\%$)

Splits were stratified on `Machine failure` with seed 42. The test split was kept strictly untouched until final evaluation.

#### Leakage Control Justifications
Input features were strictly constrained to the 6 legitimate pre-failure operational measurements: `Type` (categorical: L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. All other columns were intentionally excluded:
1. **`UDI`**: Arbitrary row index ($1 \dots 10000$). Possesses zero causal or mechanical relation to failure; tree models could memorize index partitions.
2. **`Product ID`**: Contains machine quality variant prefix (`L`/`M`/`H`) plus a sequential serial number. The variant is already captured in `Type`. The sequential integer reflects manufacturing sequence/batch timing, causing temporal leakage.
3. **`Machine failure`**: The target ground truth. Retaining it would constitute direct label leakage.
4. **`TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), `RNF` (Random Failure)**: These binary flags describe the specific post-hoc failure modes that constitute the target ($Machine\ failure = TWF \lor HDF \lor PWF \lor OSF \lor RNF$). In practical plant operations, operators do not know the failure mode prior to diagnosis; using them at inference time represents downstream symptom leakage.

### 1.2 Model Training & Imbalance Handling
Because accuracy is uninformative in a $96.6\%$ negative dataset (a trivial negative baseline achieves $96.6\%$ accuracy with $0.0\%$ recall), PR-AUC, ROC-AUC, Precision, Recall, and $F_1$-score were the primary evaluation metrics. Two models were developed and compared:
1. **Baseline Model**: Logistic Regression with `class_weight='balanced'`, preceded by `StandardScaler` for continuous features and `OneHotEncoder` for `Type`.
2. **Advanced Model**: Gradient Boosted Decision Trees (`XGBClassifier`) parameterized with `scale_pos_weight = N_neg / N_pos \approx 28.5`, `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`, and sub-sampling ($0.8$).

### 1.3 Local Explainability Engine
To avoid treating the model as an opaque black box, every evaluated row passes through `TreeSHAP` (`shap.TreeExplainer`). TreeSHAP computes the exact additive marginal contribution of each feature in log-odds space:
$$\ln\left(\frac{P(\text{failure})}{1 - P(\text{failure})}\right) = \phi_0 + \sum_{i=1}^{M} \phi_i$$
Categorical one-hot encoded SHAP values (`categorical__Type_H`, `categorical__Type_L`, etc.) are mapped back to their original domain feature (`Type`). The engine ranks features by absolute magnitude and labels the directionality (e.g., whether high torque pushed the prediction toward failure or moderate speed pulled it toward normal operation).

### 1.4 Knowledge Base & Indexed Retrieval Pipeline
Rather than passing an unindexed document dump to a language model, an indexed information retrieval pipeline was constructed over 7 public technical publications across condition monitoring, rotating equipment, and industrial maintenance:
* **NIST**: Systematic review of condition monitoring technologies; Manufacturing machinery maintenance; Monitoring, diagnostics, and prognostics for smart operations.
* **NASA**: NPR 8831.2F Reliability-Centered Maintenance & Predictive Testing.
* **Fluke**: Vibration monitoring guidelines; Thermal imaging on motors and gearboxes.
* **SKF**: Machine health and condition monitoring indicators.

Documents were extracted from PDF/HTML, cleaned, and partitioned into 454 semantic passages ($180$ words per chunk with $40$ words overlap). The corpus is indexed via `TfidfVectorizer` (sub-linear TF scaling, n-gram range (1,2), English stop-words) and queried using cosine similarity with a maximum allocation of 2 passages per source to guarantee source diversity.

### 1.5 Grounded RAG Synthesis & Epistemic Boundaries
Queries are dynamically synthesized by uniting the physical sensor measurements, the top SHAP drivers, and condition screening rules (such as thermal differential $\Delta T < 8.6\text{ K}$, power band deviations, or elevated tool wear). 

The response generation module enforces two critical epistemic guardrails:
1. **Model Prediction vs. Physical Diagnosis**: The generated recommendation explicitly states that the ML failure probability is a statistical estimate based on sensor patterns, not a confirmed physical teardown diagnosis.
2. **Claim Grounding & Citations**: Recommendations are generated using verified claim-level rules where an inspection action (thermal imaging, vibration trending, lubricant analysis, baseline comparison) is only emitted if the retrieved passages explicitly discuss that modality. Every assertion is accompanied by an inline source citation (e.g., `[fluke_thermal_1]`).

---

## 2. Experimental Results & Metric Analysis

### 2.1 Model Comparison on Validation Set
Both models were evaluated on the 1,000-row validation split. XGBoost significantly outperformed the baseline across all discriminative metrics:

| Model Architecture | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.1613 | **0.8824** | 0.2727 | 0.9098 | 0.3586 |
| **XGBoost (Advanced)** | **0.6667** | 0.7059 | **0.6857** | **0.9732** | **0.7689** |

*Analysis*: While Logistic Regression achieved high recall ($88.24\%$), its precision collapsed to $16.13\%$, yielding more than 5 false alarms for every true failure detected. XGBoost achieved an ROC-AUC of $0.9732$ and more than doubled the PR-AUC ($0.7689$ vs $0.3586$), demonstrating superior non-linear feature interaction modeling (e.g., power $= \text{torque} \times \omega$, thermal delta $= T_{\text{process}} - T_{\text{air}}$).

### 2.2 Decision Threshold Optimization
Threshold calibration was conducted across validation probabilities to identify the optimal operational decision boundary:

| Decision Threshold ($\tau$) | Validation Precision | Validation Recall | Validation $F_1$-Score | Operational Trade-off |
| :---: | :---: | :---: | :---: | :--- |
| **0.10** | 0.3333 | 0.8824 | 0.4839 | Aggressive screening; high false alarm cost |
| **0.20** | 0.4375 | 0.8235 | 0.5714 | Moderate screening |
| **0.30** | 0.5098 | 0.7647 | 0.6118 | Balanced conservative setting |
| **0.40** | 0.5952 | 0.7353 | 0.6579 | High precision setting |
| **0.50** | **0.6667** | **0.7059** | **0.6857** | **Optimal validation $F_1$ balance** |

$\tau = 0.50$ was selected as the optimal operating point. At lower thresholds ($\tau < 0.30$), maintenance crews would experience unacceptable alarm fatigue.

### 2.3 Final Evaluation on Untouched Test Set
The selected XGBoost model and threshold ($\tau = 0.50$) were deployed once onto the untouched test set ($N=1,000$):

* **Precision**: $65.91\%$ ($0.6591$)
* **Recall**: $85.29\%$ ($0.8529$)
* **$F_1$-Score**: $74.36\%$ ($0.7436$)
* **ROC-AUC**: $97.05\%$ ($0.9705$)
* **PR-AUC**: $87.86\%$ ($0.8786$)

#### Test Confusion Matrix
| Actual \ Predicted | Predicted Healthy ($\hat{y}=0$) | Predicted Failure ($\hat{y}=1$) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Healthy ($y=0$)** | **951** (True Negatives) | **15** (False Positives) | 966 |
| **Actual Failure ($y=1$)** | **5** (False Negatives) | **29** (True Positives) | 34 |

The test set exhibited strong generalization, achieving $85.29\%$ recall (identifying 29 out of 34 machine failures) while maintaining a low false alarm rate ($15$ false positives out of 966 healthy machines, a False Positive Rate of only $1.55\%$).

---

## 3. End-to-End Evaluation & Case Studies (20 Cases)

An end-to-end evaluation was executed over 20 test cases (10 actual failures, 10 actual healthy machines). Below is a representative cross-section of evaluation cases illustrating the interaction between sensor values, SHAP explanations, retrieval, and grounded synthesis:

| Case ID | True | $P(\text{fail})$ | Pred | Outcome | Top Contributing SHAP Factors | Retrieved Sources | Grounded RAG Guidance | Support |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| **CASE_01** (Row 425) | 1 | 0.970 | 1 | **TP** | Air temp (+3.64), Speed (+1.76), Tool wear (-1.34) | Fluke (Thermal), NIST, NASA | Screened low heat dissipation. Recommended thermal imaging comparison and vibration trending. Citations: `[fluke_thermal_1]`, `[nist_condition_monitoring_65]`. | **100%** |
| **CASE_02** (Row 816) | 0 | 0.002 | 0 | **TN** | Torque (-1.99), Tool wear (-1.89), Air temp (-1.73) | NASA, NIST, SKF | Normal operation. Clarified low risk does not guarantee zero wear; recommended baseline tracking. Citation: `[nist_condition_monitoring_154]`. | **100%** |
| **CASE_06** (Row 77) | 1 | 0.999 | 1 | **TP** | Torque (+3.58), Tool wear (+1.61), Speed (+1.16) | NIST, Fluke, NASA | Screened extreme overstrain (torque 60.1 Nm, wear 215 min). Recommended tool replacement and acoustic/vibration check. Citations: `[nist_condition_monitoring_65]`, `[fluke_thermal_1]`. | **100%** |
| **CASE_10** (Row 717) | 1 | 0.076 | 0 | **FN** | Tool wear (-1.27), Torque (-1.24), Air temp (-0.97) | NIST, NASA, SKF | Subtle failure missed at $\tau=0.50$, but flagged in secondary "Watch" band ($>0.05$). Flagged baseline comparison. Citation: `[nist_condition_monitoring_154]`. | **100%** |

Across all 20 evaluated test cases:
* **Support Rate**: $100\%$ of emitted recommendations were confirmed to be supported by retrieved text passages.
* **Retrieval Diversity**: 14 distinct passage combinations were retrieved, verifying that queries dynamically adapt to the physical condition screens rather than returning a static boilerplate.

---

## 4. Failure Cases & Diagnostic Analysis

### 4.1 False Negative Analysis (Missed Failures)
Out of 34 failure events in the test set, 5 were missed at $\tau = 0.50$. Inspection reveals two distinct failure mechanisms:
1. **Random Failures (RNF mode)**: In the synthetic AI4I dataset, random failures are injected stochastically without antecedent sensor degradation. Because sensor readings remain within normal distributions, TreeSHAP values for all features remain negative (e.g., CASE_10 where torque, tool wear, and temperatures all appeared benign).
2. **Borderline Multi-Factor Degradation**: In instances where tool wear is elevated ($180$ min) but torque is slightly below average ($35$ Nm), the tree model assigns an intermediate failure probability ($P \in [0.15, 0.45]$).
*Mitigation*: The RAG system incorporates a secondary **"Watch" risk band** for probabilities between $0.10$ and $0.50$. Even when $\hat{y} = 0$, machines in this band trigger proactive inspection recommendations rather than standard passive monitoring.

### 4.2 False Positive Analysis (Nuisance Alarms)
The model produced 15 false positives out of 966 healthy machines ($1.55\%$). In these cases, machines operated under high torque ($> 55\text{ Nm}$) or low heat dissipation conditions that mimic failure regimes, yet the tool had not worn down enough to trigger physical breakdown.
*Benefit*: In industrial practice, a false alarm that prompts a non-destructive thermal or vibration check is substantially less costly than an unpredicted in-service machine seizure.

### 4.3 Retrieval Edge Cases
TF-IDF lexical retrieval can occasionally favor high-frequency overview terms over niche mechanical diagnostics (e.g., retrieving systematic literature review tables rather than actionable troubleshooting protocols). To safeguard against this, the pipeline enforces strict regex-based action-term filtering before ranking passages.

---

## 5. What Would Be Improved With More Time

1. **Dense Semantic Embeddings & Neural Reranking**:
   * Replace TF-IDF lexical retrieval with dense bi-encoder embeddings (e.g., `BGE-M3` or `bge-small-en-v1.5`) fine-tuned on industrial maintenance ontologies.
   * Add a cross-encoder reranker (`bge-reranker-large`) over the top 25 retrieved candidates to capture nuanced semantic alignment between sensor anomalies and diagnostic guidance.

2. **Quantified Uncertainty via Conformal Prediction**:
   * Implement split conformal prediction to generate finite-sample valid prediction intervals for machine failure risk, guaranteeing that true failures fall within the uncertainty bounds at a user-specified confidence level ($1 - \alpha = 95\%$).

3. **Multi-Task Failure Mode Diagnostics**:
   * Train a secondary multi-label diagnostic head that activates *only after* a primary failure is predicted, classifying whether the failure is thermal (HDF), mechanical (OSF), electrical (PWF), or tooling (TWF).

4. **Instruction-Tuned LLM with Structured Guardrails**:
   * Integrate a local quantized LLM (e.g., `Llama-3.1-8B-Instruct` or `Mistral-7B`) via vLLM with grammar-constrained decoding (e.g., Outlines) to generate natural language explanations while mathematically guaranteeing that all emitted citations exist in the retrieved passage subset.

---

## 6. Declarations & Compliance

* **Data Sources**: UCI AI4I 2020 Predictive Maintenance Dataset (CC BY 4.0).
* **Technical Literature**: Public condition monitoring and maintenance standards published by NIST, NASA, Fluke Corporation, and SKF Group.
* **Open-Source Software**: Python 3.10+, Scikit-Learn 1.5, XGBoost 2.1.4, SHAP 0.46.0, BeautifulSoup4, PyPDF, Joblib, Pandas, NumPy.
* **Epistemic Standard**: Statistical failure probability is explicitly delineated from physical diagnosis in all system outputs.

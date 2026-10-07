import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


TEST_FILE = "data/processed/test.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"


os.makedirs(RESULTS_DIR, exist_ok=True)


test_df = pd.read_csv(TEST_FILE)

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


with open(
    os.path.join(MODEL_DIR, "selected_model.json"),
    "r"
) as f:
    model_info = json.load(f)


selected_model_name = model_info["selected_model"]
threshold = model_info["selected_threshold"]


if selected_model_name == "XGBoost":
    model_file = "xgboost.joblib"
else:
    model_file = "logistic_regression.joblib"


model = joblib.load(
    os.path.join(MODEL_DIR, model_file)
)


probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)


precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

matrix = confusion_matrix(
    y_test,
    predictions
)


results = {
    "model": selected_model_name,
    "threshold": threshold,
    "test_samples": len(test_df),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "confusion_matrix": matrix.tolist()
}


with open(
    os.path.join(RESULTS_DIR, "final_metrics.json"),
    "w"
) as f:
    json.dump(results, f, indent=4)


predictions_df = test_df.copy()

predictions_df["Failure Probability"] = probabilities
predictions_df["Prediction"] = predictions

predictions_df.to_csv(
    os.path.join(RESULTS_DIR, "final_predictions.csv"),
    index=False
)

print("\nFinal Test Evaluation")
print(f"  Model    : {selected_model_name}")
print(f"  Threshold: {threshold:.2f}")

print(f"\n  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1       : {f1:.4f}")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"  PR-AUC   : {pr_auc:.4f}")

print("\n  Confusion Matrix:")
print(f"    {matrix[0].tolist()}")
print(f"    {matrix[1].tolist()}")

print("\nFinal evaluation saved.")

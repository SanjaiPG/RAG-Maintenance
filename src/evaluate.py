import os
import json
import joblib
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix)


VALIDATION_FILE = "data/processed/validation.csv"
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

validation_df = pd.read_csv(VALIDATION_FILE)

X_val = validation_df[FEATURES]
y_val = validation_df[TARGET]

models = {
    "Logistic Regression": joblib.load(
        os.path.join(MODEL_DIR, "logistic_regression.joblib")
    ),
    "XGBoost": joblib.load(
        os.path.join(MODEL_DIR, "xgboost.joblib")
    )
}

def evaluate_model(model, X, y, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    matrix = confusion_matrix(
        y,
        predictions
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": matrix.tolist(),
        "probabilities": probabilities
    }


results = {}

for name, model in models.items():

    metrics = evaluate_model(
        model,
        X_val,
        y_val,
        threshold=0.5
    )

    results[name] = metrics

    print(f"\n{name}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC   : {metrics['pr_auc']:.4f}")

    print("  Confusion Matrix:")
    print(
        f"    {metrics['confusion_matrix'][0]}"
    )
    print(
        f"    {metrics['confusion_matrix'][1]}"
    )


comparison = []

for name, metrics in results.items():
    comparison.append({
        "Model": name,
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "ROC-AUC": metrics["roc_auc"],
        "PR-AUC": metrics["pr_auc"]
    })


comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    os.path.join(RESULTS_DIR, "model_comparison.csv"),
    index=False
)


best_model_name = comparison_df.loc[
    comparison_df["F1"].idxmax(),
    "Model"
]

best_model = models[best_model_name]


threshold_results = []

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30,
                  0.35, 0.40, 0.45, 0.50]:

    probabilities = best_model.predict_proba(X_val)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


threshold_df = pd.DataFrame(threshold_results)

best_threshold_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_threshold_row["threshold"]
)


threshold_df.to_csv(
    os.path.join(RESULTS_DIR, "threshold_comparison.csv"),
    index=False
)


evaluation_info = {
    "selected_model": best_model_name,
    "selected_threshold": best_threshold,
    "selection_metric": "F1",
    "validation_samples": len(validation_df)
}

with open(
    os.path.join(MODEL_DIR, "selected_model.json"),
    "w"
) as f:
    json.dump(evaluation_info, f, indent=4)


print("\nModel selected")
print(f"  Model    : {best_model_name}")
print(f"  Threshold: {best_threshold:.2f}")

print("\nEvaluation results saved.")

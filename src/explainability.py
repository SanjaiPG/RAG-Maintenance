import os
import joblib
import numpy as np
import pandas as pd
import shap

TEST_FILE = "data/processed/test.csv"
MODEL_FILE = "models/xgboost.joblib"
PREDICTIONS_FILE = "results/final_predictions.csv"
OUTPUT_FILE = "results/explanations.csv"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

TOP_N = 6

test_df = pd.read_csv(TEST_FILE)

model = joblib.load(MODEL_FILE)

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["classifier"]


X_test = test_df[FEATURES]

X_transformed = preprocessor.transform(X_test)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

transformed_names = preprocessor.get_feature_names_out()

original_feature_names = []

for name in transformed_names:

    if name.startswith("categorical__Type_"):
        original_feature_names.append("Type")

    elif name.startswith("numerical__"):
        original_feature_names.append(
            name.replace("numerical__", "", 1)
        )

    else:
        original_feature_names.append(name)


print("Calculating SHAP explanations...")

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_transformed)

if hasattr(shap_values, "values"):
    shap_values = shap_values.values

if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

aggregated_shap = np.zeros(
    (len(X_test), len(FEATURES))
)

for feature_index, feature in enumerate(FEATURES):

    matching_columns = [
        i
        for i, original_name in enumerate(original_feature_names)
        if original_name == feature
    ]

    aggregated_shap[:, feature_index] = (
        shap_values[:, matching_columns].sum(axis=1)
    )

if os.path.exists(PREDICTIONS_FILE):

    predictions_df = pd.read_csv(PREDICTIONS_FILE)

else:

    predictions_df = test_df.copy()

    predictions_df["Failure Probability"] = (
        model.predict_proba(X_test)[:, 1]
    )

    predictions_df["Prediction"] = (
        predictions_df["Failure Probability"] >= 0.50
    ).astype(int)

explanation_df = predictions_df.copy()

explanation_df.insert(
    0,
    "Case ID",
    range(1, len(explanation_df) + 1)
)


for rank in range(TOP_N):

    factor_names = []
    factor_values = []
    factor_directions = []

    for row in range(len(X_test)):

        row_shap = aggregated_shap[row]

        ranked_indices = np.argsort(
            np.abs(row_shap)
        )[::-1]

        feature_index = ranked_indices[rank]

        feature_name = FEATURES[feature_index]
        shap_value = row_shap[feature_index]

        factor_names.append(feature_name)
        factor_values.append(shap_value)

        if shap_value > 0:
            factor_directions.append(
                "Pushes toward failure"
            )
        elif shap_value < 0:
            factor_directions.append(
                "Pushes away from failure"
            )
        else:
            factor_directions.append(
                "No contribution"
            )

    explanation_df[
        f"Top Factor {rank + 1}"
    ] = factor_names

    explanation_df[
        f"Top Factor {rank + 1} SHAP"
    ] = factor_values

    explanation_df[
        f"Top Factor {rank + 1} Direction"
    ] = factor_directions

os.makedirs("results", exist_ok=True)

explanation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSHAP explainability completed.")

print(f"Rows explained : {len(explanation_df)}")
print(f"Output file    : {OUTPUT_FILE}")

print("\nExample explanations:")

print(
    explanation_df[
        [
            "Case ID",
            "Machine failure",
            "Failure Probability",
            "Prediction",
            "Top Factor 1",
            "Top Factor 1 SHAP",
            "Top Factor 1 Direction",
            "Top Factor 2",
            "Top Factor 2 SHAP",
            "Top Factor 3",
            "Top Factor 3 SHAP"
        ]
    ].head(5).to_string(index=False)
)

import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


TRAIN_FILE = "data/processed/train.csv"
VALIDATION_FILE = "data/processed/validation.csv"

MODEL_DIR = "models"

RANDOM_STATE = 42

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

CATEGORICAL_FEATURES = [
    "Type"
]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


os.makedirs(MODEL_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = validation_df[FEATURES]


class_counts = y_train.value_counts()

negative_count = class_counts.get(0, 0)
positive_count = class_counts.get(1, 0)

scale_pos_weight = negative_count / positive_count


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES
            )
        ]
    )


logistic_model = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)


xgb_model = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

xgb_model.fit(X_train, y_train)


logistic_probabilities = logistic_model.predict_proba(X_val)[:, 1]
xgb_probabilities = xgb_model.predict_proba(X_val)[:, 1]


joblib.dump(
    logistic_model,
    os.path.join(MODEL_DIR, "logistic_regression.joblib")
)

joblib.dump(
    xgb_model,
    os.path.join(MODEL_DIR, "xgboost.joblib")
)


training_info = {
    "features": FEATURES,
    "target": TARGET,
    "random_state": RANDOM_STATE,
    "train_samples": len(train_df),
    "validation_samples": len(validation_df),
    "negative_samples": int(negative_count),
    "positive_samples": int(positive_count),
    "xgboost_scale_pos_weight": float(scale_pos_weight)
}

with open(
    os.path.join(MODEL_DIR, "training_info.json"),
    "w"
) as f:
    json.dump(training_info, f, indent=4)


print("\nLogistic Regression")
print(f"  Min : {logistic_probabilities.min():.4f}")
print(f"  Max : {logistic_probabilities.max():.4f}")
print(f"  Mean: {logistic_probabilities.mean():.4f}")

print("\nXGBoost")
print(f"  Min : {xgb_probabilities.min():.4f}")
print(f"  Max : {xgb_probabilities.max():.4f}")
print(f"  Mean: {xgb_probabilities.mean():.4f}")

print("\nModels saved successfully.")

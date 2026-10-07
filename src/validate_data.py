import os
import pandas as pd


DATA_DIR = "data/processed"

FILES = {
    "Train": "train.csv",
    "Validation": "validation.csv",
    "Test": "test.csv"
}

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

EXPECTED_COLUMNS = FEATURES + [TARGET]


def validate_dataset(name, filepath):
    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET")
    print("=" * 70)

    df = pd.read_csv(filepath)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_columns = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_columns:
        print(f"\nERROR: Missing columns: {missing_columns}")

    if extra_columns:
        print(f"\nWARNING: Extra columns: {extra_columns}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nMachine failure distribution:")
    print(df[TARGET].value_counts())

    print("\nMachine failure percentage:")
    print(
        (df[TARGET].value_counts(normalize=True) * 100)
        .round(2)
    )

    numerical_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    print("\nNumerical feature ranges:")

    for column in numerical_features:
        print(
            f"{column}: "
            f"min={df[column].min()}, "
            f"max={df[column].max()}, "
            f"mean={df[column].mean():.2f}"
        )

    print("\nType values:")
    print(df["Type"].value_counts())

    invalid_targets = set(df[TARGET].unique()) - {0, 1}

    if invalid_targets:
        print(f"\nERROR: Invalid target values: {invalid_targets}")
    else:
        print("\nTarget validation: PASS")

    return df


def main():

    print("MAINT-RAG DATA VALIDATION")

    datasets = {}

    for name, filename in FILES.items():

        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            print(f"\nERROR: File not found: {filepath}")
            continue

        datasets[name] = validate_dataset(name, filepath)

    print("\n" + "=" * 70)
    print("SPLIT CONSISTENCY CHECK")
    print("=" * 70)

    if len(datasets) == 3:

        train_columns = list(datasets["Train"].columns)
        val_columns = list(datasets["Validation"].columns)
        test_columns = list(datasets["Test"].columns)

        if train_columns == val_columns == test_columns:
            print("All splits have identical columns: PASS")
        else:
            print("Column consistency: FAIL")

        for name, df in datasets.items():
            if len(df) > 0:
                print(f"{name} is non-empty: PASS")
            else:
                print(f"{name} is empty: FAIL")


if __name__ == "__main__":
    main()

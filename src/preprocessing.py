import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/raw/ai4i2020.csv"
OUTPUT_DIR = "data/processed"

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

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape} \n")

print("Columns in dataset:")
print(df.columns.tolist())
print()

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

for feature in FEATURES:
    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found.")

data = df[FEATURES + [TARGET]].copy()

print("Selected columns:")
print(data.columns.tolist())
print()

print("=" * 60)
print("Overall class distribution")
print("=" * 60)

print(data[TARGET].value_counts())
print()

print("Overall class percentage:")
print(data[TARGET].value_counts(normalize=True) * 100)
print()

X = data[FEATURES]
y = data[TARGET]


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)


train = X_train.copy()
train[TARGET] = y_train

validation = X_val.copy()
validation[TARGET] = y_val

test = X_test.copy()
test[TARGET] = y_test

os.makedirs(OUTPUT_DIR, exist_ok=True)

train.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

validation.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False
)

test.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

print("=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(f"Training set:    {len(train)} rows")
print(f"Validation set:  {len(validation)} rows")
print(f"Test set:        {len(test)} rows")
print()

print("=" * 60)
print("CLASS DISTRIBUTION AFTER SPLIT")
print("=" * 60)

for name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test)
]:
    print(f"\n{name}:")
    print(dataset[TARGET].value_counts())
    print(
        dataset[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

print(f"Files saved in: {OUTPUT_DIR}/")
print("  - train.csv")
print("  - validation.csv")
print("  - test.csv")

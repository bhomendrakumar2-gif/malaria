

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset and model paths
DATA_PATH = "data/Malaria Diseases dataset - .csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "malaria_model.pkl")

# Actual columns in your dataset
NUMERIC_FEATURES = [
    "Age",
    "Hemoglobin(Hb%)",
    "Total WBC count(/cumm)",
    "Neutrophils",
    "Lymphocytes",
    "Total Cir.Eosinophils",
    "HTC/PCV(%)",
    "MCH(pg)",
    "MCHC(g/dl)",
    "RDW-CV(%)",
    "Platelet Count"
]

CATEGORICAL_FEATURES = ["Sex"]
TARGET = "Result"

def main():
    print("Loading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Clean column names and target labels
    df.columns = df.columns.str.strip()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Keep only positive and negative records
    df = df[df[TARGET].isin(["positive", "negative"])].copy()

    # Convert numeric columns
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    print("Total records:", len(df))
    print("\nClass distribution:")
    print(y.value_counts())

    # Preprocessing
    preprocessing = ColumnTransformer(
        transformers=[
            ("sex", OneHotEncoder(handle_unknown="ignore"),
             CATEGORICAL_FEATURES)
        ],
        remainder="passthrough"
    )

    # Machine Learning model
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("\nModel saved successfully at:", MODEL_PATH)


if __name__ == "__main__":
    main()

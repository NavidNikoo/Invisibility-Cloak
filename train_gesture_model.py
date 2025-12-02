import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

CSV_PATH = "gesture_data.csv"
MODEL_PATH = "peace_model.pkl"

def main():
    # --------- Load data ----------
    print(f"[INFO] Loading data from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    # label column is 'label'; everything else is features
    y = df["label"].values
    X = df.drop(columns=["label"]).values

    print(f"[INFO] Dataset shape: {X.shape}, labels: {np.unique(y, return_counts=True)}")

    # --------- Train / test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # keep class balance in both splits
    )

    # --------- Build model pipeline ----------
    # Standardize features + Logistic Regression classifier
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200, solver="lbfgs")
    )

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # --------- Evaluation ----------
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # --------- Save model ----------
    joblib.dump(model, MODEL_PATH)
    print(f"\n[INFO] Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

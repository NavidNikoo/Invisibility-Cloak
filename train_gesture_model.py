import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

CSV_PATH = "gesture_data.csv"
MODEL_PATH = "gesture_model.pkl"  # updated name


def main():
    print(f"[INFO] Loading data from {CSV_PATH}...")

    # 🔹 No header in the CSV – first row is already data
    df = pd.read_csv(CSV_PATH, header=None)
    print("[DEBUG] Shape:", df.shape)

    # column 0 = label, columns 1..end = features
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    print(f"[INFO] Dataset shape: {X.shape}, labels: {np.unique(y, return_counts=True)}")

    # stratified train/test because we have multiple gestures
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # pipeline: scaling + multi-class classifier
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=400,
            solver="lbfgs",
            multi_class="auto"
        )
    )

    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\n[INFO] Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

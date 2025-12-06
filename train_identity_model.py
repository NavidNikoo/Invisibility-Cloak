# train_identity_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

CSV_PATH = "identity_data.csv"
MODEL_PATH = "identity_model.pkl"

def main():
    print(f"[INFO] Loading identity data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # label column is 'label'; everything else are features
    y = df["label"].values
    X = df.drop(columns=["label"]).values

    print(f"[INFO] Dataset shape: {X.shape}, labels: {np.unique(y, return_counts=True)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=400,
            solver="lbfgs",
            multi_class="auto"
        )
    )

    print("[INFO] Training identity model (navid vs other)...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Classification Report (Identity) ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix (Identity) ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\n[INFO] Saved identity model to {MODEL_PATH}")

if __name__ == "__main__":
    main()

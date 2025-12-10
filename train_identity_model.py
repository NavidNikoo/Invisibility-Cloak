# train_identity_model.py
#
# This script trains the **identity classifier**, which distinguishes
# between "navid" and "other" faces using the vectors extracted by
# face_to_vector() in identity_features.py.
#
# Workflow:
#   1. Load identity_data.csv (collected using collect_identity_data.py)
#   2. Separate labels ("navid"/"other") from the numerical face features
#   3. Split into training and test sets
#   4. Train a machine-learning model (StandardScaler + LogisticRegression)
#   5. Evaluate the model performance
#   6. Save the trained classifier as identity_model.pkl
#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # splits dataset
from sklearn.preprocessing import StandardScaler  # normalizes features
from sklearn.pipeline import make_pipeline  # creates a model pipeline
from sklearn.linear_model import LogisticRegression  # classifier algorithm
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # saves the trained model to disk

# Path to the CSV dataset containing identity samples collected earlier
CSV_PATH = "identity_data.csv"

# Model file that cloak.py will load for determining who is Navid
MODEL_PATH = "identity_model.pkl"


def main():
    print(f"[INFO] Loading identity data from {CSV_PATH}...")

    # ----------------------------------------------------------------------
    # 1. LOAD IDENTITY DATASET
    # ----------------------------------------------------------------------
    # identity_data.csv was created by collect_identity_data.py and contains:
    #   label,f0,f1,f2,...
    #
    #   label → 'navid' or 'other'
    #   f0...fN → normalized facial feature coordinates (selected FaceMesh points)
    df = pd.read_csv(CSV_PATH)

    # ----------------------------------------------------------------------
    # 2. SEPARATE LABELS FROM FEATURES
    # ----------------------------------------------------------------------
    # Labels are in the column named "label"
    y = df["label"].values  # array like ['navid','other','navid',...]

    # All remaining columns are numeric facial features
    # drop() removes the label column and keeps only feature values
    X = df.drop(columns=["label"]).values

    print(f"[INFO] Dataset shape: {X.shape}, labels: {np.unique(y, return_counts=True)}")
    # Example:
    #   Dataset shape: (869, 20) → 869 samples, 20 facial features per sample
    #   Labels: (['navid','other'], [400, 469])

    # ----------------------------------------------------------------------
    # 3. TRAIN-TEST SPLIT
    # ----------------------------------------------------------------------
    # test_size=0.2 → 20% goes to testing, 80% stays for training
    # stratify=y → keeps class proportions balanced (important!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,  # ensures reproducibility
        stratify=y  # preserves distribution of 'navid' vs 'other'
    )

    # ----------------------------------------------------------------------
    # 4. DEFINE THE MODEL PIPELINE
    # ----------------------------------------------------------------------
    # The model is a pipeline:
    #   StandardScaler → normalizes features to similar ranges
    #   LogisticRegression → learns a decision boundary between (navid, other)
    #
    # Logistic Regression works well because the facial features extracted
    # from face_to_vector() are numerical and linearly separable.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=400,  # more iterations = reliable convergence
            solver="lbfgs",  # good general-purpose optimizer
            multi_class="auto"  # automatically handles binary classification
        )
    )

    print("[INFO] Training identity model (navid vs other)...")

    # ----------------------------------------------------------------------
    # 5. TRAIN THE MODEL
    # ----------------------------------------------------------------------
    # model.fit learns the relationship:
    #   input facial vector → 'navid' or 'other'
    model.fit(X_train, y_train)

    # ----------------------------------------------------------------------
    # 6. TEST THE MODEL
    # ----------------------------------------------------------------------
    # Predict labels for unseen (test) data
    y_pred = model.predict(X_test)

    print("\n=== Classification Report (Identity) ===")
    # Shows precision, recall, f1-score for each class
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix (Identity) ===")
    # Rows: true labels, Columns: predicted labels
    # Helps visualize which class gets confused with which
    print(confusion_matrix(y_test, y_pred))

    # ----------------------------------------------------------------------
    # 7. SAVE THE TRAINED MODEL FOR USE IN cloak.py
    # ----------------------------------------------------------------------
    # After saving, cloak.py will load identity_model.pkl and do:
    #   identity_model.predict([face_vector]) → "navid" or "other"
    joblib.dump(model, MODEL_PATH)
    print(f"\n[INFO] Saved identity model to {MODEL_PATH}")


# Standard execution guard:
if __name__ == "__main__":
    main()

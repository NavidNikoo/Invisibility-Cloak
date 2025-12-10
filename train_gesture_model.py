# train_gesture_model.py
#
# This script trains the **hand gesture classifier** used by cloak.py.
# It:
#   1. Loads the gesture dataset from gesture_data.csv
#   2. Splits it into training and test sets
#   3. Builds a pipeline: StandardScaler + LogisticRegression
#   4. Trains the model and prints evaluation metrics
#   5. Saves the trained pipeline to gesture_model.pkl for later use

import pandas as pd          # for loading and manipulating the CSV data
import numpy as np           # for numerical operations, arrays, etc.
from sklearn.model_selection import train_test_split   # to split data into train/test sets
from sklearn.preprocessing import StandardScaler       # to normalize feature scales
from sklearn.pipeline import make_pipeline             # to chain scaler + model together
from sklearn.linear_model import LogisticRegression    # our classifier (multi-class)
from sklearn.metrics import classification_report, confusion_matrix  # evaluation metrics
import joblib                    # for saving/loading the trained model pipeline

# Path to the dataset created by collect_gesture_data.py
CSV_PATH = "gesture_data.csv"

# Path where the trained gesture classifier will be saved.
# cloak.py will load this file to recognize gestures in real time.
MODEL_PATH = "gesture_model.pkl"


def main():
    print(f"[INFO] Loading data from {CSV_PATH}...")

    # ------------------------------------------------------------------
    # 1. LOAD THE DATA
    # ------------------------------------------------------------------
    # NOTE: This line uses header=None, which means:
    #   - pandas will treat EVERY row as data
    #   - there is no header row with column names
    #
    # Assumption here:
    #   - Column 0: gesture label (e.g., "peace", "thumbs_up", etc.)
    #   - Columns 1..end: numeric features (63 hand landmark coordinates)
    df = pd.read_csv(CSV_PATH, header=None)
    print("[DEBUG] Shape:", df.shape)  # (num_samples, num_columns)

    # ------------------------------------------------------------------
    # 2. SPLIT INTO FEATURES (X) AND LABELS (y)
    # ------------------------------------------------------------------
    # y = labels → first column (index 0)
    y = df.iloc[:, 0].values

    # X = features → all remaining columns (1 to end)
    # These are the 63 numbers from MediaPipe landmarks (x0,y0,z0,...)
    X = df.iloc[:, 1:].values

    print(f"[INFO] Dataset shape: {X.shape}, labels: {np.unique(y, return_counts=True)}")
    # Example:
    #   X.shape = (4142, 63)  # 4142 samples, 63 features each
    #   labels might be: (['open_palm', 'other', 'peace', 'thumbs_up'], [count0,...])

    # ------------------------------------------------------------------
    # 3. TRAIN/TEST SPLIT (STRATIFIED)
    # ------------------------------------------------------------------
    # train_test_split:
    #   - test_size=0.2 → 20% of data held out for testing
    #   - random_state=42 → reproducible split
    #   - stratify=y → keeps the class distribution similar in train and test
    #
    # Stratification is important for multi-class:
    #   - ensures each gesture (peace, thumbs_up, etc.) is represented in both
    #     the training set and the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------------------------
    # 4. BUILD A PIPELINE: SCALER + LOGISTIC REGRESSION
    # ------------------------------------------------------------------
    # Why a pipeline?
    #   - StandardScaler: normalizes each feature to have 0 mean and unit variance
    #     so that all 63 features are on similar scales.
    #   - LogisticRegression: multi-class linear classifier that works well with
    #     scaled numeric features.
    #
    # Using make_pipeline lets us treat this as ONE model object:
    #   model.fit(...)  → scales X_train internally, then fits the classifier
    #   model.predict(...) → scales features in the same way, then predicts.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=400,      # allow up to 400 iterations for convergence
            solver="lbfgs",    # robust optimizer that supports multi-class
            multi_class="auto" # let sklearn choose OvR or multinomial automatically
        )
    )

    print("[INFO] Training model...")

    # ------------------------------------------------------------------
    # 5. TRAIN THE MODEL
    # ------------------------------------------------------------------
    # This line:
    #   - fits the StandardScaler on X_train
    #   - transforms X_train
    #   - trains LogisticRegression on the scaled data
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. EVALUATE ON HELD-OUT TEST SET
    # ------------------------------------------------------------------
    # Predict labels for the test set:
    #   - model automatically scales X_test the same way as X_train
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    # classification_report shows:
    #   - precision: how many predicted samples for a class are correct
    #   - recall: how many true samples of a class are correctly found
    #   - f1-score: harmonic mean of precision and recall
    #   - support: number of true samples for each class
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    # confusion_matrix shows:
    #   - rows: true classes
    #   - columns: predicted classes
    #   - each cell [i,j]: how many samples of true class i were predicted as j
    print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------------
    # 7. SAVE THE TRAINED MODEL PIPELINE
    # ------------------------------------------------------------------
    # joblib.dump saves the *entire pipeline* (scaler + logistic regression)
    # to disk, so cloak.py can load it and do:
    #   gesture_model.predict([feature_vector]) → "peace" / "open_palm" / etc.
    joblib.dump(model, MODEL_PATH)
    print(f"\n[INFO] Saved trained model to {MODEL_PATH}")


# Standard Python pattern:
# Only run main() if this file is executed directly.
# If someone imports this file as a module, main() will NOT run automatically.
if __name__ == "__main__":
    main()

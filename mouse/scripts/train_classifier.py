import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def get_project_root() -> str:
    """Return project root assuming this file is inside scripts/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    project_root = get_project_root()
    csv_path = os.path.join(project_root, "landmarks_dataset.csv")
    models_dir = os.path.join(project_root, "models")
    model_path = os.path.join(models_dir, "grip_model.pkl")

    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("[ERROR] CSV is empty. Please run extract_landmarks.py first.")
        return

    required_columns = {"label", "filename"}
    if not required_columns.issubset(df.columns):
        print("[ERROR] CSV format is invalid. Required columns: label, filename")
        return

    feature_columns = [col for col in df.columns if col not in ["label", "filename"]]

    if len(feature_columns) != 63:
        print(f"[WARNING] Expected 63 feature columns, but got {len(feature_columns)}")

    X = df[feature_columns]
    y = df["label"]

    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(y.value_counts())
    print()

    unique_labels = y.unique()
    if len(unique_labels) < 2:
        print("[ERROR] Need at least 2 classes to train a classifier.")
        return

    min_class_count = y.value_counts().min()

    # If dataset is too small, stratified split may fail.
    if len(df) < 10 or min_class_count < 2:
        print("[WARNING] Dataset is very small. Training on full dataset without test split.")
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        clf.fit(X, y)
        joblib.dump(clf, model_path)
        print(f"Saved trained model to: {model_path}")
        print("No evaluation was performed because the dataset is too small.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    except ValueError as e:
        print(f"[WARNING] Stratified split failed: {e}")
        print("Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:")
    print(accuracy_score(y_test, y_pred))
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print()

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    joblib.dump(clf, model_path)
    print(f"Saved trained model to: {model_path}")


if __name__ == "__main__":
    main()
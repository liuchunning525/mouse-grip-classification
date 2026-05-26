import argparse
import json
import os
from collections import Counter

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import LabelEncoder

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    feature_names = sorted({
        k
        for sample in data
        for k in sample.get("features", {}).keys()
    })

    X, y, users, mice, conditions = [], [], [], [], []
    for sample in data:
        features = sample.get("features", {})
        X.append([float(features.get(k, 0.0)) for k in feature_names])
        y.append(sample["label"])
        users.append(sample.get("user_id", "unknown_user"))
        mice.append(sample.get("mouse_id", "unknown_mouse"))
        conditions.append(sample.get("condition", "unknown"))

    return (
        np.array(X, dtype=np.float32),
        np.array(y),
        np.array(users),
        np.array(mice),
        np.array(conditions),
        feature_names,
    )

def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2
    )

def evaluate_split(X_train, X_test, y_train, y_test, le):
    model = make_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    y_test_label = le.inverse_transform(y_test)
    pred_label = le.inverse_transform(pred)

    acc = accuracy_score(y_test, pred)
    report = classification_report(
        y_test_label,
        pred_label,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_test_label, pred_label, labels=le.classes_)

    return acc, report, cm, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/datasets/training_dataset_3mouse_v2_relative.json")
    parser.add_argument("--mode", choices=["random", "group_user", "leave_one_user", "leave_one_mouse"], default="leave_one_user")
    parser.add_argument("--model_out", default="models/trained/grip_model_3mouse.pkl")
    parser.add_argument("--report_out", default="data/datasets/train_report_3mouse.json")
    args = parser.parse_args()

    X, y, users, mice, conditions, feature_names = load_dataset(args.input)

    if len(X) == 0:
        print("No data found.")
        return

    print("=== Dataset ===")
    print("Samples:", len(X))
    print("Features:", len(feature_names))
    print("Labels:", dict(Counter(y)))
    print("Users:", sorted(set(users)))
    print("Mice:", sorted(set(mice)))
    print("Conditions:", sorted(set(conditions)))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    results = []
    final_model = None

    if args.mode == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
        )
        acc, report, cm, final_model = evaluate_split(X_train, X_test, y_train, y_test, le)
        results.append({
            "mode": "random",
            "accuracy": float(acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        })

        print("\n=== Random Split Evaluation ===")
        print("Accuracy:", round(acc, 4))
        print("Labels:", list(le.classes_))
        print(cm)

    elif args.mode == "group_user":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y_enc, groups=users))
        acc, report, cm, final_model = evaluate_split(
            X[train_idx], X[test_idx], y_enc[train_idx], y_enc[test_idx], le
        )
        results.append({
            "mode": "group_user",
            "accuracy": float(acc),
            "train_users": sorted(set(users[train_idx])),
            "test_users": sorted(set(users[test_idx])),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        })

        print("\n=== Group By User Evaluation ===")
        print("Train users:", sorted(set(users[train_idx])))
        print("Test users:", sorted(set(users[test_idx])))
        print("Accuracy:", round(acc, 4))
        print("Labels:", list(le.classes_))
        print(cm)

    elif args.mode == "leave_one_user":
        logo = LeaveOneGroupOut()
        all_true = []
        all_pred = []

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_enc, groups=users)):
            test_user = sorted(set(users[test_idx]))[0]
            model = make_model()
            model.fit(X[train_idx], y_enc[train_idx])
            pred = model.predict(X[test_idx])
            acc = accuracy_score(y_enc[test_idx], pred)

            all_true.extend(y_enc[test_idx].tolist())
            all_pred.extend(pred.tolist())

            results.append({
                "mode": "leave_one_user",
                "fold": fold,
                "test_user": test_user,
                "accuracy": float(acc),
                "n_test": int(len(test_idx))
            })

            print(f"Fold {fold} test_user={test_user} accuracy={acc:.4f} n={len(test_idx)}")

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        overall = accuracy_score(all_true, all_pred)
        cm = confusion_matrix(le.inverse_transform(all_true), le.inverse_transform(all_pred), labels=le.classes_)
        report = classification_report(
            le.inverse_transform(all_true),
            le.inverse_transform(all_pred),
            output_dict=True,
            zero_division=0
        )

        final_model = make_model()
        final_model.fit(X, y_enc)

        print("\n=== Leave One User Overall ===")
        print("Accuracy:", round(overall, 4))
        print("Labels:", list(le.classes_))
        print(cm)

        results.append({
            "mode": "leave_one_user_overall",
            "accuracy": float(overall),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        })

    elif args.mode == "leave_one_mouse":
        logo = LeaveOneGroupOut()
        all_true = []
        all_pred = []

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_enc, groups=mice)):
            test_mouse = sorted(set(mice[test_idx]))[0]
            model = make_model()
            model.fit(X[train_idx], y_enc[train_idx])
            pred = model.predict(X[test_idx])
            acc = accuracy_score(y_enc[test_idx], pred)

            all_true.extend(y_enc[test_idx].tolist())
            all_pred.extend(pred.tolist())

            results.append({
                "mode": "leave_one_mouse",
                "fold": fold,
                "test_mouse": test_mouse,
                "accuracy": float(acc),
                "n_test": int(len(test_idx))
            })

            print(f"Fold {fold} test_mouse={test_mouse} accuracy={acc:.4f} n={len(test_idx)}")

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        overall = accuracy_score(all_true, all_pred)
        cm = confusion_matrix(le.inverse_transform(all_true), le.inverse_transform(all_pred), labels=le.classes_)
        report = classification_report(
            le.inverse_transform(all_true),
            le.inverse_transform(all_pred),
            output_dict=True,
            zero_division=0
        )

        final_model = make_model()
        final_model.fit(X, y_enc)

        print("\n=== Leave One Mouse Overall ===")
        print("Accuracy:", round(overall, 4))
        print("Labels:", list(le.classes_))
        print(cm)

        results.append({
            "mode": "leave_one_mouse_overall",
            "accuracy": float(overall),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        })

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    os.makedirs("models/trained", exist_ok=True)

    joblib.dump(final_model, args.model_out)

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "labels": list(le.classes_),
            "feature_count": len(feature_names),
            "label_counts": dict(Counter(y)),
            "user_counts": dict(Counter(users)),
            "mouse_counts": dict(Counter(mice)),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    with open("models/trained/label_encoder_3mouse.json", "w", encoding="utf-8") as f:
        json.dump(list(le.classes_), f, ensure_ascii=False, indent=2)

    with open("models/trained/feature_columns_3mouse.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(args.model_out)
    print(args.report_out)

if __name__ == "__main__":
    main()

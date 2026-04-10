import json
import argparse
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = []
    y = []

    for sample in data:
        if sample["label"] == "unknown":
            continue

        features = sample["features"]

        # 转成固定顺序向量
        vec = list(features.values())

        if len(vec) == 0:
            continue

        X.append(vec)
        y.append(sample["label"])

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="features.json")
    args = parser.parse_args()

    print("Loading dataset...")
    X, y = load_dataset(args.input)

    if len(X) == 0:
        print("No valid data found.")
        return

    print(f"Total samples: {len(X)}")

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 模型（你可以换）
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    print("Training model...")
    model.fit(X_train, y_train)


    joblib.dump(model, "grip_model.pkl")
    joblib.dump(le, "label_encoder.pkl")

    # 预测
    y_pred = model.predict(X_test)

    # 还原标签
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    # 评估
    acc = accuracy_score(y_test_labels, y_pred_labels)

    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels))


if __name__ == "__main__":
    main()
"""
Scenario B — XGBoost multiclass classifier
Gradient boosted trees on 23 flow features.
Expected to outperform RF and CNN on this tabular data.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_utils import load_data, clean_data, FEATURE_COLS, LABEL_COL

DATA_DIR = "data/scenario_b"
OUTPUT_DIR = "outputs/scenario_b"


def train_xgb(data):
    X = data[FEATURE_COLS].values
    y_raw = data[LABEL_COL].str.strip().values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nXGBoost Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return model, le, X_test, y_test, y_pred, acc


def plot_confusion_matrix(y_test, y_pred, le, out_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title("Confusion Matrix — XGBoost, Scenario B (14-class)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved: {out_path}")


def plot_feature_importance(model, out_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(FEATURE_COLS)), importances[indices])
    plt.xticks(range(len(FEATURE_COLS)), [FEATURE_COLS[i] for i in indices], rotation=45, ha="right")
    plt.title("Feature Importances — XGBoost, Scenario B")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    data = load_data(DATA_DIR)
    data = clean_data(data)
    model, le, X_test, y_test, y_pred, acc = train_xgb(data)
    plot_confusion_matrix(y_test, y_pred, le, f"{OUTPUT_DIR}/confusion_matrix_xgb.png")
    plot_feature_importance(model, f"{OUTPUT_DIR}/feature_importance_xgb.png")
    joblib.dump({"model": model, "label_encoder": le}, f"{OUTPUT_DIR}/xgb_model.pkl")
    print("Model saved.")

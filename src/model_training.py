# src/model_training.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

# Guard XGBoost import so script can run without it
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception as e:
    print(f"[WARN] XGBoost unavailable: {e}")
    XGB_AVAILABLE = False

from data_prep import ensure_data_exists, load_data
from feature_engineering import (
    train_test_split_data, get_numeric_columns, scale_numeric, DEFAULT_TARGET
)

# File-relative output directories
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, '..', 'results'))
MODELS_DIR = os.path.normpath(os.path.join(HERE, '..', 'models'))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def evaluate_and_report(y_true, y_pred, y_pred_proba, model_name: str) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }
    print(f"\n{model_name} Performance:")
    for k, v in metrics.items():
        print(f"- {k}: {v:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join(RESULTS_DIR, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_pred_proba)
    plt.title(f'{model_name} - ROC Curve')
    roc_path = os.path.join(RESULTS_DIR, f'{model_name.lower().replace(" ", "_")}_roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    return {**metrics, "confusion_matrix_path": cm_path, "roc_curve_path": roc_path}

def train_logistic_regression(X_train, y_train, X_test, y_test) -> dict:
    lr = LogisticRegression(max_iter=500, n_jobs=None)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    metrics = evaluate_and_report(y_test, y_pred, y_proba, "Logistic Regression")
    metrics["y_pred"] = y_pred
    metrics["y_proba"] = y_proba
    model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
    joblib.dump(lr, model_path)
    metrics["model_path"] = model_path
    return metrics

def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    metrics = evaluate_and_report(y_test, y_pred, y_proba, "XGBoost")
    metrics["y_pred"] = y_pred
    metrics["y_proba"] = y_proba

    # Feature importance
    importance = xgb.feature_importances_
    cols = X_train.columns
    imp = pd.DataFrame({"feature": cols, "importance": importance}).sort_values("importance", ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="importance", y="feature", data=imp.head(15), orient='h', palette="viridis")
    plt.title("XGBoost - Top Feature Importances")
    plt.tight_layout()
    fi_path = os.path.join(RESULTS_DIR, 'xgboost_feature_importance.png')
    plt.savefig(fi_path)
    plt.close()

    model_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    joblib.dump(xgb, model_path)
    metrics["model_path"] = model_path
    metrics["feature_importance_path"] = fi_path
    return metrics

def train_hgb(X_train, y_train, X_test, y_test) -> dict:
    hgb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, random_state=42)
    hgb.fit(X_train, y_train)
    y_pred = hgb.predict(X_test)
    y_proba = hgb.predict_proba(X_test)[:, 1]
    metrics = evaluate_and_report(y_test, y_pred, y_proba, "HistGradientBoosting")
    metrics["y_pred"] = y_pred
    metrics["y_proba"] = y_proba
    return metrics

def plot_roc_comparison(y_true, model_curves, out_path):
    plt.figure(figsize=(6.5, 5.5))
    for name, y_proba in model_curves:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_pr_comparison(y_true, model_curves, out_path):
    plt.figure(figsize=(6.5, 5.5))
    for name, y_proba in model_curves:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Comparison")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusions_side_by_side(y_true, model_preds, out_path):
    fig, axes = plt.subplots(1, len(model_preds), figsize=(11, 4))
    if len(model_preds) == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, model_preds):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_training_pipeline():
    ensure_data_exists()
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split_data(df, target=DEFAULT_TARGET)

    numeric_cols = get_numeric_columns(df, exclude=[DEFAULT_TARGET])
    X_train_s, X_test_s, scaler = scale_numeric(X_train, X_test, numeric_cols)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'feature_scaler.pkl'))

    print("\nTraining Logistic Regression...")
    lr_metrics = train_logistic_regression(X_train_s, y_train, X_test_s, y_test)

    xgb_or_fallback_label = "XGBoost"
    xgb_metrics = None
    if XGB_AVAILABLE:
        try:
            print("\nTraining XGBoost...")
            xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)  # raw scale for XGB
        except Exception as e:
            print(f"[WARN] XGBoost training failed: {e}\nFalling back to HistGradientBoosting.")
    if xgb_metrics is None:
        print("\nTraining HistGradientBoosting (fallback)...")
        xgb_metrics = train_hgb(X_train_s, y_train, X_test_s, y_test)
        xgb_or_fallback_label = "HistGradientBoosting"

    comparison = pd.DataFrame([lr_metrics, xgb_metrics], index=["Logistic Regression", xgb_or_fallback_label])
    comp_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    comparison[["accuracy", "precision", "recall", "f1", "roc_auc"]].to_csv(comp_path)
    print(f"\nSaved model comparison to: {comp_path}")

    # Combined plots
    roc_path = os.path.join(RESULTS_DIR, 'roc_comparison.png')
    pr_path = os.path.join(RESULTS_DIR, 'pr_comparison.png')
    cm_side_path = os.path.join(RESULTS_DIR, 'confusion_matrices_side_by_side.png')

    plot_roc_comparison(
        y_test,
        [("Logistic Regression", lr_metrics["y_proba"]), (xgb_or_fallback_label, xgb_metrics["y_proba"])],
        roc_path
    )
    plot_pr_comparison(
        y_test,
        [("Logistic Regression", lr_metrics["y_proba"]), (xgb_or_fallback_label, xgb_metrics["y_proba"])],
        pr_path
    )
    plot_confusions_side_by_side(
        y_test,
        [("Logistic Regression", lr_metrics["y_pred"]), (xgb_or_fallback_label, xgb_metrics["y_pred"])],
        cm_side_path
    )

    print(f"Saved combined ROC: {roc_path}")
    print(f"Saved combined PR: {pr_path}")
    print(f"Saved side-by-side confusions: {cm_side_path}")

if __name__ == "__main__":
    run_training_pipeline()
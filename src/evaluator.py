"""Evaluation helpers for anomaly detection."""
from __future__ import annotations
import numpy as np
from typing import Dict, Any

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def _manual_precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, tn, fn

def evaluate(y_true, y_pred, scores=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if HAS_SKLEARN:
        metrics = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "n_true_anomalies": int(y_true.sum()),
            "n_predicted_anomalies": int(y_pred.sum()),
            "n_total": len(y_true),
        }
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
        if scores is not None and len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            except ValueError:
                metrics["roc_auc"] = None
    else:
        precision, recall, f1, tp, fp, tn, fn = _manual_precision_recall_f1(y_true, y_pred)
        metrics = {"precision": precision, "recall": recall, "f1": f1,
                   "n_true_anomalies": int(y_true.sum()), "n_predicted_anomalies": int(y_pred.sum()),
                   "n_total": len(y_true), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "roc_auc": None}
    return metrics

def print_report(metrics, method=""):
    title = f" Evaluation Report {('— ' + method) if method else ''} "
    print("\n" + "=" * 50)
    print(title.center(50, "="))
    print("=" * 50)
    print(f"  Total flows          : {metrics.get('n_total', 'N/A')}")
    print(f"  True anomalies       : {metrics.get('n_true_anomalies', 'N/A')}")
    print(f"  Predicted anomalies  : {metrics.get('n_predicted_anomalies', 'N/A')}")
    print("-" * 50)
    print(f"  Precision            : {metrics.get('precision', 0):.4f}")
    print(f"  Recall               : {metrics.get('recall', 0):.4f}")
    print(f"  F1-score             : {metrics.get('f1', 0):.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC              : {metrics['roc_auc']:.4f}")
    if "tp" in metrics:
        print("-" * 50)
        print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
    print("=" * 50 + "\n")

"""Anomaly detection engines: Statistical, Isolation Forest, Rule-based, Hybrid."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from .features import prepare_features

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    IsolationForest = None
    StandardScaler = None
    joblib = None

@dataclass
class DetectionResult:
    labels: np.ndarray
    scores: np.ndarray
    method: str
    details: Dict[str, Any] = field(default_factory=dict)
    @property
    def n_anomalies(self) -> int:
        return int(self.labels.sum())

class StatisticalDetector:
    def __init__(self, threshold: float = 4.5, use_mad: bool = True, min_features: int = 2):
        self.threshold = threshold
        self.use_mad = use_mad
        self.min_features = min_features
        self.medians_ = self.mads_ = self.means_ = self.stds_ = None
        self.feature_names_: List[str] = []
    def fit(self, df, feature_cols=None):
        X, names = prepare_features(df, feature_cols, log_transform=True)
        self.feature_names_ = names
        if self.use_mad:
            self.medians_ = X.median()
            self.mads_ = (X - self.medians_).abs().median().replace(0, 1e-9)
        else:
            self.means_ = X.mean()
            self.stds_ = X.std().replace(0, 1e-9)
        return self
    def predict(self, df):
        X, _ = prepare_features(df, self.feature_names_, log_transform=True)
        abs_z = (0.6745 * (X - self.medians_) / self.mads_).abs() if self.use_mad else ((X - self.means_) / self.stds_).abs()
        scores = abs_z.max(axis=1).values
        labels = ((abs_z > self.threshold).sum(axis=1).values >= self.min_features).astype(int)
        return DetectionResult(labels=labels, scores=scores, method="statistical",
            details={"threshold": self.threshold, "min_features": self.min_features})
    def fit_predict(self, df, feature_cols=None):
        return self.fit(df, feature_cols).predict(df)

class IsolationForestDetector:
    def __init__(self, contamination=0.05, n_estimators=200, random_state=42, n_jobs=-1):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required. pip install scikit-learn")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_ = self.scaler_ = None
        self.feature_names_: List[str] = []
    def fit(self, df, feature_cols=None):
        X, names = prepare_features(df, feature_cols, log_transform=True)
        self.feature_names_ = names
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = IsolationForest(contamination=self.contamination, n_estimators=self.n_estimators,
            random_state=self.random_state, n_jobs=self.n_jobs)
        self.model_.fit(X_scaled)
        return self
    def predict(self, df):
        X, _ = prepare_features(df, self.feature_names_, log_transform=True)
        X_scaled = self.scaler_.transform(X)
        labels = (self.model_.predict(X_scaled) == -1).astype(int)
        scores = -self.model_.decision_function(X_scaled)
        return DetectionResult(labels=labels, scores=scores, method="isolation_forest",
            details={"contamination": self.contamination})
    def fit_predict(self, df, feature_cols=None):
        return self.fit(df, feature_cols).predict(df)

class RuleBasedDetector:
    def __init__(self):
        self.rules = [
            {"name": "port_scan", "condition": lambda r: r.get("unique_dst_ports", 0) >= 30, "severity": 0.8},
            {"name": "syn_flood", "condition": lambda r: r.get("syn_count", 0) >= 50, "severity": 0.9},
            {"name": "rst_storm", "condition": lambda r: r.get("rst_count", 0) >= 20, "severity": 0.7},
            {"name": "high_packet_rate", "condition": lambda r: r.get("packets_per_sec", 0) >= 2000, "severity": 0.85},
            {"name": "large_transfer", "condition": lambda r: r.get("byte_count", 0) >= 5_000_000, "severity": 0.75},
            {"name": "suspicious_port", "condition": lambda r: r.get("dst_port", 0) in {31337, 4444, 6667, 12345, 1337, 65535}, "severity": 0.6},
            {"name": "short_burst", "condition": lambda r: r.get("duration_sec", 999) < 1.0 and r.get("packet_count", 0) > 500, "severity": 0.8},
        ]
    def predict(self, df):
        labels = np.zeros(len(df), dtype=int)
        scores = np.zeros(len(df), dtype=float)
        for i, row in enumerate(df.to_dict(orient="records")):
            max_sev = 0.0
            for rule in self.rules:
                try:
                    if rule["condition"](row):
                        max_sev = max(max_sev, rule["severity"])
                except Exception:
                    pass
            if max_sev > 0:
                labels[i] = 1
                scores[i] = max_sev
        return DetectionResult(labels=labels, scores=scores, method="rule_based", details={"n_rules": len(self.rules)})
    def fit(self, df=None, **kwargs):
        return self
    def fit_predict(self, df, **kwargs):
        return self.predict(df)

def _normalize(arr):
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    return np.zeros_like(arr) if mx - mn < 1e-12 else (arr - mn) / (mx - mn)

class HybridDetector:
    def __init__(self, contamination=0.05, z_threshold=3.5, min_votes=2):
        self.stat = StatisticalDetector(threshold=z_threshold)
        self.iforest = IsolationForestDetector(contamination=contamination) if HAS_SKLEARN else None
        self.rules = RuleBasedDetector()
        self.min_votes = min_votes
        self._fitted = False
    def fit(self, df, feature_cols=None):
        self.stat.fit(df, feature_cols)
        if self.iforest is not None:
            self.iforest.fit(df, feature_cols)
        self.rules.fit(df)
        self._fitted = True
        return self
    def predict(self, df):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        r_stat, r_rule = self.stat.predict(df), self.rules.predict(df)
        if self.iforest is not None:
            r_if = self.iforest.predict(df)
            votes = r_stat.labels + r_if.labels + r_rule.labels
            scores = 0.35 * _normalize(r_stat.scores) + 0.40 * _normalize(r_if.scores) + 0.25 * r_rule.scores
            if_anom = int(r_if.labels.sum())
        else:
            votes = r_stat.labels + r_rule.labels
            scores = 0.6 * _normalize(r_stat.scores) + 0.4 * r_rule.scores
            if_anom = 0
        effective_min = self.min_votes if self.iforest is not None else min(self.min_votes, 2)
        labels = (votes >= effective_min).astype(int)
        return DetectionResult(labels=labels, scores=scores, method="hybrid",
            details={"min_votes": effective_min, "stat_anomalies": int(r_stat.labels.sum()),
                     "iforest_anomalies": if_anom, "rule_anomalies": int(r_rule.labels.sum()),
                     "sklearn_available": self.iforest is not None})
    def fit_predict(self, df, feature_cols=None):
        return self.fit(df, feature_cols).predict(df)

def get_detector(name, **kwargs):
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("stat", "statistical", "zscore", "mad"):
        return StatisticalDetector(**kwargs)
    if name in ("iforest", "isolation_forest", "isolation", "ml"):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for isolation_forest")
        return IsolationForestDetector(**kwargs)
    if name in ("rule", "rules", "rule_based"):
        return RuleBasedDetector()
    if name in ("hybrid", "ensemble", "all"):
        return HybridDetector(**kwargs)
    raise ValueError(f"Unknown detector: {name}")

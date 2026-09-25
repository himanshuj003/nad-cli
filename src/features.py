"""
Feature engineering for network flow anomaly detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

NUMERIC_FEATURES: List[str] = [
    "packet_count",
    "byte_count",
    "duration_sec",
    "packets_per_sec",
    "bytes_per_sec",
    "syn_count",
    "fin_count",
    "rst_count",
    "unique_dst_ports",
]

NUMERIC_FEATURES_WITH_PORTS: List[str] = NUMERIC_FEATURES + [
    "src_port",
    "dst_port",
]


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    log_transform: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    cols = feature_cols or NUMERIC_FEATURES
    available = [c for c in cols if c in df.columns]
    X = df[available].copy()
    X = X.fillna(0)
    if log_transform:
        for c in available:
            if X[c].min() >= 0:
                X[c] = np.log1p(X[c].astype(float))
    return X, available


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "packet_count" in out.columns and "duration_sec" in out.columns:
        out["packets_per_sec"] = out["packet_count"] / out["duration_sec"].clip(lower=0.01)
    if "byte_count" in out.columns and "duration_sec" in out.columns:
        out["bytes_per_sec"] = out["byte_count"] / out["duration_sec"].clip(lower=0.01)
    if "byte_count" in out.columns and "packet_count" in out.columns:
        out["avg_packet_size"] = out["byte_count"] / out["packet_count"].clip(lower=1)
    return out

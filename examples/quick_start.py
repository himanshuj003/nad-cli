#!/usr/bin/env python3
"""Minimal example: generate data → detect with hybrid → print top anomalies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_generator import generate_dataset
from src.detectors import HybridDetector
from src.evaluator import evaluate, print_report

def main():
    print("Generating 2000 synthetic flows ...")
    df = generate_dataset(n_samples=2000, anomaly_ratio=0.06, seed=7)
    print("Fitting hybrid detector ...")
    detector = HybridDetector(contamination=0.06, z_threshold=3.5, min_votes=2)
    result = detector.fit_predict(df)
    metrics = evaluate(df["is_anomaly"].values, result.labels, result.scores)
    print_report(metrics, method="hybrid")
    df["anomaly"] = result.labels
    df["score"] = result.scores
    top = df[df["anomaly"] == 1].nlargest(8, "score")
    print("Top anomalous flows:")
    cols = ["src_ip", "dst_ip", "dst_port", "protocol", "packet_count", "byte_count", "score"]
    print(top[[c for c in cols if c in top.columns]].to_string(index=False))

if __name__ == "__main__":
    main()

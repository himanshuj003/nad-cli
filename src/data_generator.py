"""Synthetic network traffic data generator with injected anomalies."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

def _random_ip(rng, private=True):
    if private:
        return f"192.168.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
    return f"{rng.integers(1, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"

def generate_normal_flows(n_samples=5000, start_time=None, seed=42):
    rng = np.random.default_rng(seed)
    start = start_time or datetime.now() - timedelta(hours=2)
    protocols = rng.choice(["TCP", "UDP", "ICMP"], size=n_samples, p=[0.75, 0.20, 0.05])
    common_ports = [80, 443, 22, 53, 25, 110, 143, 993, 995, 3306, 5432, 8080]
    dst_ports = rng.choice(common_ports + list(range(1024, 5000)), size=n_samples)
    dst_ports = np.where(rng.random(n_samples) < 0.6, rng.choice(common_ports, size=n_samples), dst_ports)
    packet_count = rng.lognormal(mean=3.0, sigma=1.2, size=n_samples).astype(int).clip(1, 5000)
    byte_count = (packet_count * rng.uniform(40, 1400, size=n_samples)).astype(int)
    duration = rng.exponential(scale=2.5, size=n_samples).clip(0.01, 120)
    data = {
        "timestamp": [start + timedelta(seconds=float(x)) for x in np.cumsum(rng.exponential(0.5, n_samples))],
        "src_ip": [_random_ip(rng) for _ in range(n_samples)],
        "dst_ip": [_random_ip(rng, private=False) for _ in range(n_samples)],
        "src_port": rng.integers(1024, 65535, size=n_samples),
        "dst_port": dst_ports,
        "protocol": protocols,
        "packet_count": packet_count,
        "byte_count": byte_count,
        "duration_sec": np.round(duration, 3),
        "packets_per_sec": np.round(packet_count / duration, 2),
        "bytes_per_sec": np.round(byte_count / duration, 2),
        "syn_count": rng.integers(0, 3, size=n_samples),
        "fin_count": rng.integers(0, 3, size=n_samples),
        "rst_count": rng.integers(0, 2, size=n_samples),
        "unique_dst_ports": rng.integers(1, 5, size=n_samples),
        "is_anomaly": np.zeros(n_samples, dtype=int),
    }
    return pd.DataFrame(data)

def inject_anomalies(df, anomaly_ratio=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(df)
    n_anom = max(1, int(n * anomaly_ratio))
    indices = rng.choice(n, size=n_anom, replace=False)
    df = df.copy()
    anomaly_types = rng.choice(["port_scan", "ddos_flood", "exfiltration", "rare_port", "rst_storm"], size=n_anom)
    for idx, atype in zip(indices, anomaly_types):
        if atype == "port_scan":
            df.loc[idx, "unique_dst_ports"] = rng.integers(50, 200)
            df.loc[idx, "packet_count"] = rng.integers(50, 300)
            df.loc[idx, "byte_count"] = df.loc[idx, "packet_count"] * rng.integers(40, 80)
            df.loc[idx, "duration_sec"] = rng.uniform(0.5, 5)
            df.loc[idx, "dst_port"] = rng.integers(1, 1024)
        elif atype == "ddos_flood":
            df.loc[idx, "packet_count"] = rng.integers(5000, 50000)
            df.loc[idx, "byte_count"] = df.loc[idx, "packet_count"] * rng.integers(60, 200)
            df.loc[idx, "duration_sec"] = rng.uniform(0.1, 2)
            df.loc[idx, "syn_count"] = rng.integers(100, 1000)
        elif atype == "exfiltration":
            df.loc[idx, "byte_count"] = rng.integers(5_000_000, 50_000_000)
            df.loc[idx, "packet_count"] = rng.integers(2000, 20000)
            df.loc[idx, "duration_sec"] = rng.uniform(30, 300)
        elif atype == "rare_port":
            df.loc[idx, "dst_port"] = rng.choice([31337, 4444, 6667, 12345, 65535, 1337])
            df.loc[idx, "protocol"] = "TCP"
        elif atype == "rst_storm":
            df.loc[idx, "rst_count"] = rng.integers(50, 500)
            df.loc[idx, "packet_count"] = rng.integers(100, 1000)
            df.loc[idx, "syn_count"] = rng.integers(20, 200)
        dur = max(df.loc[idx, "duration_sec"], 0.01)
        df.loc[idx, "packets_per_sec"] = round(df.loc[idx, "packet_count"] / dur, 2)
        df.loc[idx, "bytes_per_sec"] = round(df.loc[idx, "byte_count"] / dur, 2)
        df.loc[idx, "is_anomaly"] = 1
    return df

def generate_dataset(n_samples=5000, anomaly_ratio=0.05, seed=42, save_path=None):
    df = generate_normal_flows(n_samples=n_samples, seed=seed)
    df = inject_anomalies(df, anomaly_ratio=anomaly_ratio, seed=seed + 1)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} flows ({df['is_anomaly'].sum()} anomalies) → {save_path}")
    return df

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--samples", type=int, default=5000)
    parser.add_argument("-r", "--ratio", type=float, default=0.05)
    parser.add_argument("-o", "--output", type=str, default="data/synthetic_flows.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(args.samples, args.ratio, args.seed, args.output)

#!/usr/bin/env python3
"""Network Anomaly Detector - CLI"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from src.data_generator import generate_dataset
from src.detectors import get_detector
from src.evaluator import evaluate, print_report

console = Console()

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Network Anomaly Detector — detect suspicious traffic patterns."""
    pass

@cli.command()
@click.option("-n", "--samples", default=5000)
@click.option("-r", "--ratio", default=0.05)
@click.option("-o", "--output", default="data/synthetic_flows.csv")
@click.option("--seed", default=42)
def generate(samples, ratio, output, seed):
    """Generate synthetic network flow data with injected anomalies."""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(samples, ratio, seed, save_path=output)
    console.print(Panel.fit(
        f"[green]Generated {len(df)} flows[/green]\nAnomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)\nSaved → {output}",
        title="Data Generator"))

@cli.command()
@click.option("-i", "--input", "input_path", required=True)
@click.option("-m", "--method", default="hybrid",
    type=click.Choice(["statistical", "isolation_forest", "rule_based", "hybrid"], case_sensitive=False))
@click.option("-o", "--output", default=None)
@click.option("--threshold", default=3.5)
@click.option("--contamination", default=0.05)
@click.option("--min-votes", default=2)
def detect(input_path, method, output, threshold, contamination, min_votes):
    """Run anomaly detection on a flow CSV."""
    df = pd.read_csv(input_path)
    kwargs = {}
    if method == "statistical":
        kwargs["threshold"] = threshold
    elif method == "isolation_forest":
        kwargs["contamination"] = contamination
    elif method == "hybrid":
        kwargs.update({"contamination": contamination, "z_threshold": threshold, "min_votes": min_votes})
    result = get_detector(method, **kwargs).fit_predict(df)
    df_out = df.copy()
    df_out["anomaly"] = result.labels
    df_out["anomaly_score"] = result.scores
    n_anom = result.n_anomalies
    pct = 100 * n_anom / len(df) if len(df) else 0
    table = Table(title=f"Detection Results — {method}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total flows", str(len(df)))
    table.add_row("Anomalies found", f"{n_anom} ({pct:.2f}%)")
    table.add_row("Method", result.method)
    console.print(table)
    top = df_out[df_out["anomaly"] == 1].nlargest(5, "anomaly_score")
    if len(top):
        console.print("\n[bold]Top 5 anomalies by score:[/bold]")
        cols = [c for c in ["src_ip", "dst_ip", "dst_port", "protocol", "packet_count", "byte_count", "anomaly_score"] if c in top.columns]
        console.print(top[cols].to_string(index=False))
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output, index=False)
        console.print(f"\n[green]Results saved → {output}[/green]")

@cli.command(name="evaluate")
@click.option("-i", "--input", "input_path", required=True)
@click.option("-m", "--method", default="hybrid",
    type=click.Choice(["statistical", "isolation_forest", "rule_based", "hybrid"], case_sensitive=False))
@click.option("--threshold", default=3.5)
@click.option("--contamination", default=0.05)
@click.option("--min-votes", default=2)
def evaluate_cmd(input_path, method, threshold, contamination, min_votes):
    """Evaluate detector against ground-truth labels."""
    df = pd.read_csv(input_path)
    if "is_anomaly" not in df.columns:
        console.print("[red]CSV must contain an 'is_anomaly' column.[/red]")
        sys.exit(1)
    kwargs = {}
    if method == "statistical":
        kwargs["threshold"] = threshold
    elif method == "isolation_forest":
        kwargs["contamination"] = contamination
    elif method == "hybrid":
        kwargs.update({"contamination": contamination, "z_threshold": threshold, "min_votes": min_votes})
    result = get_detector(method, **kwargs).fit_predict(df)
    metrics = evaluate(df["is_anomaly"].values, result.labels, result.scores)
    print_report(metrics, method=method)

@cli.command()
@click.option("-n", "--samples", default=3000)
def demo(samples):
    """Quick end-to-end demo."""
    console.print(Panel.fit("[bold cyan]Network Anomaly Detector — Demo[/bold cyan]"))
    console.print("\n[1/4] Generating synthetic data ...")
    df = generate_dataset(n_samples=samples, anomaly_ratio=0.05, seed=42)
    console.print(f"    → {len(df)} flows, {df['is_anomaly'].sum()} true anomalies")
    methods = ["statistical", "isolation_forest", "rule_based", "hybrid"]
    console.print("\n[2/4] Running all detectors ...\n")
    results = {}
    for m in methods:
        try:
            res = get_detector(m).fit_predict(df)
            metrics = evaluate(df["is_anomaly"].values, res.labels, res.scores)
            results[m] = (res, metrics)
            console.print(f"  [green]{m:20s}[/green]  found={res.n_anomalies:4d}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
        except ImportError as e:
            console.print(f"  [yellow]{m:20s}[/yellow]  skipped — {e}")
    console.print("\n[3/4] Summary table")
    table = Table(title="Method Comparison")
    for col in ["Method", "Predicted", "Precision", "Recall", "F1"]:
        table.add_column(col, justify="right" if col != "Method" else "left")
    for m, (res, met) in results.items():
        table.add_row(m, str(res.n_anomalies), f"{met['precision']:.3f}", f"{met['recall']:.3f}", f"{met['f1']:.3f}")
    console.print(table)
    console.print("\n[4/4] [green]Demo complete.[/green]")

if __name__ == "__main__":
    cli()

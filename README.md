# Network Anomaly Detector — CLI

**Command-line only** variant. No web UI.

Detect anomalous network flows using:
- Statistical (robust MAD)
- Isolation Forest
- Rule-based signatures
- Hybrid ensemble

## Install

```bash
cd nad-cli
pip install -r requirements.txt
```

## Usage

```bash
python cli.py demo
python cli.py generate -n 5000 -r 0.05 -o data/flows.csv
python cli.py detect -i data/flows.csv -m hybrid -o results.csv
python cli.py detect -i data/flows.csv -m statistical --threshold 4.5
python cli.py detect -i data/flows.csv -m isolation_forest --contamination 0.05
python cli.py detect -i data/flows.csv -m rule_based
python cli.py evaluate -i data/flows.csv -m hybrid
```

## Sibling projects

- [nad-poc](https://github.com/himanshuj003/nad-poc) — single-script proof of concept
- [nad-dashboard](https://github.com/himanshuj003/nad-dashboard) — Streamlit web UI
- [nad-scaffold](https://github.com/himanshuj003/nad-scaffold) — full production scaffold
- Combined: [network-anomaly-detector](https://github.com/himanshuj003/network-anomaly-detector)

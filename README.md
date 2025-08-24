# DDoS Detection with Federated Learning

This project implements DDoS attack detection using federated learning with the [Flower](https://flower.ai/) framework. It supports centralized and federated training on benchmark intrusion detection datasets.

## 📂 Datasets

This project uses 2 publicly available datasets:

- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html)

### Setup

1. Download one of the datasets from the official websites above.
2. Place the downloaded zip file (containing CSV files) inside the `data/` directory.
3. Only one dataset should be present in `data/` at a time (the code is designed to handle them individually).

> ⚠️ Note: These datasets are very large (exceeding Git LFS free limits of 2GB). They are not included in this repository and must be downloaded manually.

## ⚙️ Installation

Install dependencies in editable mode:

```
pip install -e .
```

## 🚀 Usage

All commands (except Flower’s native ones) are defined in `app/main.py`.
Some configurations must be changed directly in the code, as not all are exposed via CLI arguments.

### Data Preparation

```
# Preprocess raw data
python -m app.main data --preprocess

# Downsample with a specific anomaly ratio
python -m app.main data --downsample --anomaly-ratio .2
```

### Centralized Training

```
python -m app.main train --model-name=test --centralized --epochs=20
```

### Federated Training

> Flower's configuration can be customized in the `project.toml` file.

```
flwr run .
```

### Model Evaluation

```
python -m app.main train --model-name=test --evaluate
```

## 📌 Note

- This is a personal research project; not all parameters are configurable via CLI.
- If you need different configurations, adjust them directly in the source code.

# Two-Stage Semi-Supervised Network Intrusion Detection

This project provides a two-stage training pipeline for network intrusion detection on tabular network-flow data.

Stage 1 performs binary classification between benign and malicious traffic. Stage 2 classifies malicious samples by attack stage or activity. The pipeline includes feature engineering, semi-supervised learning, threshold selection, evaluation, and artifact export.

## Supported Datasets

- `dapt2020`
- `zapt`
- `earlycrow`
- `BAI-Net26`
- `Unraveled`
- `UNSW-NB15-2024`

ZAPT and BAI-Net26 are included through Git LFS. The sources for the other four datasets are listed in [DATASETS.md](DATASETS.md). Use `--dataset` to select a dataset and `--data_dir` to provide its local path.

## Project Structure

```text
.
├── config.py          # Experiment configuration
├── data_loader.py     # Dataset loading, feature engineering, and splitting
├── losses.py          # Supervised and FixMatch losses
├── model.py           # PyTorch and XGBoost models and wrappers
├── train_model.py     # Training and evaluation entry point
├── utils.py           # Reproducibility, augmentation, and metrics
├── requirements.txt   # Python dependencies
├── dataset/           # Local datasets
├── checkpoints/       # Model checkpoints and evaluation results
└── artifacts/         # Threshold and prediction exports
```

## Tested Environment

```text
Python: 3.10.12
OS: Linux x86_64, glibc 2.35
PyTorch: 1.13.0+cu117
PyTorch CUDA runtime: 11.7
XGBoost: 2.0.3
NVIDIA driver: 570.133.07
GPU: NVIDIA GeForce RTX 4090
```

The default single-process training path uses the XGBoost CUDA backend and requires a CUDA-capable NVIDIA GPU.

## Installation and Running

Clone the repository:

```bash
git clone https://github.com/bithead-stack/THEMIS.git
cd THEMIS
```

Create and activate a Python 3.10 virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the training pipeline with a dataset name and its local path:

```bash
python train_model.py \
  --dataset <dataset_name> \
  --data_dir <dataset_path>
```

Display all available options:

```bash
python train_model.py --help
```

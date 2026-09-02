# Two-Stage Semi-Supervised Network Intrusion Detection

This project provides a two-stage training pipeline for network intrusion detection on tabular network-flow data.

Stage 1 performs binary classification between benign and malicious traffic. Stage 2 classifies malicious samples by attack stage or activity. The pipeline includes feature engineering, semi-supervised learning, threshold selection, evaluation, and artifact export.

## Features

- Two-stage benign/malicious and attack-category classification
- Semi-supervised learning with pseudo-labeling
- Network-context and statistical feature engineering
- Train, validation, and test splitting with feature scaling
- Validation-based decision-threshold selection
- Closed-set and `suspicious_unknown` inference policies
- Class balancing and rare-class oversampling
- Model checkpoint, metric, error-sample, and prediction export
- Optional distributed XGBoost training with Dask

## Supported Datasets

- `dapt2020`
- `zapt`
- `cic2024`
- `merged_bai`
- `weekdata`
- `earlycrow`

Dataset files are not included in this repository. Use `--dataset` to select a dataset and `--data_dir` to provide its local path.

## Pipeline

```text
Raw network-flow data
        │
        ▼
Cleaning and feature engineering
        │
        ▼
Train / validation / test split
        │
        ▼
Stage 1: Benign vs. Malicious
        │
        ▼
Stage 2: Attack stage or activity
        │
        ▼
Threshold selection and evaluation
        │
        ▼
Checkpoints, metrics, and predictions
```

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

## Usage

Run the complete two-stage pipeline:

```bash
python train_model.py \
  --dataset <dataset_name> \
  --data_dir <dataset_path>
```

Run Stage 1 only:

```bash
python train_model.py \
  --dataset <dataset_name> \
  --data_dir <dataset_path> \
  --task stage1
```

Use activity labels for Stage 2:

```bash
python train_model.py \
  --dataset <dataset_name> \
  --data_dir <dataset_path> \
  --stage2_label activity
```

Enable suspicious and unknown output labels:

```bash
python train_model.py \
  --dataset <dataset_name> \
  --data_dir <dataset_path> \
  --inference_policy suspicious_unknown
```

Enable verbose output:

```bash
python train_model.py \
  --dataset <dataset_name> \
  --data_dir <dataset_path> \
  --verbose
```

Display all command-line options:

```bash
python train_model.py --help
```

## Configuration Files

The `--config_path` option accepts JSON, YAML, and YML files.

Example YAML configuration:

```yaml
experiment_config:
  dataset: cic2024
  data_dir: ./dataset/cic2024/cic2024.csv
  task: all
  stage2_label: stage
  inference_policy: original
  seed: 42
  test_size: 0.2
  val_size: 0.1
  labeled_ratio: 0.5
  pseudo_label_threshold: 0.9
  checkpoint_dir: checkpoints
  artifacts_dir: artifacts
```

Run with a configuration file:

```bash
python train_model.py --config_path ./configs/cic2024.yaml
```

Explicit command-line experiment options override matching values from the configuration file.

## Common Options

| Option | Description | Default |
| --- | --- | --- |
| `--dataset` | Dataset identifier | `dapt2020` |
| `--data_dir` | Local dataset path | `./dataset/dapt2020` |
| `--task` | Run `all` stages or `stage1` only | `all` |
| `--stage2_label` | Use `stage` or `activity` labels | `stage` |
| `--inference_policy` | Use `original` or `suspicious_unknown` | `original` |
| `--seed` | Random seed | `42` |
| `--test_size` | Test-set fraction | `0.2` |
| `--val_size` | Validation-set fraction | `0.1` |
| `--labeled_ratio` | Labeled fraction for semi-supervised training | `0.5` |
| `--pseudo_label_threshold` | Pseudo-label confidence threshold | `0.9` |
| `--checkpoint_dir` | Checkpoint and metric output directory | `checkpoints` |
| `--artifacts_dir` | Threshold and prediction output directory | `artifacts` |
| `--verbose` | Enable detailed console output | disabled |

## Outputs

Depending on the task and inference policy, the pipeline may generate:

```text
checkpoints/
├── <dataset>_stage_binary_seed<seed>_best.pt
├── <dataset>_cascade_feedback_seed<seed>_best.pt
├── <dataset>_seed<seed>_stage1_metrics.json
├── <dataset>_seed<seed>_stage2_metrics.json
├── <dataset>_seed<seed>_end2end_metrics.json
├── <dataset>_seed<seed>_errors.csv
└── <dataset>_hard_errors.csv

artifacts/
├── thresholds/
│   └── <dataset>_<seed>.json
└── predictions/
    └── <dataset>_seed<seed>_closed_set.csv
```

Reported evaluation results include accuracy, macro and weighted precision, recall, F1, AUC, FPR, and confusion matrices.

## Reproducibility

- Keep the random seed fixed across comparable experiments.
- Record the exact command or configuration file used for each run.
- Record the dataset version, preprocessing rules, class filters, and split strategy.
- Use separate checkpoint and artifact directories for independent experiments.
- Keep the Python, CUDA, GPU driver, and dependency versions with published results.

## Data and License

The datasets are not distributed with this repository. Users are responsible for obtaining them from legitimate sources and complying with their respective licenses and terms of use.

The source code in this repository is licensed under the [MIT License](LICENSE). Dataset samples remain subject to their original licenses and terms of use.

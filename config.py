from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "dapt2020"
    task: str = "all"
    model_type: str = "tree"
    malicious_method: str = "semi"
    inference_policy: str = "original"
    seed: int = 42
    data_dir: str = "./dataset/dapt2020"
    drop_stages: str = ""
    verbose: bool = False
    force_moe: bool = False
    stage2_label: str = "stage"
    stage2_class_boost_json: str = ""

    test_size: float = 0.2
    val_size: float = 0.1

    batch_size: int = 2048
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4

    labeled_ratio: float = 0.5
    unlabeled_ratio: float = 0.9
    pseudo_label_threshold: float = 0.9
    unsup_loss_weight: float = 1.0

    weak_noise_std: float = 0.05
    strong_noise_std: float = 0.15
    strong_feature_dropout: float = 0.05

    max_print_errors: int = 80
    checkpoint_dir: str = "checkpoints"

    min_class_count: int = 1

    tree_n_estimators: int = 2500
    tree_max_features: str = "sqrt"

    hgb_max_iter: int = 800
    hgb_learning_rate: float = 0.06

    xgb_num_boost_round: int = 8000
    xgb_early_stopping_rounds: int = 200
    xgb_eta: float = 0.05
    xgb_max_depth: int = 8
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_min_child_weight: float = 1.0
    xgb_max_bin: int = 256
    xgb_scale_pos_weight: float = 1.0

    cascade_threshold_min: float = 0.05
    cascade_threshold_max: float = 0.95
    cascade_threshold_steps: int = 19
    cascade_objective: str = "f1"

    stage1_threshold_objective: str = "min"
    stage1_min_recall: float = 0.9
    stage1_gate_method: str = "threshold"
    stage1_fpr_budget: float = 0.001
    fixed_stage1_threshold: float | None = None
    stage1_tau_b: float | None = None
    stage1_tau_m: float | None = None
    stage2_margin_min: float = 0.0
    oversample_rare_classes: bool = False
    oversample_target_count: int = 50
    artifacts_dir: str = "artifacts"
    federated_enabled: bool = False
    federated_backend: str = "single_process"
    federated_num_workers: int = 3
    federated_gateway_map_path: str = ""

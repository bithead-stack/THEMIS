from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weak_augment(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return x
    return x + noise_std * torch.randn_like(x)


def strong_augment(x: torch.Tensor, noise_std: float, feature_dropout: float) -> torch.Tensor:
    out = x
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    if feature_dropout > 0:
        drop_mask = torch.rand_like(out).ge(feature_dropout).float()
        out = out * drop_mask
    return out


@dataclass(frozen=True)
class Metrics:
    acc: float
    f1: float
    precision: float
    recall: float
    auc: float
    cm: np.ndarray


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray, average: str | None) -> float:
    y_true = y_true.astype(np.int64)
    n_classes = y_prob.shape[1]
    aucs: list[float] = []
    for c in range(n_classes):
        y_c = (y_true == c).astype(np.int64)
        if y_c.min() == y_c.max():
            continue
        try:
            aucs.append(float(roc_auc_score(y_c, y_prob[:, c])))
        except Exception:
            continue
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    num_classes: int,
) -> Metrics:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    auc = float("nan")
    if y_prob is not None:
        if num_classes == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            auc = _safe_auc(y_true, y_prob, average="macro")

    return Metrics(acc=acc, f1=f1, precision=precision, recall=recall, auc=auc, cm=cm)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_confusion_matrix(cm: np.ndarray, labels: list[str]) -> str:
    df = pd.DataFrame(cm, index=labels, columns=labels)
    return df.to_string()


def tune_binary_threshold(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_key = (-1.0, -1.0)
    for t in thresholds:
        y_pred = (prob_pos >= t).astype(np.int64)
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        acc = float(accuracy_score(y_true, y_pred))
        key = (f1, acc)
        if key > best_key:
            best_key = key
            best_t = float(t)
    return best_t

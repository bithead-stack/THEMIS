from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import yaml
except Exception:
    yaml = None

from config import ExperimentConfig
from data_loader import (
    META_COLS_DAPT,
    add_network_context_group_stats_train_only,
    refresh_cic2024_derived_features,
    load_cic2024_dataset,
    load_dapt2020_dataset,
    load_earlycrow_dataset,
    load_zapt_dataset,
    make_activity_task,
    make_stage_task,
    scale_by_indices,
    split_indices,
    split_scale,
)
from model import (
    _AvgProbaEnsemble,
    _TorchTabularWrapper,
    _XGBBoosterWrapper,
    TabularMLP,
    _can_use_xgb_cuda,
    _fit_xgb_binary,
    _fit_xgb_multiclass,
    _predict_binary_prob_pos,
    _predict_multiclass,
    _use_federated_dask_backend,
)
from losses import fixmatch_unsup_loss, supervised_ce_loss
from utils import Metrics, compute_metrics, ensure_dir, format_confusion_matrix, set_seed, strong_augment, weak_augment

SUSPICIOUS_LABEL = "Suspicious"
MALICIOUS_UNKNOWN_LABEL = "Malicious-Unknown"

def _macro_weighted_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true).astype(np.int64, copy=False)
    y_pred = np.asarray(y_pred).astype(np.int64, copy=False)
    kwargs = {"zero_division": 0}
    if labels is not None:
        kwargs["labels"] = list(np.asarray(labels, dtype=np.int64).tolist())
    return {
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", **kwargs)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", **kwargs)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", **kwargs)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", **kwargs)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", **kwargs)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", **kwargs)),
    }


def _macro_acc_from_cm(cm: np.ndarray) -> float:
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return float("nan")
    denom = cm.sum(axis=1).astype(np.float64, copy=False)
    diag = np.diag(cm).astype(np.float64, copy=False)
    valid = denom > 0
    if not bool(valid.any()):
        return float("nan")
    return float(np.mean(diag[valid] / denom[valid]))


def _stage1_fpr_from_cm(cm: np.ndarray) -> float:
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        return float("nan")
    denom = float(cm[0, 0] + cm[0, 1])
    if denom <= 0:
        return float("nan")
    return float(cm[0, 1] / denom)


def _benign_fpr_from_cm(cm: np.ndarray) -> float:
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] < 2 or cm.shape[0] != cm.shape[1]:
        return float("nan")
    denom = float(cm[0].sum())
    if denom <= 0:
        return float("nan")
    return float(1.0 - (float(cm[0, 0]) / denom))


def _fpr_macro_weighted_from_cm(cm: np.ndarray) -> tuple[float, float]:
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return float("nan"), float("nan")
    total = float(cm.sum())
    if total <= 0:
        return float("nan"), float("nan")
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    diag = np.diag(cm)
    fp = col_sum - diag
    tn = total - row_sum - col_sum + diag
    denom = fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        fpr = np.where(denom > 0, fp / denom, np.nan)
    macro = float(np.nanmean(fpr))
    wsum = float(np.sum(row_sum))
    weighted = float(np.nansum(fpr * (row_sum / wsum))) if wsum > 0 else float("nan")
    return macro, weighted


def _auc_macro_weighted_ovr(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 2 or len(prob) != len(y_true) or prob.shape[1] < 2:
        return float("nan"), float("nan")
    num_classes = int(prob.shape[1])
    aucs: list[float] = []
    supports: list[float] = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(np.int64, copy=False)
        pos = int(y_bin.sum())
        neg = int(len(y_bin) - pos)
        if pos == 0 or neg == 0:
            aucs.append(float("nan"))
            supports.append(float(pos))
            continue
        try:
            a = float(roc_auc_score(y_bin, prob[:, c]))
        except Exception:
            a = float("nan")
        aucs.append(a)
        supports.append(float(pos))
    a = np.asarray(aucs, dtype=np.float64)
    s = np.asarray(supports, dtype=np.float64)
    macro = float(np.nanmean(a))
    denom = float(np.nansum(s[~np.isnan(a)]))
    weighted = float(np.nansum(a * (s / max(denom, 1e-12)))) if denom > 0 else float("nan")
    return macro, weighted


def _write_json_silent(path: str, payload: dict) -> None:
    try:
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_config_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files.")
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if isinstance(raw, dict) and "experiment_config" in raw and isinstance(raw["experiment_config"], dict):
        return raw["experiment_config"]
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON/YAML object: {path}")
    return raw


def _top2_margin(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2 or len(prob) == 0:
        return np.zeros(0, dtype=np.float32)
    if prob.shape[1] <= 1:
        return np.ones(len(prob), dtype=np.float32)
    top2 = np.sort(prob, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float32, copy=False)


def _policy_label_names(stage_names: list[str], inference_policy: str) -> list[str]:
    labels = ["Benign"] + [str(x) for x in stage_names]
    if str(inference_policy) == "suspicious_unknown":
        labels.extend([SUSPICIOUS_LABEL, MALICIOUS_UNKNOWN_LABEL])
    return labels


def _closed_set_metric_label_ids(
    stage_names: list[str],
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
) -> list[int]:
    known_ids = list(range(1 + len(stage_names)))
    if y_true is None and y_pred is None:
        return known_ids
    active: set[int] = set()
    for arr in (y_true, y_pred):
        if arr is None:
            continue
        vals = np.asarray(arr, dtype=np.int64).ravel()
        for v in vals.tolist():
            iv = int(v)
            if 0 <= iv < len(known_ids):
                active.add(iv)
    return sorted(active) if active else known_ids


def _policy_thresholds(
    cfg: ExperimentConfig,
    stage1_threshold: float,
    stage2_min_conf: float,
    stage2_entropy_max: float,
    stage2_margin_min: float | None = None,
) -> dict[str, float]:
    tau_b = float(stage1_threshold if getattr(cfg, "stage1_tau_b", None) is None else getattr(cfg, "stage1_tau_b"))
    tau_m_default = max(tau_b, float(stage1_threshold))
    tau_m = float(tau_m_default if getattr(cfg, "stage1_tau_m", None) is None else getattr(cfg, "stage1_tau_m"))
    tau_c = float(stage2_min_conf)
    tau_e = float(stage2_entropy_max)
    tau_delta_cfg = getattr(cfg, "stage2_margin_min", None)
    tau_delta = float(
        0.0 if tau_delta_cfg is None and stage2_margin_min is None else (
            tau_delta_cfg if tau_delta_cfg is not None else stage2_margin_min
        )
    )
    tau_m = max(tau_b, tau_m)
    return {
        "tau_b": tau_b,
        "tau_m": tau_m,
        "tau_c": tau_c,
        "tau_e": tau_e,
        "tau_delta": tau_delta,
    }


def _build_closed_set_auc_prob(
    prob_pos: np.ndarray,
    gate_pos: np.ndarray,
    stage_prob: np.ndarray | None,
    labels_all_closed: list[str],
) -> np.ndarray:
    prob_pos = np.asarray(prob_pos, dtype=np.float32)
    out = np.zeros((len(prob_pos), len(labels_all_closed)), dtype=np.float32)
    out[:, 0] = 1.0 - prob_pos
    if len(gate_pos) > 0 and stage_prob is not None:
        mal_prob = np.asarray(stage_prob, dtype=np.float32)
        denom = np.maximum(mal_prob.sum(axis=1, keepdims=True), 1e-12)
        mal_prob = mal_prob / denom
        out[gate_pos, 1:] = mal_prob * prob_pos[gate_pos].reshape(-1, 1)
    row_sum = np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return (out / row_sum).astype(np.float32, copy=False)


def _assemble_final_outputs(
    prob_pos: np.ndarray,
    gate_pos: np.ndarray,
    stage_pred: np.ndarray | None,
    stage_prob: np.ndarray | None,
    lat_force: np.ndarray | None,
    ex_force: np.ndarray | None,
    stage_names: list[str],
    inference_policy: str,
    thresholds: dict[str, float],
) -> dict[str, object]:
    prob_pos = np.asarray(prob_pos, dtype=np.float32)
    n = int(len(prob_pos))
    route = np.full(n, "Benign", dtype=object)
    top1 = np.full(n, "", dtype=object)
    conf = np.zeros(n, dtype=np.float32)
    ent = np.zeros(n, dtype=np.float32)
    margin = np.zeros(n, dtype=np.float32)
    release_accepted = np.zeros(n, dtype=bool)
    local_expert_triggered = np.zeros(n, dtype=bool)
    labels_all = _policy_label_names(stage_names, inference_policy)
    y_pred = np.zeros(n, dtype=np.int64)
    final_output = np.full(n, "Benign", dtype=object)

    tau_b = float(thresholds["tau_b"])
    tau_m = float(thresholds["tau_m"])
    tau_c = float(thresholds["tau_c"])
    tau_e = float(thresholds["tau_e"])
    tau_delta = float(thresholds["tau_delta"])

    if str(inference_policy) == "suspicious_unknown":
        suspicious_mask = (prob_pos >= np.float32(tau_b)) & (prob_pos < np.float32(tau_m))
        malicious_mask = prob_pos >= np.float32(tau_m)
        route[suspicious_mask] = "SuspiciousCandidate"
        route[malicious_mask] = "MaliciousCandidate"
        suspicious_id = 1 + len(stage_names)
        unknown_id = suspicious_id + 1
    else:
        suspicious_mask = np.zeros(n, dtype=bool)
        malicious_mask = prob_pos >= np.float32(tau_b)
        route[malicious_mask] = "MaliciousCandidate"
        suspicious_id = -1
        unknown_id = -1

    if len(gate_pos) > 0 and stage_pred is not None and stage_prob is not None:
        sp = np.asarray(stage_prob, dtype=np.float32)
        pred_local = np.asarray(stage_pred, dtype=np.int64)
        conf_local = sp.max(axis=1).astype(np.float32, copy=False)
        ent_local = _norm_entropy(sp)
        margin_local = _top2_margin(sp)
        keep = np.ones(len(pred_local), dtype=bool)
        if tau_c > 0.0:
            keep = keep & (conf_local >= np.float32(tau_c))
        if tau_e < 1.0:
            keep = keep & (ent_local <= np.float32(tau_e))
        if tau_delta > 0.0:
            keep = keep & (margin_local >= np.float32(tau_delta))
        if lat_force is not None:
            keep = keep | np.asarray(lat_force, dtype=bool)
            local_expert_triggered[gate_pos] = local_expert_triggered[gate_pos] | np.asarray(lat_force, dtype=bool)
        if ex_force is not None:
            keep = keep | np.asarray(ex_force, dtype=bool)
            local_expert_triggered[gate_pos] = local_expert_triggered[gate_pos] | np.asarray(ex_force, dtype=bool)
        release_accepted[gate_pos] = keep
        conf[gate_pos] = conf_local
        ent[gate_pos] = ent_local
        margin[gate_pos] = margin_local
        top1[gate_pos] = np.asarray([stage_names[int(i)] for i in pred_local.tolist()], dtype=object)
        if keep.any():
            pos_keep = gate_pos[keep]
            pred_keep = pred_local[keep]
            y_pred[pos_keep] = 1 + pred_keep.astype(np.int64, copy=False)
            final_output[pos_keep] = np.asarray([stage_names[int(i)] for i in pred_keep.tolist()], dtype=object)
        if str(inference_policy) == "suspicious_unknown":
            rej = ~keep
            if rej.any():
                pos_rej = gate_pos[rej]
                route_rej = route[pos_rej]
                susp_rej = route_rej == "SuspiciousCandidate"
                mal_rej = route_rej == "MaliciousCandidate"
                if susp_rej.any():
                    y_pred[pos_rej[susp_rej]] = int(suspicious_id)
                    final_output[pos_rej[susp_rej]] = SUSPICIOUS_LABEL
                if mal_rej.any():
                    y_pred[pos_rej[mal_rej]] = int(unknown_id)
                    final_output[pos_rej[mal_rej]] = MALICIOUS_UNKNOWN_LABEL

    return {
        "labels_all": labels_all,
        "y_pred": y_pred,
        "route": route,
        "top1": top1,
        "confidence": conf,
        "entropy": ent,
        "margin": margin,
        "release_accepted": release_accepted,
        "local_expert_triggered": local_expert_triggered,
        "final_output": final_output,
    }


def _unique_threshold_grid(prob: np.ndarray, extra: list[float] | None = None, max_points: int = 401) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    vals = np.unique(np.clip(prob, 0.0, 1.0))
    if len(vals) > max_points:
        qs = np.linspace(0.0, 1.0, max_points)
        vals = np.quantile(vals, qs).astype(np.float32)
    if extra:
        vals = np.unique(np.concatenate([vals.astype(np.float32), np.asarray(extra, dtype=np.float32)]))
    return np.clip(vals, 0.0, 1.0).astype(np.float32)


def _select_tau_b(y_true_bin: np.ndarray, prob_pos: np.ndarray) -> float:
    y_true_bin = np.asarray(y_true_bin, dtype=np.int64)
    prob_pos = np.asarray(prob_pos, dtype=np.float32)
    thresholds = _unique_threshold_grid(prob_pos, extra=[0.5])
    recall_targets = [0.995, 0.990, 0.985, 0.980]
    best_t = 0.5
    for target in recall_targets:
        best = None
        for t in thresholds.tolist():
            pred = prob_pos >= np.float32(t)
            mal_total = max(1, int((y_true_bin == 1).sum()))
            benign_total = max(1, int((y_true_bin == 0).sum()))
            mal_recall = float(((y_true_bin == 1) & pred).sum() / mal_total)
            if mal_recall + 1e-12 < float(target):
                continue
            benign_filtered = float(((y_true_bin == 0) & (~pred)).sum() / benign_total)
            cand = (float(t), benign_filtered)
            if best is None or cand > best[0]:
                best = (cand, float(t))
        if best is not None:
            return float(best[1])
    return float(best_t)


def _select_tau_m(y_true_bin: np.ndarray, prob_pos: np.ndarray, tau_b: float) -> float:
    y_true_bin = np.asarray(y_true_bin, dtype=np.int64)
    prob_pos = np.asarray(prob_pos, dtype=np.float32)
    thresholds = _unique_threshold_grid(prob_pos, extra=[tau_b, 0.95, 0.99])
    precision_targets = [0.99, 0.985, 0.980, 0.975, 0.950]
    for target in precision_targets:
        best = None
        for t in thresholds.tolist():
            if float(t) <= float(tau_b):
                continue
            pred = prob_pos >= np.float32(t)
            covered = int(pred.sum())
            if covered <= 0:
                continue
            tp = int(((y_true_bin == 1) & pred).sum())
            precision = float(tp / max(1, covered))
            if precision + 1e-12 < float(target):
                continue
            cand = (covered, -float(t))
            if best is None or cand > best[0]:
                best = (cand, float(t))
        if best is not None:
            return float(best[1])
    return float(max(tau_b, 0.95))


def _neighbor_values(base: float, candidates: list[float], extra: list[float] | None = None) -> list[float]:
    vals = set(float(v) for v in candidates)
    if extra:
        vals.update(float(v) for v in extra)
    vals = sorted(vals)
    if not vals:
        return [float(base)]
    order = sorted(vals, key=lambda x: (abs(float(x) - float(base)), float(x)))
    keep = sorted(set(order[: min(7, len(order))] + [float(base)]))
    return [float(v) for v in keep]


def _stage1_threshold_candidates(base_tau_b: float, base_tau_m: float) -> tuple[list[float], list[float]]:
    tau_b_vals = sorted(
        set(
            float(v)
            for v in [
                0.001,
                0.01,
                0.03,
                0.05,
                0.1,
                0.2,
                0.5,
                float(base_tau_b),
            ]
            if 0.0 <= float(v) < 1.0
        )
    )
    tau_m_vals = sorted(
        set(
            float(v)
            for v in [
                0.8,
                0.9,
                0.95,
                0.99,
                float(base_tau_m),
            ]
            if 0.0 < float(v) <= 1.0
        )
    )
    return tau_b_vals, tau_m_vals


def _class_accept_prob_grid() -> np.ndarray:
    return np.asarray(
        [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99],
        dtype=np.float32,
    )


def _write_prediction_trace(
    out_path: str,
    df_rows: pd.DataFrame,
    cfg: ExperimentConfig,
    true_label: np.ndarray,
    stage1_score: np.ndarray,
    final_outputs: dict[str, object],
    gateway_id: str = "centralized",
) -> None:
    rows = pd.DataFrame(
        {
            "sample_id": df_rows["__row_id"].to_numpy(dtype=np.int64, copy=False),
            "dataset": str(cfg.dataset),
            "seed": int(cfg.seed),
            "gateway_id": str(gateway_id),
            "true_label": np.asarray(true_label, dtype=object),
            "stage1_malicious_score": np.asarray(stage1_score, dtype=np.float32),
            "stage1_route": np.asarray(final_outputs["route"], dtype=object),
            "stage2_top1": np.asarray(final_outputs["top1"], dtype=object),
            "stage2_confidence": np.asarray(final_outputs["confidence"], dtype=np.float32),
            "stage2_entropy": np.asarray(final_outputs["entropy"], dtype=np.float32),
            "stage2_margin": np.asarray(final_outputs["margin"], dtype=np.float32),
            "local_expert_triggered": np.asarray(final_outputs["local_expert_triggered"], dtype=bool),
            "release_accepted": np.asarray(final_outputs["release_accepted"], dtype=bool),
            "final_output": np.asarray(final_outputs["final_output"], dtype=object),
        }
    )
    ensure_dir(os.path.dirname(out_path) or ".")
    rows.to_csv(out_path, index=False)


def _load_split_indices(path: str, n_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    idx_train = np.asarray(payload.get("idx_train", []), dtype=np.int64)
    idx_val = np.asarray(payload.get("idx_val", []), dtype=np.int64)
    idx_test = np.asarray(payload.get("idx_test", []), dtype=np.int64)
    if n_total > 0:
        for name, arr in (("idx_train", idx_train), ("idx_val", idx_val), ("idx_test", idx_test)):
            if arr.ndim != 1:
                raise ValueError(f"Invalid split file: {path}, {name} must be 1D.")
            if len(arr) == 0:
                raise ValueError(f"Invalid split file: {path}, {name} is empty.")
            if (arr < 0).any() or (arr >= n_total).any():
                raise ValueError(f"Invalid split file: {path}, {name} out of range 0..{n_total-1}.")
        if len(np.intersect1d(idx_train, idx_val)) > 0:
            raise ValueError(f"Invalid split file: {path}, train/val overlap.")
        if len(np.intersect1d(idx_train, idx_test)) > 0:
            raise ValueError(f"Invalid split file: {path}, train/test overlap.")
        if len(np.intersect1d(idx_val, idx_test)) > 0:
            raise ValueError(f"Invalid split file: {path}, val/test overlap.")
    return idx_train, idx_val, idx_test


def _load_or_create_split_indices(
    path: str,
    y: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y)
    if os.path.exists(path):
        return _load_split_indices(path, int(len(y)))
    idx_train, idx_val, idx_test = split_indices(y=y, seed=int(seed), test_size=float(test_size), val_size=float(val_size))
    _write_json_silent(
        path,
        {
            "n": int(len(y)),
            "seed": int(seed),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "idx_train": idx_train.tolist(),
            "idx_val": idx_val.tolist(),
            "idx_test": idx_test.tolist(),
        },
    )
    return idx_train, idx_val, idx_test


def _split_indices_flowkey(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    test_size: float,
    val_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_i64 = np.asarray(y, dtype=np.int64)
    cols = [
        "Src IP",
        "Dst IP",
        "Src Port",
        "Dst Port",
        "Timestamp",
        "Flow Duration",
        "Absolute_Time",
    ]
    use = [c for c in cols if c in df.columns]
    if len(use) < 2:
        return split_indices(y=y_i64, seed=int(seed), test_size=float(test_size), val_size=float(val_size))
    key = df[use].copy()
    for c in use:
        if key[c].dtype == "O":
            key[c] = key[c].astype(str)
        else:
            key[c] = pd.to_numeric(key[c], errors="coerce")
    group_id = pd.util.hash_pandas_object(key, index=False).to_numpy(dtype=np.uint64)
    order = np.argsort(group_id, kind="mergesort")
    gid_sorted = group_id[order]
    y_sorted = y_i64[order]
    boundaries = np.flatnonzero(np.r_[True, gid_sorted[1:] != gid_sorted[:-1], True]).astype(np.int64, copy=False)
    starts = boundaries[:-1]
    ends = boundaries[1:]
    group_ids = gid_sorted[starts]
    group_y = np.maximum.reduceat(y_sorted, starts).astype(np.int64, copy=False)

    rng = np.random.default_rng(int(seed))
    g_mal = np.where(group_y == 1)[0].astype(np.int64)
    g_ben = np.where(group_y == 0)[0].astype(np.int64)
    rng.shuffle(g_mal)
    rng.shuffle(g_ben)

    def _assign(group_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(len(group_idx))
        n_test = max(1, int(round(float(test_size) * n)))
        n_val = max(1, int(round(float(val_size) * n)))
        n_train = max(1, n - n_test - n_val)
        if n_train <= 0:
            n_train = max(1, n - n_test)
            n_val = max(1, n - n_test - n_train)
        train = group_idx[:n_train]
        val = group_idx[n_train : n_train + n_val]
        test = group_idx[n_train + n_val : n_train + n_val + n_test]
        return train, val, test

    tr_m, va_m, te_m = _assign(g_mal)
    tr_b, va_b, te_b = _assign(g_ben)
    g_train = np.concatenate([tr_m, tr_b], axis=0)
    g_val = np.concatenate([va_m, va_b], axis=0)
    g_test = np.concatenate([te_m, te_b], axis=0)
    rng.shuffle(g_train)
    rng.shuffle(g_val)
    rng.shuffle(g_test)

    split_by_group = np.full(len(group_ids), -1, dtype=np.int8)
    split_by_group[g_train] = 0
    split_by_group[g_val] = 1
    split_by_group[g_test] = 2

    row_split_sorted = np.empty(len(order), dtype=np.int8)
    for gi, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        row_split_sorted[s:e] = split_by_group[int(gi)]
    row_split = np.empty(len(order), dtype=np.int8)
    row_split[order] = row_split_sorted

    idx_train = np.where(row_split == 0)[0].astype(np.int64)
    idx_val = np.where(row_split == 1)[0].astype(np.int64)
    idx_test = np.where(row_split == 2)[0].astype(np.int64)
    if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
        return split_indices(y=y_i64, seed=int(seed), test_size=float(test_size), val_size=float(val_size))
    return idx_train, idx_val, idx_test


def _refresh_log1p_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    try:
        out = refresh_cic2024_derived_features(out)
    except Exception:
        pass
    log_cols = [c for c in out.columns if isinstance(c, str) and c.startswith("Log1p_")]
    if not log_cols:
        return out
    for log_c in log_cols:
        base = log_c[len("Log1p_") :]
        if base not in out.columns:
            continue
        v = pd.to_numeric(out[base], errors="coerce").clip(lower=0)
        out[log_c] = np.log1p(v).astype(np.float32)
    return out


def _load_selected_feature_cols(path: str | None) -> list[str] | None:
    if not str(path or "").strip():
        return None
    with open(str(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    cols = payload.get("selected_feature_cols", payload) if isinstance(payload, dict) else payload
    if not isinstance(cols, list):
        raise ValueError("Feature subset file must be a JSON list or a dict with key 'selected_feature_cols'.")
    selected = [str(c) for c in cols if str(c).strip()]
    if not selected:
        raise ValueError("Feature subset file is empty.")
    return selected

def _norm_entropy(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 2 or prob.shape[1] == 0:
        return np.zeros(len(prob), dtype=np.float32)
    eps = 1e-12
    p = np.clip(prob, eps, 1.0)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), eps)
    ent = -(p * np.log(p)).sum(axis=1)
    denom = float(np.log(max(2, int(p.shape[1]))))
    return (ent / max(denom, eps)).astype(np.float32, copy=False)

def _restore_pickled_objects(x):
    if isinstance(x, (bytes, bytearray)):
        try:
            return pickle.loads(x)
        except ModuleNotFoundError as e:
            if e.name == "numpy._core.numeric":
                sys.modules.setdefault("numpy._core", np.core)
                sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
                try:
                    return pickle.loads(x)
                except Exception:
                    return x
            return x
        except Exception:
            return x
    if isinstance(x, dict):
        return {k: _restore_pickled_objects(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_restore_pickled_objects(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_restore_pickled_objects(v) for v in x)
    return x

def _load_checkpoint(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    if "model_pickle" in payload:
        payload["model_pickle"] = _restore_pickled_objects(payload["model_pickle"])
    if "extra" in payload:
        payload["extra"] = _restore_pickled_objects(payload["extra"])
    return payload

def _default_ckpt_path(checkpoint_dir: str, dataset: str, seed: int, kind: str) -> str:
    if kind == "stage1":
        return os.path.join(checkpoint_dir, f"{dataset}_stage_binary_seed{seed}_best.pt")
    if kind == "stage2":
        return os.path.join(checkpoint_dir, f"{dataset}_cascade_feedback_seed{seed}_best.pt")
    raise ValueError(f"Unknown checkpoint kind: {kind}")

def replay_from_checkpoints(
    cfg: ExperimentConfig,
    stage1_ckpt_path: str,
    stage2_ckpt_path: str,
    split_path: str | None = None,
    split_seed: int = 42,
    split_mode: str = "random",
    allow_context_leakage: bool = False,
) -> None:
    stage1_payload = _load_checkpoint(stage1_ckpt_path)
    cfg_ckpt = ExperimentConfig(**{k: v for k, v in stage1_payload.get("config", {}).items() if k in asdict(cfg)})
    cfg = ExperimentConfig(**(asdict(cfg_ckpt) | {"data_dir": cfg.data_dir, "checkpoint_dir": cfg.checkpoint_dir, "verbose": cfg.verbose}))

    feature_cols = [str(c) for c in stage1_payload.get("feature_cols", [])]
    if not feature_cols:
        raise ValueError("Checkpoint missing feature_cols, cannot replay.")

    stage1_model = stage1_payload.get("model_pickle", None)
    if stage1_model is None:
        raise ValueError("Stage1 checkpoint missing model.")
    stage1_extra = stage1_payload.get("extra", {}) or {}
    threshold = float(stage1_extra.get("threshold", 0.5))

    if cfg.dataset == "dapt2020":
        df = load_dapt2020_dataset(cfg.data_dir)
    elif cfg.dataset == "zapt":
        df = load_zapt_dataset(cfg.data_dir)
    elif cfg.dataset == "cic2024":
        df = load_cic2024_dataset(cfg.data_dir)
    elif cfg.dataset == "merged_bai":
        df = load_cic2024_dataset(cfg.data_dir)
    elif cfg.dataset == "weekdata":
        df = load_cic2024_dataset(cfg.data_dir)
    elif cfg.dataset == "earlycrow":
        df = load_earlycrow_dataset(cfg.data_dir)
    else:
        raise NotImplementedError(f"Unsupported dataset: {cfg.dataset}")

    if cfg.dataset in {"cic2024", "merged_bai", "weekdata"} and not str(cfg.drop_stages).strip():
        cfg = ExperimentConfig(**(asdict(cfg) | {"drop_stages": "Fuzzers"}))

    drop_stages = [s.strip() for s in str(cfg.drop_stages).split(",") if str(s).strip()]
    if drop_stages:
        df = df[~df["Stage"].astype(str).isin(drop_stages)].copy()

    fixed_split = None
    if not bool(allow_context_leakage):
        stage_series = df["Stage"].astype(str)
        y_bin = (stage_series != "Benign").astype(np.int64).to_numpy()
        y_split = stage_series.to_numpy() if cfg.dataset in {"cic2024", "merged_bai", "weekdata"} else y_bin
        if split_path:
            idx_train, idx_val, idx_test = _load_or_create_split_indices(
                path=str(split_path),
                y=y_split,
                seed=int(split_seed),
                test_size=cfg.test_size,
                val_size=cfg.val_size,
            )
        elif str(split_mode) == "flowkey":
            idx_train, idx_val, idx_test = _split_indices_flowkey(
                df=df,
                y=y_bin,
                seed=int(split_seed),
                test_size=cfg.test_size,
                val_size=cfg.val_size,
            )
        else:
            idx_train, idx_val, idx_test = split_indices(
                y=y_split,
                seed=cfg.seed,
                test_size=cfg.test_size,
                val_size=cfg.val_size,
            )
        df = add_network_context_group_stats_train_only(df, idx_train=idx_train)
        df = _refresh_log1p_columns(df)
        fixed_split = (idx_train, idx_val, idx_test)

    ensure_dir(cfg.checkpoint_dir)
    errors_out_path = os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_seed{cfg.seed}_errors.csv")
    if not os.path.exists(errors_out_path):
        meta_cols = [c for c in META_COLS_DAPT if c != "__row_id"]
        pd.DataFrame(
            columns=["dataset", "seed", "drop_stages", "phase", "__row_id", "row_1based"] + meta_cols + ["y_true", "y_pred"]
        ).to_csv(errors_out_path, index=False)

    if cfg.dataset == "dapt2020":
        use_act = bool(str(getattr(cfg, "stage2_label", "stage")) == "activity")
        df_task, y_stage1, _ = make_stage_task(df, use_activity_as_stage=use_act)
    else:
        df_task, y_stage1, _ = make_stage_task(df)
    if fixed_split is None:
        stage_split = split_scale(
            df=df_task,
            y=y_stage1,
            feature_cols=feature_cols,
            seed=cfg.seed,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
        )
    else:
        idx_train, idx_val, idx_test = fixed_split
        stage_split = scale_by_indices(
            df=df_task,
            y=y_stage1,
            feature_cols=feature_cols,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
        )

    val_prob = _predict_binary_prob_pos(stage1_model, stage_split.X_val)
    test_prob = _predict_binary_prob_pos(stage1_model, stage_split.X_test)
    train_prob = _predict_binary_prob_pos(stage1_model, stage_split.X_train)

    use_guard = bool(stage1_extra.get("guard_enabled", False))
    guard_model = stage1_extra.get("guard_model", None)
    guard_thr = stage1_extra.get("guard_threshold", None)
    guard_prob_val = _predict_binary_prob_pos(guard_model, stage_split.X_val) if (use_guard and guard_model is not None) else None
    guard_prob_test = _predict_binary_prob_pos(guard_model, stage_split.X_test) if (use_guard and guard_model is not None) else None

    use_fuzz_rescue = bool(stage1_extra.get("fuzz_rescue_enabled", False))
    fuzz_model = stage1_extra.get("fuzz_model", None)
    fuzz_thr = stage1_extra.get("fuzz_rescue_threshold", None)
    fuzz_t_low = stage1_extra.get("fuzz_rescue_t_low", None)
    fuzz_mode = str(stage1_extra.get("fuzz_rescue_mode", "") or "")
    fuzz_prob_val = _predict_binary_prob_pos(fuzz_model, stage_split.X_val) if (use_fuzz_rescue and fuzz_model is not None) else None
    fuzz_prob_test = _predict_binary_prob_pos(fuzz_model, stage_split.X_test) if (use_fuzz_rescue and fuzz_model is not None) else None

    base_val_pred = (
        ((val_prob >= float(threshold)) & (guard_prob_val >= float(guard_thr))).astype(np.int64)
        if (use_guard and guard_prob_val is not None and guard_thr is not None)
        else (val_prob >= float(threshold)).astype(np.int64)
    )
    base_test_pred = (
        ((test_prob >= float(threshold)) & (guard_prob_test >= float(guard_thr))).astype(np.int64)
        if (use_guard and guard_prob_test is not None and guard_thr is not None)
        else (test_prob >= float(threshold)).astype(np.int64)
    )
    if (
        use_fuzz_rescue
        and fuzz_prob_val is not None
        and fuzz_prob_test is not None
        and fuzz_thr is not None
        and fuzz_t_low is not None
        and float(fuzz_t_low) < float(threshold)
    ):
        if fuzz_mode == "over_guard":
            near_val = (base_val_pred == 0) & (val_prob >= float(fuzz_t_low)) & (val_prob < float(threshold))
            near_test = (base_test_pred == 0) & (test_prob >= float(fuzz_t_low)) & (test_prob < float(threshold))
        else:
            near_val = (base_val_pred == 0) & (val_prob >= float(fuzz_t_low)) & (val_prob < float(threshold))
            near_test = (base_test_pred == 0) & (test_prob >= float(fuzz_t_low)) & (test_prob < float(threshold))
        rescue_val = near_val & (fuzz_prob_val >= float(fuzz_thr))
        rescue_test = near_test & (fuzz_prob_test >= float(fuzz_thr))
        y_pred_val = (base_val_pred | rescue_val.astype(np.int64)).astype(np.int64)
        y_pred_test = (base_test_pred | rescue_test.astype(np.int64)).astype(np.int64)
    else:
        y_pred_val = base_val_pred.astype(np.int64, copy=False)
        y_pred_test = base_test_pred.astype(np.int64, copy=False)

    labels_stage1 = ["Benign", "Malicious"]
    cm_val = compute_metrics(y_true=stage_split.y_val, y_pred=y_pred_val, y_prob=None, num_classes=2).cm
    cm_test = compute_metrics(y_true=stage_split.y_test, y_pred=y_pred_test, y_prob=None, num_classes=2).cm
    s1_val = _macro_weighted_summary(stage_split.y_val, y_pred_val)
    s1_val["macro_acc"] = _macro_acc_from_cm(cm_val)
    s1_val["fpr"] = _stage1_fpr_from_cm(cm_val)
    print("Stage1 Val:", s1_val)
    print(format_confusion_matrix(cm_val, labels=labels_stage1))
    s1_test = _macro_weighted_summary(stage_split.y_test, y_pred_test)
    s1_test["macro_acc"] = _macro_acc_from_cm(cm_test)
    s1_test["fpr"] = _stage1_fpr_from_cm(cm_test)
    print("Stage1 Test:", s1_test)
    print(format_confusion_matrix(cm_test, labels=labels_stage1))
    _print_and_collect_errors(
        df_task=df_task,
        idx_test=stage_split.idx_test,
        y_true=stage_split.y_test,
        y_pred=y_pred_test,
        label_names=labels_stage1,
        max_print=cfg.max_print_errors,
        do_print=False,
        phase="stage1_test",
        out_path=errors_out_path,
        cfg=cfg,
    )

    stage2_payload = _load_checkpoint(stage2_ckpt_path)
    stage2_model = stage2_payload.get("model_pickle", None)
    stage2_extra = stage2_payload.get("extra", {}) or {}
    if stage2_model is None and isinstance(stage2_payload.get("class_names", None), list) and len(stage2_payload["class_names"]) == 2:
        stage1_thr = float(stage2_extra.get("stage1_threshold", threshold))
        y_true_all_val = stage_split.y_val.astype(np.int64, copy=False)
        y_pred_all_val = np.zeros(len(y_true_all_val), dtype=np.int64)
        y_pred_all_val[val_prob >= stage1_thr] = 1
        y_true_all_test = stage_split.y_test.astype(np.int64, copy=False)
        y_pred_all_test = np.zeros(len(y_true_all_test), dtype=np.int64)
        y_pred_all_test[test_prob >= stage1_thr] = 1
        labels_all = [str(x) for x in stage2_payload["class_names"]]
        cm2_val = compute_metrics(y_true=y_true_all_val, y_pred=y_pred_all_val, y_prob=None, num_classes=2).cm
        cm2_test = compute_metrics(y_true=y_true_all_test, y_pred=y_pred_all_test, y_prob=None, num_classes=2).cm
        if cfg.verbose:
            print("Stage2 Val:", _macro_weighted_summary(y_true_all_val, y_pred_all_val))
            print(format_confusion_matrix(cm2_val, labels=labels_all))
            print("Stage2 Test:", _macro_weighted_summary(y_true_all_test, y_pred_all_test))
            print(format_confusion_matrix(cm2_test, labels=labels_all))
        e2e = _macro_weighted_summary(y_true_all_test, y_pred_all_test)
        e2e["macro_acc"] = _macro_acc_from_cm(cm2_test)
        e2e["fpr"] = _benign_fpr_from_cm(cm2_test)
        print("End2End Test:", e2e)
        print(format_confusion_matrix(cm2_test, labels=labels_all))
        return

    stage1_threshold = float(stage2_extra.get("stage1_threshold", threshold))
    stage2_min_conf = float(stage2_extra.get("stage2_min_conf", 0.0) or 0.0)
    stage2_ent_thr = float(stage2_extra.get("stage2_entropy_max", 1.01) or 1.01)
    stage2_margin_min = float(stage2_extra.get("stage2_margin_min", getattr(cfg, "stage2_margin_min", 0.0)) or 0.0)
    thresholds_bundle = _policy_thresholds(
        cfg=cfg,
        stage1_threshold=stage1_threshold,
        stage2_min_conf=stage2_min_conf,
        stage2_entropy_max=stage2_ent_thr,
        stage2_margin_min=stage2_margin_min,
    )

    feature_cols2 = stage_split.feature_cols

    use_activity = bool(cfg.dataset == "dapt2020" and str(getattr(cfg, "stage2_label", "stage")) == "activity")
    malicious = df_task[df_task["Stage"].astype(str) != "Benign"].copy()
    if use_activity:
        act = malicious["Activity"].astype(str).where(malicious["Activity"].astype(str) != "Normal", other="Other")
        if int(getattr(cfg, "min_class_count", 1)) > 1:
            counts = act.value_counts()
            rare = counts[counts < int(getattr(cfg, "min_class_count", 1))].index.tolist()
            if rare:
                act = act.where(~act.isin(rare), other="Other")
        stage = act.astype(str)
    else:
        stage = malicious["Stage"].astype(str)
    stage_names = sorted(stage.unique().tolist())
    stage_to_id = {s: i for i, s in enumerate(stage_names)}
    y_stage = stage.map(stage_to_id).astype(np.int64).to_numpy()
    row_ids = malicious["__row_id"].to_numpy(dtype=np.int64, copy=False)

    idx_train_mal = np.where(np.isin(row_ids, stage_split.row_id_train))[0].astype(np.int64)
    idx_val_mal = np.where(np.isin(row_ids, stage_split.row_id_val))[0].astype(np.int64)
    idx_test_mal = np.where(np.isin(row_ids, stage_split.row_id_test))[0].astype(np.int64)
    if len(idx_train_mal) == 0:
        idx_train_mal = np.arange(len(row_ids), dtype=np.int64)
    if len(idx_val_mal) == 0:
        idx_val_mal = idx_train_mal[: min(1, len(idx_train_mal))]
    if len(idx_test_mal) == 0:
        idx_test_mal = idx_val_mal[: min(1, len(idx_val_mal))]

    rng_stage2 = np.random.default_rng(cfg.seed)
    stage2_val_ratio = float(np.clip(cfg.val_size, 0.05, 0.3))
    idx_train_pool = idx_train_mal.astype(np.int64, copy=False)
    idx_val_from_train: list[int] = []
    if len(idx_train_pool) > 0:
        for c in range(len(stage_names)):
            idx_c = idx_train_pool[y_stage[idx_train_pool] == int(c)]
            if len(idx_c) <= 1:
                continue
            idx_c = idx_c.copy()
            rng_stage2.shuffle(idx_c)
            k = int(np.ceil(len(idx_c) * stage2_val_ratio))
            k = max(1, min(len(idx_c) - 1, k))
            idx_val_from_train.extend(idx_c[:k].tolist())

    if len(idx_val_from_train) > 0:
        idx_val_mal = np.asarray(sorted(set(idx_val_from_train)), dtype=np.int64)
        val_set = set(int(i) for i in idx_val_mal.tolist())
        idx_train_mal = np.asarray([int(i) for i in idx_train_pool.tolist() if int(i) not in val_set], dtype=np.int64)
        if len(idx_train_mal) == 0:
            idx_train_mal = idx_train_pool

    if len(set(y_stage[idx_train_mal].tolist())) != len(stage_names):
        idx_train_list = idx_train_mal.tolist()
        idx_val_list = idx_val_mal.tolist()
        idx_test_list = idx_test_mal.tolist()
        present = set(int(v) for v in y_stage[idx_train_mal].tolist())
        for c in range(len(stage_names)):
            if c in present:
                continue
            take_idx: int | None = None
            for src in (idx_val_list, idx_test_list):
                for j in list(src):
                    if int(y_stage[j]) == int(c):
                        take_idx = int(j)
                        src.remove(j)
                        break
                if take_idx is not None:
                    break
            if take_idx is None:
                cand = np.where(y_stage == int(c))[0]
                if len(cand) > 0:
                    take_idx = int(cand[0])
                    if take_idx in idx_val_list:
                        idx_val_list.remove(take_idx)
                    if take_idx in idx_test_list:
                        idx_test_list.remove(take_idx)
            if take_idx is not None:
                idx_train_list.append(int(take_idx))
        idx_train_mal = np.asarray(sorted(set(idx_train_list)), dtype=np.int64)
        idx_val_mal = np.asarray(idx_val_list, dtype=np.int64)
        idx_test_mal = np.asarray(idx_test_list, dtype=np.int64)
        if len(idx_val_mal) == 0:
            idx_val_mal = idx_train_mal[: min(1, len(idx_train_mal))]
        if len(idx_test_mal) == 0:
            idx_test_mal = idx_val_mal[: min(1, len(idx_val_mal))]

    split2 = scale_by_indices(
        df=malicious,
        y=y_stage,
        feature_cols=feature_cols2,
        idx_train=idx_train_mal,
        idx_val=idx_val_mal,
        idx_test=idx_test_mal,
    )

    stage2_extra = stage2_extra.copy()
    stage2_extra["stage1_threshold"] = float(stage1_threshold)
    stage2_extra["stage2_min_conf"] = float(stage2_min_conf)
    stage2_extra["stage2_entropy_max"] = float(stage2_ent_thr)
    stage2_extra["stage2_margin_min"] = float(stage2_margin_min)
    stage2_extra["inference_policy"] = str(getattr(cfg, "inference_policy", "original"))

    stage2_pred_val, stage2_prob_val = _predict_multiclass(stage2_model, split2.X_val, stage2_extra)
    stage2_pred_test, stage2_prob_test = _predict_multiclass(stage2_model, split2.X_test, stage2_extra)
    cm_s2_val = compute_metrics(y_true=split2.y_val, y_pred=stage2_pred_val, y_prob=stage2_prob_val, num_classes=len(stage_names)).cm
    cm_s2_test = compute_metrics(y_true=split2.y_test, y_pred=stage2_pred_test, y_prob=stage2_prob_test, num_classes=len(stage_names)).cm
    if cfg.verbose:
        print("Stage2 Val:", _macro_weighted_summary(split2.y_val, stage2_pred_val))
        print(format_confusion_matrix(cm_s2_val, labels=stage_names))
        print("Stage2 Test:", _macro_weighted_summary(split2.y_test, stage2_pred_test))
        print(format_confusion_matrix(cm_s2_test, labels=stage_names))
    _print_and_collect_errors(
        df_task=malicious,
        idx_test=split2.idx_test,
        y_true=split2.y_test,
        y_pred=stage2_pred_test,
        label_names=stage_names,
        max_print=cfg.max_print_errors,
        do_print=False,
        phase="stage2_test",
        out_path=errors_out_path,
        cfg=cfg,
    )

    idx_test_all = stage_split.idx_test.astype(np.int64, copy=False)
    gate_mask_test = test_prob >= np.float32(thresholds_bundle["tau_b"])
    y_true_e2e = np.zeros(len(idx_test_all), dtype=np.int64)
    if use_activity:
        rows_test = df_task.iloc[idx_test_all][["Stage", "Activity"]]
        stage_true_test = rows_test["Stage"].astype(str).to_numpy()
        act_true_test = rows_test["Activity"].astype(str).to_numpy()
        act_true_test = np.where(stage_true_test == "Benign", "Normal", act_true_test).astype(str)
        act_true_test = np.where(act_true_test == "Normal", "Other", act_true_test).astype(str)
        if int(getattr(cfg, "min_class_count", 1)) > 1:
            counts = pd.Series(stage).value_counts()
            rare_set = set(counts[counts < int(getattr(cfg, "min_class_count", 1))].index.tolist())
            act_true_test = np.array(["Other" if a in rare_set else a for a in act_true_test], dtype=object)
        for i, st in enumerate(stage_true_test.tolist()):
            if st != "Benign":
                y_true_e2e[i] = 1 + int(stage_to_id.get(str(act_true_test[i]), 0))
    else:
        stage_true_test = df_task.iloc[idx_test_all]["Stage"].astype(str).to_numpy()
        for i, s in enumerate(stage_true_test):
            if s != "Benign":
                y_true_e2e[i] = 1 + int(stage_to_id[s])

    gate_pos_test = np.empty(0, dtype=np.int64)
    stage_pred = None
    stage_prob = None
    lat_force = None
    ex_force = None
    if gate_mask_test.any():
        gate_pos_test = np.where(gate_mask_test)[0].astype(np.int64)
        X_gate = df_task.iloc[idx_test_all[gate_pos_test]][feature_cols2].to_numpy(dtype=np.float32, copy=True)
        X_gate = split2.scaler.transform(X_gate).astype(np.float32, copy=False)
        stage_pred, stage_prob = _predict_multiclass(stage2_model, X_gate, stage2_extra)
        lat_force, ex_force = _force_keep(stage2_extra, X_gate)

    final_outputs = _assemble_final_outputs(
        prob_pos=test_prob,
        gate_pos=gate_pos_test,
        stage_pred=stage_pred,
        stage_prob=stage_prob,
        lat_force=lat_force,
        ex_force=ex_force,
        stage_names=stage_names,
        inference_policy=str(getattr(cfg, "inference_policy", "original")),
        thresholds=thresholds_bundle,
    )
    y_pred_e2e = np.asarray(final_outputs["y_pred"], dtype=np.int64)
    labels_all = list(final_outputs["labels_all"])
    cm_e2e = compute_metrics(y_true=y_true_e2e, y_pred=y_pred_e2e, y_prob=None, num_classes=len(labels_all)).cm
    metric_label_ids = _closed_set_metric_label_ids(stage_names, y_true=y_true_e2e)
    e2e = _macro_weighted_summary(y_true_e2e, y_pred_e2e, labels=metric_label_ids)
    e2e["macro_acc"] = _macro_acc_from_cm(cm_e2e)
    e2e["fpr"] = _benign_fpr_from_cm(cm_e2e)
    prob_full_auc = _build_closed_set_auc_prob(
        prob_pos=test_prob,
        gate_pos=gate_pos_test,
        stage_prob=stage_prob,
        labels_all_closed=["Benign"] + stage_names,
    )
    e2e["macro_auc"], e2e["weighted_auc"] = _auc_macro_weighted_ovr(y_true_e2e, prob_full_auc)
    e2e["malicious_to_benign_rate"] = float(((y_true_e2e != 0) & (y_pred_e2e == 0)).sum() / max(1, int((y_true_e2e != 0).sum())))
    unknown_pred_ids = [i for i, x in enumerate(labels_all) if x in {SUSPICIOUS_LABEL, MALICIOUS_UNKNOWN_LABEL}]
    known_to_unknown = np.isin(y_pred_e2e, np.asarray(unknown_pred_ids, dtype=np.int64)) if unknown_pred_ids else np.zeros(len(y_pred_e2e), dtype=bool)
    e2e["known_to_unknown_rate"] = float((known_to_unknown & (y_true_e2e != 0)).sum() / max(1, int((y_true_e2e != 0).sum())))
    print("End2End Test:", e2e)
    print(format_confusion_matrix(cm_e2e, labels=labels_all))
    _print_and_collect_errors(
        df_task=df_task,
        idx_test=idx_test_all,
        y_true=y_true_e2e,
        y_pred=y_pred_e2e,
        label_names=labels_all,
        max_print=cfg.max_print_errors,
        do_print=False,
        phase="end2end_test",
        out_path=errors_out_path,
        cfg=cfg,
    )

def _align_proba_to_num_classes(prob: np.ndarray, classes: np.ndarray, num_classes: int) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    out = np.zeros((len(prob), int(num_classes)), dtype=np.float32)
    cls = np.asarray(classes)
    for j in range(len(cls)):
        out[:, int(cls[j])] = prob[:, j]
    return out

def replay_cic_stage2_with_benign_experiment(
    cfg: ExperimentConfig,
    stage1_ckpt_path: str,
    benign_return_min_conf: float,
) -> None:
    if cfg.dataset != "cic2024":
        raise ValueError("This experiment mode is only supported for --dataset cic2024.")

    stage1_payload = _load_checkpoint(stage1_ckpt_path)
    cfg_ckpt = ExperimentConfig(**{k: v for k, v in stage1_payload.get("config", {}).items() if k in asdict(cfg)})
    cfg = ExperimentConfig(**(asdict(cfg_ckpt) | {"data_dir": cfg.data_dir, "checkpoint_dir": cfg.checkpoint_dir, "verbose": cfg.verbose, "drop_stages": cfg.drop_stages}))

    feature_cols = [str(c) for c in stage1_payload.get("feature_cols", [])]
    if not feature_cols:
        raise ValueError("Stage1 checkpoint missing feature_cols, cannot run experiment.")
    stage1_model = stage1_payload.get("model_pickle", None)
    if stage1_model is None:
        raise ValueError("Stage1 checkpoint missing model, cannot run experiment.")
    stage1_extra = stage1_payload.get("extra", {}) or {}
    stage1_threshold = float(stage1_extra.get("threshold", 0.5))

    df = load_cic2024_dataset(cfg.data_dir)
    if not str(cfg.drop_stages).strip():
        cfg = ExperimentConfig(**(asdict(cfg) | {"drop_stages": "Fuzzers"}))
    drop_stages = [s.strip() for s in str(cfg.drop_stages).split(",") if str(s).strip()]
    if drop_stages:
        df = df[~df["Stage"].astype(str).isin(drop_stages)].copy()

    ensure_dir(cfg.checkpoint_dir)
    errors_out_path = os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_seed{cfg.seed}_errors.csv")
    if not os.path.exists(errors_out_path):
        meta_cols = [c for c in META_COLS_DAPT if c != "__row_id"]
        pd.DataFrame(
            columns=["dataset", "seed", "drop_stages", "phase", "__row_id", "row_1based"] + meta_cols + ["y_true", "y_pred"]
        ).to_csv(errors_out_path, index=False)

    df_task, y_stage1, _ = make_stage_task(df)
    stage_split = split_scale(
        df=df_task,
        y=y_stage1,
        feature_cols=feature_cols,
        seed=cfg.seed,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
    )

    p_train = _predict_binary_prob_pos(stage1_model, stage_split.X_train)
    p_val = _predict_binary_prob_pos(stage1_model, stage_split.X_val)
    p_test = _predict_binary_prob_pos(stage1_model, stage_split.X_test)

    y_pred_val = (p_val >= float(stage1_threshold)).astype(np.int64)
    y_pred_test = (p_test >= float(stage1_threshold)).astype(np.int64)
    labels_stage1 = ["Benign", "Malicious"]
    cm_val = compute_metrics(y_true=stage_split.y_val, y_pred=y_pred_val, y_prob=None, num_classes=2).cm
    cm_test = compute_metrics(y_true=stage_split.y_test, y_pred=y_pred_test, y_prob=None, num_classes=2).cm
    s1_val = _macro_weighted_summary(stage_split.y_val, y_pred_val)
    s1_val["macro_acc"] = _macro_acc_from_cm(cm_val)
    s1_val["fpr"] = _stage1_fpr_from_cm(cm_val)
    print("Stage1 Val:", s1_val)
    print(format_confusion_matrix(cm_val, labels=labels_stage1))
    s1_test = _macro_weighted_summary(stage_split.y_test, y_pred_test)
    s1_test["macro_acc"] = _macro_acc_from_cm(cm_test)
    s1_test["fpr"] = _stage1_fpr_from_cm(cm_test)
    print("Stage1 Test:", s1_test)
    print(format_confusion_matrix(cm_test, labels=labels_stage1))
    _print_and_collect_errors(
        df_task=df_task,
        idx_test=stage_split.idx_test,
        y_true=stage_split.y_test,
        y_pred=y_pred_test,
        label_names=labels_stage1,
        max_print=cfg.max_print_errors,
        do_print=False,
        phase="stage1_test",
        out_path=errors_out_path,
        cfg=cfg,
    )

    stage_all = df_task["Stage"].astype(str).to_numpy()
    stage_names = sorted(set(stage_all.tolist()) - {"Benign"})
    stage_to_id = {s: i for i, s in enumerate(stage_names)}
    num_classes = 1 + len(stage_names)
    labels_all = ["Benign"] + stage_names

    def _build_gate_split(idx_split: np.ndarray, X_split: np.ndarray, p_split: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gate_mask = p_split >= float(stage1_threshold)
        if not gate_mask.any():
            return np.zeros((0, X_split.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        idx_gate = idx_split[gate_mask].astype(np.int64, copy=False)
        X_gate = X_split[gate_mask].astype(np.float32, copy=False)
        y_gate = np.zeros(len(idx_gate), dtype=np.int64)
        st = stage_all[idx_gate]
        for i, s in enumerate(st.tolist()):
            if s != "Benign":
                y_gate[i] = 1 + int(stage_to_id[str(s)])
        return X_gate, y_gate

    Xg_train, yg_train = _build_gate_split(stage_split.idx_train, stage_split.X_train, p_train)
    Xg_val, yg_val = _build_gate_split(stage_split.idx_val, stage_split.X_val, p_val)
    Xg_test, yg_test = _build_gate_split(stage_split.idx_test, stage_split.X_test, p_test)

    if len(Xg_train) == 0 or len(set(yg_train.tolist())) <= 1:
        raise RuntimeError("No usable gated samples to train Stage2+Benign.")

    stage2_model = ExtraTreesClassifier(
        n_estimators=int(cfg.tree_n_estimators),
        random_state=int(cfg.seed),
        n_jobs=-1,
        class_weight="balanced",
        max_features=str(cfg.tree_max_features),
    )
    stage2_model.fit(Xg_train, yg_train.astype(np.int64, copy=False))

    prob_val = stage2_model.predict_proba(Xg_val) if len(Xg_val) else None
    pred_val = stage2_model.predict(Xg_val).astype(np.int64) if len(Xg_val) else np.zeros((0,), dtype=np.int64)
    if prob_val is not None:
        prob_val = _align_proba_to_num_classes(prob_val, stage2_model.classes_, num_classes)
    prob_test = stage2_model.predict_proba(Xg_test) if len(Xg_test) else None
    pred_test = stage2_model.predict(Xg_test).astype(np.int64) if len(Xg_test) else np.zeros((0,), dtype=np.int64)
    if prob_test is not None:
        prob_test = _align_proba_to_num_classes(prob_test, stage2_model.classes_, num_classes)

    m_val = compute_metrics(y_true=yg_val, y_pred=pred_val, y_prob=prob_val, num_classes=num_classes) if len(yg_val) else None
    m_test = compute_metrics(y_true=yg_test, y_pred=pred_test, y_prob=prob_test, num_classes=num_classes) if len(yg_test) else None

    if m_val is not None:
        print("Stage2+Benign Val:", _macro_weighted_summary(yg_val, pred_val))
        print(format_confusion_matrix(m_val.cm, labels=labels_all))
    if m_test is not None:
        print("Stage2+Benign Test:", _macro_weighted_summary(yg_test, pred_test))
        print(format_confusion_matrix(m_test.cm, labels=labels_all))
        _print_and_collect_errors(
            df_task=df_task,
            idx_test=stage_split.idx_test[p_test >= float(stage1_threshold)],
            y_true=yg_test,
            y_pred=pred_test,
            label_names=labels_all,
            max_print=cfg.max_print_errors,
            do_print=False,
            phase="stage2_test",
            out_path=errors_out_path,
            cfg=cfg,
        )

    idx_test_all = stage_split.idx_test.astype(np.int64, copy=False)
    y_true_e2e = np.zeros(len(idx_test_all), dtype=np.int64)
    stage_true_test = stage_all[idx_test_all]
    for i, s in enumerate(stage_true_test.tolist()):
        if s != "Benign":
            y_true_e2e[i] = 1 + int(stage_to_id[str(s)])

    y_pred_e2e = np.zeros(len(idx_test_all), dtype=np.int64)
    gate_mask_test = p_test >= float(stage1_threshold)
    if gate_mask_test.any() and prob_test is not None and len(prob_test):
        benign_prob = prob_test[:, 0].astype(np.float32, copy=False)
        pred_mal = (1 + prob_test[:, 1:].argmax(axis=1)).astype(np.int64, copy=False) if num_classes > 1 else np.zeros(len(prob_test), dtype=np.int64)
        pred_corrected = np.where(benign_prob >= float(benign_return_min_conf), 0, pred_mal).astype(np.int64, copy=False)
        y_pred_e2e[np.where(gate_mask_test)[0].astype(np.int64)] = pred_corrected

    metric_e2e = compute_metrics(y_true=y_true_e2e, y_pred=y_pred_e2e, y_prob=None, num_classes=num_classes)
    print("End2End+Benign Test:", _macro_weighted_summary(y_true_e2e, y_pred_e2e))
    print(format_confusion_matrix(metric_e2e.cm, labels=labels_all))
    _print_and_collect_errors(
        df_task=df_task,
        idx_test=idx_test_all,
        y_true=y_true_e2e,
        y_pred=y_pred_e2e,
        label_names=labels_all,
        max_print=cfg.max_print_errors,
        do_print=False,
        phase="end2end_test",
        out_path=errors_out_path,
        cfg=cfg,
    )

def _tune_threshold_by_min_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_key = (-1.0, -1.0)
    for t in thresholds:
        y_pred = (prob_pos >= t).astype(np.int64)
        m = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=np.stack([1.0 - prob_pos, prob_pos], axis=1),
            num_classes=2,
        )
        key = (min(m.acc, m.f1, m.precision, m.recall), m.auc if not np.isnan(m.auc) else -1.0)
        if key > best_key:
            best_key = key
            best_t = float(t)
    return best_t


def _tune_threshold(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    objective: str,
    min_recall: float,
) -> float:
    objective = str(objective)
    if objective == "min":
        return _tune_threshold_by_min_metrics(y_true, prob_pos)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_key = (-1.0, -1.0)
    for t in thresholds:
        y_pred = (prob_pos >= t).astype(np.int64)
        m = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=np.stack([1.0 - prob_pos, prob_pos], axis=1),
            num_classes=2,
        )
        if objective == "f1":
            key = (float(m.f1), float(m.auc) if not np.isnan(m.auc) else -1.0)
        elif objective == "precision":
            if float(m.recall) + 1e-12 < float(min_recall):
                continue
            key = (float(m.precision), float(m.recall))
        else:
            raise ValueError(f"Unknown stage1 threshold objective: {objective}")
        if key > best_key:
            best_key = key
            best_t = float(t)
    if objective == "precision" and best_key[0] < 0:
        return _tune_threshold_by_min_metrics(y_true, prob_pos)
    return best_t


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.asarray(scores, dtype=np.float32)
    s = s[np.isfinite(s)]
    n = int(len(s))
    if n <= 0:
        return 1.0
    alpha = float(np.clip(alpha, 1e-6, 0.49))
    q = float(np.ceil((n + 1) * (1.0 - alpha)) / n)
    q = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(s, q, method="higher"))
    except TypeError:
        return float(np.quantile(s, q, interpolation="higher"))


def _tune_stage1_gate_threshold(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    cfg: ExperimentConfig,
) -> float:
    if hasattr(cfg, "fixed_stage1_threshold") and cfg.fixed_stage1_threshold is not None:
        return float(cfg.fixed_stage1_threshold)

    method = str(getattr(cfg, "stage1_gate_method", "threshold") or "threshold")
    if method != "conformal_fpr":
        return _tune_threshold(
            y_true=y_true,
            prob_pos=prob_pos,
            objective=cfg.stage1_threshold_objective,
            min_recall=cfg.stage1_min_recall,
        )

    y = y_true.astype(np.int64, copy=False)
    p = np.asarray(prob_pos, dtype=np.float32)
    neg = p[y == 0]
    if len(neg) <= 0:
        return _tune_threshold(
            y_true=y_true,
            prob_pos=prob_pos,
            objective=cfg.stage1_threshold_objective,
            min_recall=cfg.stage1_min_recall,
        )

    base_alpha = float(getattr(cfg, "stage1_fpr_budget", 0.001) or 0.001)
    for alpha in (base_alpha, base_alpha * 2.0, base_alpha * 4.0, base_alpha * 8.0, 0.02):
        thr = _conformal_quantile(neg, alpha=float(alpha))
        pred = (p >= float(thr)).astype(np.int64, copy=False)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        recall = float(tp / max(1, tp + fn))
        if recall + 1e-12 >= float(cfg.stage1_min_recall):
            return float(thr)

    return float(_conformal_quantile(neg, alpha=float(base_alpha)))


def _print_and_collect_errors(
    df_task: pd.DataFrame,
    idx_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    max_print: int,
    do_print: bool = True,
    phase: str | None = None,
    out_path: str | None = None,
    cfg: ExperimentConfig | None = None,
) -> pd.DataFrame:
    errors = np.where(y_true != y_pred)[0]
    if len(errors) == 0:
        return pd.DataFrame()

    n_print = len(errors) if max_print < 0 else min(len(errors), max_print)
    show = errors[:n_print]
    meta_cols = [c for c in META_COLS_DAPT if c in df_task.columns]
    meta_all = df_task.iloc[idx_test[errors]][meta_cols].copy()
    meta_all["y_true"] = [label_names[i] for i in y_true[errors]]
    meta_all["y_pred"] = [label_names[i] for i in y_pred[errors]]
    if "__row_id" in meta_all.columns and "row_1based" not in meta_all.columns:
        meta_all.insert(1, "row_1based", meta_all["__row_id"].astype(np.int64) + 1)
    if phase is not None:
        meta_all.insert(0, "phase", str(phase))
    if cfg is not None:
        meta_all.insert(0, "drop_stages", str(cfg.drop_stages))
        meta_all.insert(0, "seed", int(cfg.seed))
        meta_all.insert(0, "dataset", str(cfg.dataset))

    desired_meta = ["__row_id", "row_1based"] + [c for c in META_COLS_DAPT if c != "__row_id"] + ["y_true", "y_pred"]
    for c in desired_meta:
        if c not in meta_all.columns:
            meta_all[c] = np.nan
    prefix = ["dataset", "seed", "drop_stages", "phase"]
    for c in prefix:
        if c not in meta_all.columns:
            meta_all[c] = "" if c in ("dataset", "drop_stages", "phase") else 0
    meta_all = meta_all[prefix + desired_meta].copy()

    if out_path is not None:
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        exists = os.path.exists(out_path)
        meta_all.to_csv(out_path, index=False, mode="a", header=not exists)

    if cfg is not None and phase is not None:
        _update_hard_errors(meta_all, cfg)

    if do_print:
        print(f"Misclassified samples: {len(errors)} (showing {n_print})")
        meta_show = meta_all.iloc[:n_print].copy()
        print(meta_show.to_string(index=False))
    return meta_all


def _update_hard_errors(df_err: pd.DataFrame, cfg: ExperimentConfig) -> None:
    if df_err is None or len(df_err) == 0:
        return
    if "__row_id" not in df_err.columns or "phase" not in df_err.columns:
        return
    hard_path = os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_hard_errors.csv")
    now = int(time.time())
    keys = ["dataset", "phase", "__row_id"]
    keep_cols = keys + ["row_1based", "y_true", "y_pred", "seed", "drop_stages"]
    cur = df_err[keep_cols].copy()
    cur = cur.drop_duplicates(subset=keys)
    cur["n_miscls_add"] = 1
    cur["last_seen_ts"] = now
    cur = cur.rename(columns={"y_pred": "last_y_pred", "seed": "last_seed", "drop_stages": "last_drop_stages"})

    if os.path.exists(hard_path):
        try:
            old = pd.read_csv(hard_path)
        except Exception:
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()

    if len(old) == 0:
        out = cur[keys + ["row_1based", "y_true", "n_miscls_add", "last_seen_ts", "last_seed", "last_drop_stages", "last_y_pred"]].copy()
        out = out.rename(columns={"n_miscls_add": "n_miscls"})
        out.to_csv(hard_path, index=False)
        return

    if "n_miscls" not in old.columns:
        old["n_miscls"] = 0
    if "last_seen_ts" not in old.columns:
        old["last_seen_ts"] = 0
    if "last_seed" not in old.columns:
        old["last_seed"] = 0
    if "last_drop_stages" not in old.columns:
        old["last_drop_stages"] = ""
    if "last_y_pred" not in old.columns:
        old["last_y_pred"] = ""

    out = old.merge(cur, on=keys, how="outer", suffixes=("", "_new"))
    out["row_1based"] = out["row_1based"].fillna(out.get("row_1based_new"))
    out["y_true"] = out["y_true"].fillna(out.get("y_true_new"))
    out["n_miscls"] = out["n_miscls"].fillna(0).astype(np.int64) + out["n_miscls_add"].fillna(0).astype(np.int64)
    out["last_seen_ts"] = out["last_seen_ts_new"].fillna(out["last_seen_ts"]).astype(np.int64)
    out["last_seed"] = out["last_seed_new"].fillna(out["last_seed"]).astype(np.int64)
    out["last_drop_stages"] = out["last_drop_stages_new"].fillna(out["last_drop_stages"]).astype(str)
    out["last_y_pred"] = out["last_y_pred_new"].fillna(out["last_y_pred"]).astype(str)

    drop_cols = [c for c in out.columns if c.endswith("_new") or c == "n_miscls_add"]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out = out.sort_values(by=["n_miscls", "phase"], ascending=[False, True]).reset_index(drop=True)
    out.to_csv(hard_path, index=False)


def _save_checkpoint(
    model,
    cfg: ExperimentConfig,
    split,
    class_names: list[str],
    best_val: Metrics,
    out_path: str,
    extra: dict | None = None,
) -> None:
    ckpt_dir = os.path.dirname(out_path) or "."
    os.makedirs(ckpt_dir, exist_ok=True)

    def _to_safe(x):
        if x is None or isinstance(x, (bool, int, float, str, bytes)):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).detach().cpu()
        if isinstance(x, (list, tuple)):
            return [_to_safe(v) for v in x]
        if isinstance(x, dict):
            return {k: _to_safe(v) for k, v in x.items()}
        return pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)

    def _to_json_safe(x):
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, bytes):
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return x.item()
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [_to_json_safe(v) for v in x]
        if isinstance(x, dict):
            return {str(k): _to_json_safe(v) for k, v in x.items()}
        return str(type(x).__name__)

    payload = {
        "model_type": cfg.model_type,
        "config": asdict(cfg),
        "feature_cols": split.feature_cols,
        "scaler_mean": _to_safe(split.scaler.mean_.astype(np.float32)),
        "scaler_scale": _to_safe(split.scaler.scale_.astype(np.float32)),
        "class_names": class_names,
        "best_val": {
            "acc": best_val.acc,
            "f1": best_val.f1,
            "precision": best_val.precision,
            "recall": best_val.recall,
            "auc": best_val.auc,
            "cm": _to_safe(best_val.cm),
        },
    }
    payload["model_pickle"] = _to_safe(model)
    if extra:
        payload["extra"] = _to_safe(extra)
    torch.save(payload, out_path)

    json_path = out_path + ".json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": cfg.model_type,
                    "config": asdict(cfg),
                    "class_names": class_names,
                    "best_val": {
                        "acc": float(best_val.acc),
                        "f1": float(best_val.f1),
                        "precision": float(best_val.precision),
                        "recall": float(best_val.recall),
                        "auc": float(best_val.auc) if not np.isnan(best_val.auc) else None,
                    },
                    "extra": _to_json_safe(extra) if extra else None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass


def run_stage_binary(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    errors_out_path: str | None = None,
    fixed_split: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    selected_feature_cols: list[str] | None = None,
) -> tuple[object, object, float]:
    if cfg.dataset == "dapt2020":
        use_act = bool(str(getattr(cfg, "stage2_label", "stage")) == "activity")
        df_task, y, feature_cols = make_stage_task(
            df,
            use_activity_as_stage=use_act,
            selected_feature_cols=selected_feature_cols,
        )
    else:
        df_task, y, feature_cols = make_stage_task(df, selected_feature_cols=selected_feature_cols)
    if fixed_split is None:
        split = split_scale(
            df=df_task,
            y=y,
            feature_cols=feature_cols,
            seed=cfg.seed,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
        )
    else:
        idx_train, idx_val, idx_test = fixed_split
        split = scale_by_indices(
            df=df_task,
            y=y,
            feature_cols=feature_cols,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
        )

    y_stage1_train = split.y_train.astype(np.int64, copy=False)
    X_stage1_train = split.X_train
    row_id_stage1_train = split.row_id_train.astype(np.int64, copy=False)
    counts = np.bincount(y_stage1_train, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    sample_weight = w[y_stage1_train].astype(np.float32)

    stage_train = df_task.iloc[split.idx_train]["Stage"].astype(str).to_numpy()
    if cfg.dataset == "dapt2020" and str(getattr(cfg, "stage2_label", "stage")) == "activity":
        stage_train = df_task.iloc[split.idx_train]["Activity"].astype(str).to_numpy()
        stage_train = np.where(stage_train == "Normal", "Benign", stage_train)
    is_mal = y_stage1_train == 1
    if is_mal.any():
        mal_stage = stage_train[is_mal]
        stages, inv, stage_counts = np.unique(mal_stage, return_inverse=True, return_counts=True)
        stage_weights = (stage_counts.sum() / (len(stages) * stage_counts)).astype(np.float32)
        if cfg.dataset == "dapt2020":
            if getattr(cfg, "oversample_rare_classes", False):
                # Do not clip, or clip to a much higher value so rare classes get enough attention
                stage_weights = np.clip(stage_weights, 1.0, 500.0).astype(np.float32, copy=False)
                # Give an extra boost to extremely rare classes
                stage_weights[stage_counts < 50] *= 5.0
            else:
                stage_weights = np.clip(stage_weights, 1.0, 12.0).astype(np.float32, copy=False)
        elif cfg.dataset == "earlycrow":
            stage_weights = np.clip(stage_weights, 1.0, 300.0).astype(np.float32, copy=False)
            stage_weights[stage_counts <= 512] *= 1.75
            stage_weights[stage_counts <= 128] *= 2.5
            stage_weights[stage_counts <= 32] *= 3.5
            for i, stage_name in enumerate(stages.tolist()):
                if str(stage_name) in {"onionduke1", "poisonivy1", "zebrocy1", "zebrocy2", "zebrocy3"}:
                    stage_weights[i] = np.float32(stage_weights[i] * 1.75)
        stage_w = np.ones(len(stage_train), dtype=np.float32)
        stage_w[is_mal] = stage_weights[inv]
        boost = np.ones(len(stage_train), dtype=np.float32)
        boost[is_mal & (stage_train == "Fuzzers")] *= np.float32(2.0)
        boost[is_mal & (stage_train == "Analysis")] *= np.float32(2.0)
        stage_w = stage_w * boost
        stage_w = stage_w / float(np.mean(stage_w))
        sample_weight = (sample_weight * stage_w).astype(np.float32, copy=False)

        # Protect ultra-rare malicious subtypes in Stage-I so they are not routed away
        # before Stage-II can recover them under suspicious_unknown inference.
        auto_stage1_rare_augment = bool(
            getattr(cfg, "oversample_rare_classes", False)
            or (
                str(getattr(cfg, "inference_policy", "original")) == "suspicious_unknown"
                and int(np.min(stage_counts)) < 20
            )
        )
        if auto_stage1_rare_augment:
            target_n = max(16, min(int(getattr(cfg, "oversample_target_count", 50)), 64))
            if cfg.dataset == "earlycrow":
                target_n = max(target_n, min(max(96, int(getattr(cfg, "oversample_target_count", 50))), 192))
            rng_stage1 = np.random.default_rng(int(cfg.seed) + 17)
            X_parts = [X_stage1_train]
            y_parts = [y_stage1_train]
            w_parts = [sample_weight]
            row_id_parts = [row_id_stage1_train]
            for stage_name, count in zip(stages.tolist(), stage_counts.tolist()):
                count = int(count)
                if count <= 0 or count >= target_n:
                    continue
                idx_stage = np.where(is_mal & (stage_train == str(stage_name)))[0].astype(np.int64, copy=False)
                if len(idx_stage) == 0:
                    continue
                need = int(target_n - count)
                dup_idx = rng_stage1.choice(idx_stage, size=need, replace=True)
                X_dup = X_stage1_train[dup_idx].copy()
                X_dup += rng_stage1.normal(loc=0.0, scale=0.01, size=X_dup.shape).astype(np.float32, copy=False)
                X_parts.append(X_dup.astype(np.float32, copy=False))
                y_parts.append(y_stage1_train[dup_idx].astype(np.int64, copy=False))
                w_parts.append(sample_weight[dup_idx].astype(np.float32, copy=False))
                row_id_parts.append(row_id_stage1_train[dup_idx].astype(np.int64, copy=False))
            if len(X_parts) > 1:
                X_stage1_train = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
                y_stage1_train = np.concatenate(y_parts, axis=0).astype(np.int64, copy=False)
                sample_weight = np.concatenate(w_parts, axis=0).astype(np.float32, copy=False)
                row_id_stage1_train = np.concatenate(row_id_parts, axis=0).astype(np.int64, copy=False)

    if _can_use_xgb_cuda():
        model = _fit_xgb_binary(
            X_train=X_stage1_train,
            y_train=y_stage1_train,
            w_train=sample_weight,
            X_val=split.X_val,
            y_val=split.y_val.astype(np.int64, copy=False),
            cfg=cfg,
            row_id_train=row_id_stage1_train,
            row_id_val=split.row_id_val,
        )
        val_prob = model.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
        threshold = _tune_stage1_gate_threshold(split.y_val, val_prob, cfg)
    else:
        step = 50
        patience_evals = 6
        min_iters = 100

        model = HistGradientBoostingClassifier(
            learning_rate=cfg.hgb_learning_rate,
            max_iter=step,
            warm_start=True,
            random_state=cfg.seed,
        )

        best_obj = -1.0
        best_blob: bytes | None = None
        best_threshold = 0.5
        best_val_prob: np.ndarray | None = None
        no_improve = 0

        max_iter = int(cfg.hgb_max_iter)
        for it in range(step, max_iter + 1, step):
            model.set_params(max_iter=it)
            model.fit(X_stage1_train, y_stage1_train, sample_weight=sample_weight)
            val_prob = model.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
            threshold = _tune_stage1_gate_threshold(split.y_val, val_prob, cfg)
            val_metric = compute_metrics(
                y_true=split.y_val,
                y_pred=(val_prob >= threshold).astype(np.int64),
                y_prob=np.stack([1.0 - val_prob, val_prob], axis=1),
                num_classes=2,
            )
            obj = float(min(val_metric.acc, val_metric.f1, val_metric.precision, val_metric.recall))
            if obj > best_obj + 1e-12:
                best_obj = obj
                best_threshold = float(threshold)
                best_val_prob = val_prob
                best_blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                no_improve = 0
            else:
                no_improve += 1
                if it >= min_iters and no_improve >= patience_evals:
                    break

        if best_blob is None:
            model.set_params(max_iter=max_iter, warm_start=False)
            model.fit(X_stage1_train, y_stage1_train, sample_weight=sample_weight)
            val_prob = model.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
            threshold = _tune_stage1_gate_threshold(split.y_val, val_prob, cfg)
        else:
            model = pickle.loads(best_blob)
            val_prob = (
                best_val_prob
                if best_val_prob is not None
                else model.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
            )
            threshold = best_threshold
    train_prob_pos = model.predict_proba(split.X_train)[:, 1].astype(np.float32, copy=False)
    test_prob_pos = model.predict_proba(split.X_test)[:, 1].astype(np.float32, copy=False)

    def _pos_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(np.int64, copy=False)
        y_pred = y_pred.astype(np.int64, copy=False)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        return float(tp / max(1, tp + fn))

    y_pred = (test_prob_pos >= threshold).astype(np.int64)

    test_metric = compute_metrics(
        y_true=split.y_test,
        y_pred=y_pred,
        y_prob=np.stack([1.0 - test_prob_pos, test_prob_pos], axis=1),
        num_classes=2,
    )
    best_val = compute_metrics(
        y_true=split.y_val,
        y_pred=(val_prob >= threshold).astype(np.int64),
        y_prob=np.stack([1.0 - val_prob, val_prob], axis=1),
        num_classes=2,
    )

    extra: dict = {
        "threshold": float(threshold),
        "backend": (
            "xgb_dask_cpu"
            if _use_federated_dask_backend(cfg)
            else "xgb_cuda"
        ),
        "hgb_n_iter": int(getattr(model, "n_iter_", 0) or 0),
        "xgb_best_iteration": int(getattr(model, "best_iteration", -1) or -1),
    }

    use_fuzz_rescue = False
    fuzz_model = None
    fuzz_thr = None
    fuzz_prob_val = None
    fuzz_prob_test = None
    fuzz_t_low = None
    if "Stage" in df_task.columns and _can_use_xgb_cuda():
        try:
            stage_train_all = df_task.iloc[split.idx_train]["Stage"].astype(str).to_numpy()
            stage_val_all = df_task.iloc[split.idx_val]["Stage"].astype(str).to_numpy()
            stage_test_all = df_task.iloc[split.idx_test]["Stage"].astype(str).to_numpy()
            y_f_train = (stage_train_all == "Fuzzers").astype(np.int64)
            y_f_val = (stage_val_all == "Fuzzers").astype(np.int64)
            y_f_test = (stage_test_all == "Fuzzers").astype(np.int64)

            pos_idx = np.where(y_f_train == 1)[0]
            neg_idx = np.where(y_f_train == 0)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                rng = np.random.default_rng(cfg.seed + 7)
                max_neg = min(len(neg_idx), max(50_000, int(8 * len(pos_idx))))
                if len(neg_idx) > max_neg:
                    neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
                sel = np.concatenate([pos_idx, neg_idx]).astype(np.int64, copy=False)
                rng.shuffle(sel)

                Xf_tr = split.X_train[sel]
                yf_tr = y_f_train[sel].astype(np.int64, copy=False)
                sw_f = np.where(yf_tr == 1, float(max(1.0, (len(yf_tr) - int(yf_tr.sum())) / max(1, int(yf_tr.sum())))), 1.0).astype(
                    np.float32
                )
                fuzz_model = _fit_xgb_binary(
                    X_train=Xf_tr,
                    y_train=yf_tr,
                    w_train=sw_f,
                    X_val=split.X_val,
                    y_val=y_f_val.astype(np.int64, copy=False),
                    cfg=cfg,
                )
                fuzz_prob_val = fuzz_model.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
                fuzz_prob_test = fuzz_model.predict_proba(split.X_test)[:, 1].astype(np.float32, copy=False)

                base_val_pred = (val_prob >= float(threshold)).astype(np.int64)
                base_val = compute_metrics(
                    y_true=split.y_val,
                    y_pred=base_val_pred,
                    y_prob=np.stack([1.0 - val_prob, val_prob], axis=1),
                    num_classes=2,
                )

                t_low_min = float(max(0.01, min(float(threshold), float(threshold) - 0.25)))
                t_low_max = float(max(t_low_min, min(0.99, float(threshold) - 0.01)))
                t_low_cand = np.array(
                    sorted(set(np.round(np.linspace(t_low_min, t_low_max, 19), 3).tolist())), dtype=np.float32
                )
                thr_cand = np.linspace(0.01, 0.99, 99, dtype=np.float32)
                best = None
                for t_low in t_low_cand:
                    near = (val_prob >= float(t_low)) & (val_prob < float(threshold))
                    if not near.any():
                        continue
                    for thr in thr_cand:
                        rescue = near & (fuzz_prob_val >= float(thr))
                        pred_val = base_val_pred | rescue.astype(np.int64)
                        if _pos_recall(split.y_val, pred_val) + 1e-12 < float(cfg.stage1_min_recall):
                            continue
                        m = compute_metrics(
                            y_true=split.y_val,
                            y_pred=pred_val,
                            y_prob=np.stack([1.0 - val_prob, val_prob], axis=1),
                            num_classes=2,
                        )
                        key = (float(m.f1), float(m.acc))
                        if best is None or key > best[0]:
                            best = (key, float(t_low), float(thr), m)

                if best is not None:
                    (_, _), best_t_low, best_thr, best_val_rescue = best
                    if float(best_val_rescue.f1) > float(best_val.f1) + 1e-6:
                        use_fuzz_rescue = True
                        extra = extra | {"fuzz_rescue_enabled": True, "fuzz_model": fuzz_model}
                        extra = extra | {"fuzz_rescue_t_low": float(best_t_low), "fuzz_rescue_threshold": float(best_thr)}
                        fuzz_thr = float(best_thr)
                        fuzz_t_low = float(best_t_low)
                        near_test = (test_prob_pos >= float(best_t_low)) & (test_prob_pos < float(threshold))
                        rescue_test = near_test & (fuzz_prob_test >= float(fuzz_thr))
                        y_pred = ((test_prob_pos >= float(threshold)) | rescue_test).astype(np.int64)
                        test_metric = compute_metrics(
                            y_true=split.y_test,
                            y_pred=y_pred,
                            y_prob=np.stack([1.0 - test_prob_pos, test_prob_pos], axis=1),
                            num_classes=2,
                        )
                        best_val = best_val_rescue
        except Exception:
            use_fuzz_rescue = False
            fuzz_model = None
            fuzz_thr = None

    use_guard = False
    guard_model = None
    guard_thr = None
    best_t1 = float(threshold)
    best_t2 = None
    guard_prob_val = None
    guard_prob_test = None

    try:
        if getattr(cfg, "oversample_rare_classes", False) and cfg.dataset == "dapt2020":
            pass
        else:
            gate_thr_train = float(max(0.05, min(0.95, float(threshold) - 0.20)))
            gate_mask_train = train_prob_pos >= gate_thr_train
            if gate_mask_train.any():
                Xg = split.X_train[gate_mask_train]
            yg = split.y_train[gate_mask_train].astype(np.int64, copy=False)
            idx_pos = np.where(yg == 1)[0]
            idx_neg = np.where(yg == 0)[0]
            if len(idx_pos) > 0 and len(idx_neg) > 0:
                rng = np.random.default_rng(cfg.seed)
                max_neg = min(len(idx_neg), int(3 * len(idx_pos)))
                if len(idx_neg) > max_neg:
                    idx_neg = rng.choice(idx_neg, size=max_neg, replace=False)
                sel = np.concatenate([idx_pos, idx_neg]).astype(np.int64, copy=False)
                rng.shuffle(sel)
                Xg = Xg[sel]
                yg = yg[sel]

                counts_g = np.bincount(yg.astype(np.int64), minlength=2).astype(np.float64)
                counts_g = np.maximum(counts_g, 1.0)
                w_g = counts_g.sum() / (2.0 * counts_g)
                sw_g = w_g[yg.astype(np.int64)].astype(np.float32)

                guard_model = HistGradientBoostingClassifier(
                    learning_rate=float(cfg.hgb_learning_rate),
                    max_iter=max(200, int(cfg.hgb_max_iter // 2)),
                    random_state=cfg.seed,
                )
                guard_model.fit(Xg, yg, sample_weight=sw_g)

                guard_prob_val = guard_model.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
                guard_prob_test = guard_model.predict_proba(split.X_test)[:, 1].astype(np.float32, copy=False)

                base_val_pred = (val_prob >= float(threshold)).astype(np.int64)
                base_val = compute_metrics(
                    y_true=split.y_val,
                    y_pred=base_val_pred,
                    y_prob=np.stack([1.0 - val_prob, val_prob], axis=1),
                    num_classes=2,
                )

                t1_min = float(max(0.05, min(0.95, float(threshold) - 0.20)))
                t1_max = float(max(t1_min, min(0.95, float(threshold) + 0.08)))
                t1_cand = np.array(sorted(set(np.round(np.linspace(t1_min, t1_max, 27), 3).tolist())), dtype=np.float32)
                t2_cand = np.linspace(0.01, 0.99, 99, dtype=np.float32)

                best_pair = None
                for t1 in t1_cand:
                    gate_val = val_prob >= float(t1)
                    if not gate_val.any():
                        continue
                    for t2 in t2_cand:
                        pred_val = (gate_val & (guard_prob_val >= float(t2))).astype(np.int64)
                        if _pos_recall(split.y_val, pred_val) + 1e-12 < float(cfg.stage1_min_recall):
                            continue
                        m = compute_metrics(
                            y_true=split.y_val,
                            y_pred=pred_val,
                            y_prob=np.stack([1.0 - val_prob, val_prob], axis=1),
                            num_classes=2,
                        )
                        key = (float(m.f1), float(m.acc))
                        if best_pair is None or key > best_pair[0]:
                            best_pair = (key, float(t1), float(t2), m)

                if best_pair is not None:
                    (_, _), best_t1, best_t2, best_val_guard = best_pair
                    if float(best_val_guard.f1) > float(best_val.f1) + 1e-6:
                        use_guard = True
                        use_fuzz_rescue = False
                        extra = extra | {"fuzz_rescue_enabled": False}
                        guard_thr = float(best_t2)
                        guard_pred_test = ((test_prob_pos >= float(best_t1)) & (guard_prob_test >= float(best_t2))).astype(
                            np.int64
                        )
                        y_pred = guard_pred_test
                        threshold = float(best_t1)
                        test_metric = compute_metrics(
                            y_true=split.y_test,
                            y_pred=y_pred,
                            y_prob=np.stack([1.0 - test_prob_pos, test_prob_pos], axis=1),
                            num_classes=2,
                        )
                        best_val = best_val_guard
    except Exception:
        use_guard = False

    try:
        if getattr(cfg, "oversample_rare_classes", False) and cfg.dataset == "dapt2020":
            pass
        elif (
            use_guard
            and fuzz_model is not None
            and fuzz_prob_val is not None
            and fuzz_prob_test is not None
            and guard_prob_val is not None
            and guard_prob_test is not None
            and best_t2 is not None
        ):
            base_val_pred = ((val_prob >= float(threshold)) & (guard_prob_val >= float(best_t2))).astype(np.int64)
            t_low_min = float(max(0.01, min(float(threshold), float(threshold) - 0.25)))
            t_low_max = float(max(t_low_min, min(0.99, float(threshold) - 0.01)))
            t_low_cand = np.array(
                sorted(set(np.round(np.linspace(t_low_min, t_low_max, 19), 3).tolist())), dtype=np.float32
            )
            thr_cand = np.linspace(0.01, 0.99, 99, dtype=np.float32)
            best_combo = None
            base_mask = base_val_pred == 0
            for t_low in t_low_cand:
                near = base_mask & (val_prob >= float(t_low)) & (val_prob < float(threshold))
                if not near.any():
                    continue
                for thr in thr_cand:
                    rescue = near & (fuzz_prob_val >= float(thr))
                    pred_val = base_val_pred | rescue.astype(np.int64)
                    if _pos_recall(split.y_val, pred_val) + 1e-12 < float(cfg.stage1_min_recall):
                        continue
                    m = compute_metrics(
                        y_true=split.y_val,
                        y_pred=pred_val,
                        y_prob=np.stack([1.0 - val_prob, val_prob], axis=1),
                        num_classes=2,
                    )
                    key = (float(m.f1), float(m.acc))
                    if best_combo is None or key > best_combo[0]:
                        best_combo = (key, float(t_low), float(thr), m)

            if best_combo is not None:
                (_, _), best_t_low, best_thr, best_val_combo = best_combo
                if float(best_val_combo.f1) > float(best_val.f1) + 1e-6:
                    use_fuzz_rescue = True
                    fuzz_thr = float(best_thr)
                    fuzz_t_low = float(best_t_low)
                    extra = extra | {"fuzz_rescue_enabled": True, "fuzz_model": fuzz_model}
                    extra = extra | {
                        "fuzz_rescue_mode": "over_guard",
                        "fuzz_rescue_t_low": float(best_t_low),
                        "fuzz_rescue_threshold": float(best_thr),
                    }
                    base_test_pred = ((test_prob_pos >= float(threshold)) & (guard_prob_test >= float(best_t2))).astype(
                        np.int64
                    )
                    base_mask_test = base_test_pred == 0
                    near_test = base_mask_test & (test_prob_pos >= float(best_t_low)) & (test_prob_pos < float(threshold))
                    rescue_test = near_test & (fuzz_prob_test >= float(best_thr))
                    y_pred = (base_test_pred | rescue_test.astype(np.int64)).astype(np.int64)
                    test_metric = compute_metrics(
                        y_true=split.y_test,
                        y_pred=y_pred,
                        y_prob=np.stack([1.0 - test_prob_pos, test_prob_pos], axis=1),
                        num_classes=2,
                    )
                    best_val = best_val_combo
    except Exception:
        pass

    labels = ["Benign", "Malicious"]

    base_val_pred = ((val_prob >= float(threshold)) & (guard_prob_val >= float(best_t2))).astype(np.int64) if (use_guard and guard_prob_val is not None and best_t2 is not None) else (val_prob >= float(threshold)).astype(np.int64)
    if use_fuzz_rescue and fuzz_prob_val is not None and fuzz_t_low is not None and fuzz_thr is not None:
        near_val = (base_val_pred == 0) & (val_prob >= float(fuzz_t_low)) & (val_prob < float(threshold))
        rescue_val = near_val & (fuzz_prob_val >= float(fuzz_thr))
        y_pred_val = (base_val_pred | rescue_val.astype(np.int64)).astype(np.int64)
    else:
        y_pred_val = base_val_pred.astype(np.int64, copy=False)

    s1_val = _macro_weighted_summary(split.y_val, y_pred_val)
    s1_val["macro_acc"] = _macro_acc_from_cm(best_val.cm)
    s1_val["fpr"] = _stage1_fpr_from_cm(best_val.cm)
    s1_val["macro_fpr"], s1_val["weighted_fpr"] = _fpr_macro_weighted_from_cm(best_val.cm)
    s1_val["macro_auc"], s1_val["weighted_auc"] = _auc_macro_weighted_ovr(
        split.y_val,
        np.stack([1.0 - val_prob, val_prob], axis=1),
    )
    print("Stage1 Val:", s1_val)
    print(format_confusion_matrix(best_val.cm, labels=labels))
    s1_test = _macro_weighted_summary(split.y_test, y_pred)
    s1_test["macro_acc"] = _macro_acc_from_cm(test_metric.cm)
    s1_test["fpr"] = _stage1_fpr_from_cm(test_metric.cm)
    s1_test["macro_fpr"], s1_test["weighted_fpr"] = _fpr_macro_weighted_from_cm(test_metric.cm)
    s1_test["macro_auc"], s1_test["weighted_auc"] = _auc_macro_weighted_ovr(
        split.y_test,
        np.stack([1.0 - test_prob_pos, test_prob_pos], axis=1),
    )
    print("Stage1 Test:", s1_test)
    print(format_confusion_matrix(test_metric.cm, labels=labels))
    _write_json_silent(
        os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_seed{cfg.seed}_stage1_metrics.json"),
        {
            "dataset": cfg.dataset,
            "seed": int(cfg.seed),
            "drop_stages": str(cfg.drop_stages),
            "labels": labels,
            "metrics": s1_test,
            "confusion_matrix": test_metric.cm.tolist(),
        },
    )

    fp = (split.y_test == 0) & (y_pred == 1)
    fn = (split.y_test == 1) & (y_pred == 0)
    _print_and_collect_errors(
        df_task=df_task,
        idx_test=split.idx_test,
        y_true=split.y_test,
        y_pred=y_pred,
        label_names=["Benign", "Malicious"],
        max_print=cfg.max_print_errors,
        do_print=False,
        phase="stage1_test",
        out_path=errors_out_path,
        cfg=cfg,
    )
    ensure_dir(cfg.checkpoint_dir)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_stage_binary_seed{cfg.seed}_best.pt")
    if use_guard and guard_model is not None and guard_thr is not None:
        extra = extra | {
            "threshold": float(threshold),
            "guard_enabled": True,
            "guard_model": guard_model,
            "guard_threshold": float(guard_thr),
        }
    else:
        extra = extra | {"threshold": float(threshold), "guard_enabled": False}
    _save_checkpoint(
        model=model,
        cfg=cfg,
        split=split,
        class_names=["Benign", "Malicious"],
        best_val=best_val,
        out_path=ckpt_path,
        extra=extra,
    )
    return model, split, float(threshold)

def _force_keep(extra: dict | None, X_gate: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if extra is None:
        return None, None
    lat_m = extra.get("lateral_model", None)
    lat_t = extra.get("lateral_threshold", None)
    lat_force = None
    if lat_m is not None and lat_t is not None:
        lat_force = lat_m.predict_proba(X_gate)[:, 1].astype(np.float32, copy=False) >= float(lat_t)
    ex_m = extra.get("dataex_model", None)
    ex_t = extra.get("dataex_threshold", None)
    ex_force = None
    if ex_m is not None and ex_t is not None:
        ex_force = ex_m.predict_proba(X_gate)[:, 1].astype(np.float32, copy=False) >= float(ex_t)

    ovr_models = extra.get("ovr_models", None)
    ovr_thresholds = extra.get("ovr_thresholds", None)
    if ovr_models is not None and ovr_thresholds is not None:
        ovr_force = None
        for m, thr in zip(ovr_models, ovr_thresholds):
            p = m.predict_proba(X_gate)[:, 1].astype(np.float32, copy=False)
            mask = p >= float(thr)
            if ovr_force is None:
                ovr_force = mask
            else:
                ovr_force = ovr_force | mask
        if ovr_force is not None:
            if ex_force is None:
                ex_force = ovr_force
            else:
                ex_force = ex_force | ovr_force
    ovr_all_models = extra.get("ovr_all_models", None)
    ovr_all_thresholds = extra.get("ovr_all_thresholds", None)
    if ovr_all_models is not None and ovr_all_thresholds is not None:
        ovr_force = None
        for m, thr in zip(ovr_all_models, ovr_all_thresholds):
            p = m.predict_proba(X_gate)[:, 1].astype(np.float32, copy=False)
            mask = p >= float(thr)
            if ovr_force is None:
                ovr_force = mask
            else:
                ovr_force = ovr_force | mask
        if ovr_force is not None:
            if ex_force is None:
                ex_force = ovr_force
            else:
                ex_force = ex_force | ovr_force
    ovr_full_models = extra.get("ovr_full_models", None)
    ovr_full_thresholds = extra.get("ovr_full_thresholds", None)
    if ovr_full_models is not None and ovr_full_thresholds is not None:
        ovr_force = None
        for m, thr in zip(ovr_full_models, ovr_full_thresholds):
            p = m.predict_proba(X_gate)[:, 1].astype(np.float32, copy=False)
            mask = p >= float(thr)
            if ovr_force is None:
                ovr_force = mask
            else:
                ovr_force = ovr_force | mask
        if ovr_force is not None:
            if ex_force is None:
                ex_force = ovr_force
            else:
                ex_force = ex_force | ovr_force
    return lat_force, ex_force


def _fit_extratrees_with_early_stop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ExperimentConfig,
) -> ExtraTreesClassifier:
    max_trees = int(cfg.tree_n_estimators)
    if max_trees <= 0:
        raise ValueError("tree_n_estimators must be > 0")

    step = 250
    min_trees = min(1000, max_trees)
    patience_evals = 3

    y_train_i64 = y_train.astype(np.int64, copy=False)
    classes = np.unique(y_train_i64)
    counts = np.bincount(y_train_i64, minlength=int(classes.max() + 1) if len(classes) else 0).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_weight = {int(c): float(counts.sum() / (len(classes) * counts[int(c)])) for c in classes}

    model = ExtraTreesClassifier(
        n_estimators=min(step, max_trees),
        warm_start=True,
        random_state=cfg.seed,
        n_jobs=-1,
        class_weight=class_weight,
        max_features=cfg.tree_max_features,
    )

    best_key: tuple[float, float] | None = None
    best_blob: bytes | None = None
    no_improve = 0

    def _eval_key(prob: np.ndarray) -> tuple[float, ...]:
        y_pred = prob.argmax(axis=1).astype(np.int64, copy=False)
        m = compute_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_prob=prob,
            num_classes=int(prob.shape[1]),
        )
        if cfg.dataset == "cic2024":
            return (float(m.f1), float(m.recall), float(m.precision), float(m.acc))
        return (float(min(m.precision, m.recall, m.f1)), float(m.acc))

    for n_estimators in range(model.n_estimators, max_trees + 1, step):
        n_estimators = int(min(n_estimators, max_trees))
        model.set_params(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val).astype(np.float32, copy=False)
        key = _eval_key(prob)

        if best_key is None or key > best_key:
            best_key = key
            best_blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            no_improve = 0
        else:
            no_improve += 1
            if n_estimators >= min_trees and no_improve >= patience_evals:
                break

        if n_estimators >= max_trees:
            break

    if best_blob is None:
        return model
    return pickle.loads(best_blob)


def _fit_hgb_multiclass_with_early_stop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    cfg: ExperimentConfig,
) -> HistGradientBoostingClassifier:
    step = 50
    patience_evals = 6
    min_iters = 150

    model = HistGradientBoostingClassifier(
        learning_rate=cfg.hgb_learning_rate,
        max_iter=step,
        warm_start=True,
        random_state=cfg.seed,
    )

    best_key: tuple[float, float] | None = None
    best_blob: bytes | None = None
    no_improve = 0

    def _eval_key(prob: np.ndarray) -> tuple[float, ...]:
        y_pred = prob.argmax(axis=1).astype(np.int64, copy=False)
        m = compute_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_prob=prob,
            num_classes=int(num_classes),
        )
        if cfg.dataset == "cic2024":
            return (float(m.f1), float(m.recall), float(m.precision), float(m.acc))
        return (float(min(m.precision, m.recall, m.f1)), float(m.acc))

    max_iter = int(cfg.hgb_max_iter)
    for it in range(step, max_iter + 1, step):
        model.set_params(max_iter=int(it))
        model.fit(X_train, y_train, sample_weight=w_train)
        prob = model.predict_proba(X_val).astype(np.float32, copy=False)
        key = _eval_key(prob)
        if best_key is None or key > best_key:
            best_key = key
            best_blob = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            no_improve = 0
        else:
            no_improve += 1
            if it >= min_iters and no_improve >= patience_evals:
                break
    if best_blob is None:
        model.set_params(max_iter=max_iter, warm_start=False)
        model.fit(X_train, y_train, sample_weight=w_train)
        return model
    return pickle.loads(best_blob)


class _LabeledArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(int(self.y[idx]), dtype=torch.long)


class _UnlabeledArrayDataset(Dataset):
    def __init__(self, X: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.X[idx])


def _train_repr_mlp_semi(
    X_l: np.ndarray,
    y_l: np.ndarray,
    X_u: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    cfg: ExperimentConfig,
    sample_weight: np.ndarray | None = None,
    hidden_dims: tuple[int, ...] = (512, 512, 256),
    dropout: float = 0.15,
) -> tuple[_TorchTabularWrapper, np.ndarray, Metrics]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_l = np.asarray(X_l, dtype=np.float32)
    y_l = np.asarray(y_l, dtype=np.int64)
    X_u = np.asarray(X_u, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)
    if len(X_l) == 0:
        raise RuntimeError("Representation stage2 received no labeled samples.")

    if sample_weight is None or len(sample_weight) != len(y_l):
        counts = np.bincount(y_l, minlength=int(num_classes)).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        sample_weight = (counts.sum() / (num_classes * counts))[y_l].astype(np.float32)
    sample_weight = np.asarray(sample_weight, dtype=np.float32)
    sample_weight = sample_weight / max(1e-6, float(sample_weight.mean()))

    labeled_ds = _LabeledArrayDataset(X_l, y_l)
    unlabeled_ds = _UnlabeledArrayDataset(X_u)
    weights_t = torch.as_tensor(sample_weight, dtype=torch.float32)
    labeled_bs = max(128, min(int(cfg.batch_size), max(128, len(y_l) // 8 if len(y_l) >= 1024 else len(y_l))))
    unlabeled_bs = max(labeled_bs, min(int(cfg.batch_size), max(256, len(X_u) // 8 if len(X_u) >= 1024 else max(256, len(X_u)))))
    sampler = WeightedRandomSampler(weights=weights_t, num_samples=max(len(y_l), labeled_bs * 16), replacement=True)
    labeled_loader = DataLoader(labeled_ds, batch_size=labeled_bs, sampler=sampler, drop_last=False)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=unlabeled_bs, shuffle=True, drop_last=False) if len(X_u) > 0 else None

    model = TabularMLP(
        input_dim=int(X_l.shape[1]),
        num_classes=int(num_classes),
        hidden_dims=hidden_dims,
        dropout=float(dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(4, min(int(cfg.epochs), 18)))
    class_counts = np.bincount(y_l, minlength=int(num_classes)).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    class_w = torch.as_tensor(class_counts.sum() / (num_classes * class_counts), dtype=torch.float32, device=device)

    best_key: tuple[float, float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_prob = np.zeros((len(X_val), int(num_classes)), dtype=np.float32)
    best_metric = compute_metrics(y_true=y_val, y_pred=np.zeros(len(y_val), dtype=np.int64), y_prob=None, num_classes=int(num_classes))

    epochs = max(6, min(int(cfg.epochs), 18 if cfg.dataset == "cic2024" else int(cfg.epochs)))
    unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None
    for _ in range(epochs):
        model.train()
        for xb_l, yb_l in labeled_loader:
            xb_l = xb_l.to(device)
            yb_l = yb_l.to(device)
            logits_l = model(xb_l)
            sup_loss = supervised_ce_loss(logits_l, yb_l, class_w)

            unsup_loss = torch.zeros((), device=device)
            if unlabeled_iter is not None:
                try:
                    xb_u = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    xb_u = next(unlabeled_iter)
                xb_u = xb_u.to(device)
                xb_u_w = weak_augment(xb_u, noise_std=float(cfg.weak_noise_std))
                xb_u_s = strong_augment(
                    xb_u,
                    noise_std=float(cfg.strong_noise_std),
                    feature_dropout=float(cfg.strong_feature_dropout),
                )
                logits_u_w = model(xb_u_w)
                logits_u_s = model(xb_u_s)
                unsup_loss, _, _ = fixmatch_unsup_loss(
                    logits_weak=logits_u_w,
                    logits_strong=logits_u_s,
                    threshold=float(max(cfg.pseudo_label_threshold, 0.92 if cfg.dataset == "cic2024" else cfg.pseudo_label_threshold)),
                )

            loss = sup_loss + float(getattr(cfg, "unsup_loss_weight", 1.0)) * unsup_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        scheduler.step()

        model.eval()
        prob_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X_val), max(512, labeled_bs)):
                xb = torch.from_numpy(X_val[start : start + max(512, labeled_bs)]).to(device)
                logits = model(xb)
                prob_parts.append(torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32, copy=False))
        prob_val = np.concatenate(prob_parts, axis=0).astype(np.float32, copy=False) if prob_parts else np.zeros((0, int(num_classes)), dtype=np.float32)
        y_pred = prob_val.argmax(axis=1).astype(np.int64, copy=False)
        metric = compute_metrics(y_true=y_val, y_pred=y_pred, y_prob=prob_val, num_classes=int(num_classes))
        key = (float(metric.f1), float(metric.recall), float(metric.precision), float(metric.acc))
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_prob = prob_val.copy()
            best_metric = metric

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    wrapper = _TorchTabularWrapper(
        state_dict=best_state,
        input_dim=int(X_l.shape[1]),
        num_classes=int(num_classes),
        hidden_dims=hidden_dims,
        dropout=float(dropout),
        batch_size=max(1024, labeled_bs),
    )
    return wrapper, best_prob.astype(np.float32, copy=False), best_metric


def _train_stage2_semi(
    split,
    stage_names: list[str],
    cfg: ExperimentConfig,
) -> tuple[object, dict, np.ndarray, Metrics]:
    class_boost_cfg: dict[str, float] = {}
    raw_class_boost = str(getattr(cfg, "stage2_class_boost_json", "") or "").strip()
    if raw_class_boost:
        try:
            parsed = json.loads(raw_class_boost)
            if isinstance(parsed, dict):
                class_boost_cfg = {str(k): float(v) for k, v in parsed.items()}
        except Exception:
            class_boost_cfg = {}

    def _balanced_sample_weight(y_arr: np.ndarray, num_classes: int) -> np.ndarray:
        y_i64 = y_arr.astype(np.int64, copy=False)
        counts = np.bincount(y_i64, minlength=int(num_classes)).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        w = (counts.sum() / (num_classes * counts)).astype(np.float32)
        if cfg.dataset == "cic2024":
            # Enhance weights for hard-to-distinguish minority classes in cic2024
            if getattr(cfg, "oversample_rare_classes", False):
                w = np.power(w, 1.8).astype(np.float32, copy=False)
                for i, name in enumerate(stage_names[: len(w)]):
                    if name in ["Backdoor", "Worms", "Shellcode", "Generic"]:
                        w[i] = np.float32(w[i] * 5.0)
            else:
                w = np.power(w, 1.7).astype(np.float32, copy=False)
        elif cfg.dataset == "earlycrow" and getattr(cfg, "oversample_rare_classes", False):
            w = np.power(w, 1.55).astype(np.float32, copy=False)
            rare_limit = max(128, int(getattr(cfg, "oversample_target_count", 64)))
            for i, count in enumerate(counts[: len(w)]):
                if int(count) <= rare_limit:
                    w[i] = np.float32(w[i] * 3.0)
        elif cfg.dataset == "weekdata" and getattr(cfg, "oversample_rare_classes", False):
            w = np.power(w, 1.35).astype(np.float32, copy=False)
            rare_limit = max(32, int(getattr(cfg, "oversample_target_count", 64)))
            for i, count in enumerate(counts[: len(w)]):
                if int(count) <= rare_limit:
                    w[i] = np.float32(w[i] * 2.5)
        if class_boost_cfg:
            for idx, stage_name in enumerate(stage_names[: len(w)]):
                boost = float(class_boost_cfg.get(stage_name, 1.0))
                if boost > 0.0 and boost != 1.0:
                    w[idx] = np.float32(w[idx] * boost)
        sw = w[y_i64]
        sw = sw / float(np.mean(sw))
        return sw.astype(np.float32, copy=False)

    def _norm_entropy(prob: np.ndarray) -> np.ndarray:
        p = np.asarray(prob, dtype=np.float32)
        p = np.clip(p, np.float32(1e-9), np.float32(1.0))
        ent = -(p * np.log(p)).sum(axis=1)
        denom = float(np.log(max(2, p.shape[1])))
        return (ent / max(denom, 1e-9)).astype(np.float32, copy=False)

    rng = np.random.default_rng(cfg.seed)
    labeled_mask = np.zeros(len(split.y_train), dtype=bool)
    y_train_i64 = split.y_train.astype(np.int64, copy=False)
    counts = np.bincount(y_train_i64, minlength=len(stage_names)).astype(np.int64, copy=False)
    base_labeled_ratio = float(cfg.labeled_ratio)
    full_label_max = 6000 if cfg.dataset == "cic2024" else 2500
    if cfg.dataset == "cic2024":
        base_labeled_ratio = max(base_labeled_ratio, 0.7)
        full_label_max = 3000
    elif cfg.dataset == "earlycrow":
        base_labeled_ratio = max(base_labeled_ratio, 0.8)
        full_label_max = 6000
    elif cfg.dataset == "weekdata":
        base_labeled_ratio = max(base_labeled_ratio, 0.65)
        full_label_max = 4000
    for c in range(len(stage_names)):
        idx_c = np.where(y_train_i64 == c)[0]
        if len(idx_c) == 0:
            continue
        if int(counts[c]) <= full_label_max:
            labeled_mask[idx_c] = True
            continue
        k = int(np.ceil(len(idx_c) * base_labeled_ratio))
        k = max(1, min(len(idx_c), k))
        labeled_mask[rng.choice(idx_c, size=k, replace=False)] = True

    X_l = split.X_train[labeled_mask]
    y_l = split.y_train[labeled_mask]
    X_u = split.X_train[~labeled_mask]
    row_id_l = split.row_id_train[labeled_mask].astype(np.int64, copy=False)
    row_id_u = split.row_id_train[~labeled_mask].astype(np.int64, copy=False)
    if cfg.dataset == "cic2024" and 0.0 < float(cfg.unlabeled_ratio) < 1.0 and len(X_u) > 0:
        keep_u = int(np.ceil(len(X_u) * float(cfg.unlabeled_ratio)))
        keep_u = max(1, min(len(X_u), keep_u))
        sel_u = rng.choice(len(X_u), size=keep_u, replace=False)
        sel_u = np.asarray(sel_u, dtype=np.int64)
        X_u = X_u[sel_u]
        row_id_u = row_id_u[sel_u]

    if getattr(cfg, "oversample_rare_classes", False) and len(X_l) > 0:
        y_l_i64 = y_l.astype(np.int64, copy=False)
        counts = np.bincount(y_l_i64, minlength=len(stage_names))
        target_n = int(getattr(cfg, "oversample_target_count", 50))

        X_new_parts = [X_l]
        y_new_parts = [y_l]
        row_id_new_parts = [row_id_l]
        for c in range(len(stage_names)):
            if 0 < counts[c] < target_n:
                idx_c = np.where(y_l_i64 == c)[0]
                n_needed = target_n - counts[c]
                sampled_idx = rng.choice(idx_c, size=n_needed, replace=True)
                X_dup = X_l[sampled_idx].copy()
                noise = rng.normal(loc=0.0, scale=0.01, size=X_dup.shape).astype(np.float32)
                X_dup += noise
                X_new_parts.append(X_dup)
                y_new_parts.append(y_l[sampled_idx])
                row_id_new_parts.append(row_id_l[sampled_idx])

        X_l = np.concatenate(X_new_parts, axis=0)
        y_l = np.concatenate(y_new_parts, axis=0)
        row_id_l = np.concatenate(row_id_new_parts, axis=0).astype(np.int64, copy=False)

    model: object | None = None
    w_l: np.ndarray | None = None
    semi_iters = 4 if cfg.dataset in {"cic2024", "earlycrow"} else 5
    pseudo_thr = float(cfg.pseudo_label_threshold)
    pseudo_margin = 0.0
    pseudo_entropy = 1.01
    pseudo_weight = float(getattr(cfg, "unsup_loss_weight", 1.0))
    if cfg.dataset == "cic2024":
        pseudo_thr = max(pseudo_thr, 0.94)
        pseudo_margin = 0.08
        pseudo_entropy = 0.22
        pseudo_weight = min(0.85, max(0.35, float(getattr(cfg, "unsup_loss_weight", 1.0))))
    elif cfg.dataset == "earlycrow":
        # EarlyCrow tail families are easily overwhelmed by noisy pseudo-labels.
        pseudo_thr = max(pseudo_thr, 0.97)
        pseudo_margin = 0.12
        pseudo_entropy = 0.18
        pseudo_weight = min(0.65, max(0.25, float(getattr(cfg, "unsup_loss_weight", 1.0))))
    elif cfg.dataset == "weekdata":
        pseudo_thr = max(pseudo_thr, 0.95)
        pseudo_margin = 0.08
        pseudo_entropy = 0.20
        pseudo_weight = min(0.75, max(0.30, float(getattr(cfg, "unsup_loss_weight", 1.0))))
    if cfg.dataset == "cic2024" and not _can_use_xgb_cuda():
        sw = _balanced_sample_weight(y_l, num_classes=len(stage_names))
        et = _fit_extratrees_with_early_stop(
            X_train=X_l,
            y_train=y_l,
            X_val=split.X_val,
            y_val=split.y_val,
            cfg=cfg,
        )
        hgb = _fit_hgb_multiclass_with_early_stop(
            X_train=X_l,
            y_train=y_l.astype(np.int64, copy=False),
            w_train=sw,
            X_val=split.X_val,
            y_val=split.y_val.astype(np.int64, copy=False),
            num_classes=len(stage_names),
            cfg=cfg,
        )
        prob_et = et.predict_proba(split.X_val).astype(np.float32, copy=False)
        prob_hgb = hgb.predict_proba(split.X_val).astype(np.float32, copy=False)
        best = None
        for w_et in [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]:
            prob_mix = (float(w_et) * prob_et + (1.0 - float(w_et)) * prob_hgb).astype(np.float32, copy=False)
            y_pred = prob_mix.argmax(axis=1).astype(np.int64, copy=False)
            m = compute_metrics(
                y_true=split.y_val,
                y_pred=y_pred,
                y_prob=prob_mix,
                num_classes=len(stage_names),
            )
            key = (float(m.f1), float(m.acc))
            if best is None or key > best[0]:
                best = (key, float(w_et))
        w_et = float(best[1]) if best is not None else 0.5
        if w_et <= 1e-6:
            model = hgb
        elif w_et >= 1.0 - 1e-6:
            model = et
        else:
            model = _AvgProbaEnsemble(models=[et, hgb], weights=[w_et, 1.0 - w_et])
    else:
        for _ in range(semi_iters):
            if _can_use_xgb_cuda():
                sw = _balanced_sample_weight(y_l, num_classes=len(stage_names)) if w_l is None else w_l.astype(np.float32, copy=False)
                model = _fit_xgb_multiclass(
                    X_train=X_l,
                    y_train=y_l.astype(np.int64, copy=False),
                    w_train=sw,
                    X_val=split.X_val,
                    y_val=split.y_val.astype(np.int64, copy=False),
                    num_classes=len(stage_names),
                    cfg=cfg,
                    row_id_train=row_id_l,
                    row_id_val=split.row_id_val,
                )
            else:
                model = _fit_extratrees_with_early_stop(
                    X_train=X_l,
                    y_train=y_l,
                    X_val=split.X_val,
                    y_val=split.y_val,
                    cfg=cfg,
                )
            if len(X_u) == 0:
                break
            prob_u = model.predict_proba(X_u)
            conf = prob_u.max(axis=1)
            top2 = np.sort(prob_u, axis=1)[:, -2:]
            margin = (top2[:, 1] - top2[:, 0]).astype(np.float32, copy=False)
            ent = _norm_entropy(prob_u)
            eligible = (conf >= pseudo_thr) & (margin >= np.float32(pseudo_margin)) & (ent <= np.float32(pseudo_entropy))
            take_parts: list[np.ndarray] = []
            if cfg.dataset == "cic2024":
                pred_u = prob_u.argmax(axis=1).astype(np.int64, copy=False)
                cur_counts = np.bincount(y_l.astype(np.int64, copy=False), minlength=len(stage_names)).astype(np.int64, copy=False)
                nonzero = cur_counts[cur_counts > 0]
                target = int(np.percentile(nonzero, 70)) if len(nonzero) > 0 else 0
                for cid in range(len(stage_names)):
                    idx_c = np.where(eligible & (pred_u == int(cid)))[0]
                    if len(idx_c) == 0:
                        continue
                    idx_c = idx_c[np.argsort(conf[idx_c])[::-1]]
                    deficit = max(0, target - int(cur_counts[int(cid)]))
                    quota = max(20, min(len(idx_c), max(deficit, int(np.sqrt(max(1, cur_counts[int(cid)])) * 2))))
                    take_parts.append(idx_c[:quota].astype(np.int64, copy=False))
                take = np.concatenate(take_parts, axis=0).astype(np.int64, copy=False) if take_parts else np.empty(0, dtype=np.int64)
            else:
                take = np.where(eligible)[0].astype(np.int64, copy=False)
            if len(take) == 0:
                break
            y_p = prob_u[take].argmax(axis=1).astype(np.int64)
            if _can_use_xgb_cuda():
                w_base = _balanced_sample_weight(y_p, num_classes=len(stage_names))
                w_p = (w_base * np.maximum(np.float32(0.25), conf[take].astype(np.float32, copy=False)) * np.float32(pseudo_weight)).astype(np.float32, copy=False)
                if w_l is None:
                    w_l = _balanced_sample_weight(y_l, num_classes=len(stage_names))
                w_l = np.concatenate([w_l, w_p], axis=0).astype(np.float32, copy=False)
            X_l = np.concatenate([X_l, X_u[take]], axis=0)
            y_l = np.concatenate([y_l, y_p], axis=0)
            row_id_l = np.concatenate([row_id_l, row_id_u[take]], axis=0).astype(np.int64, copy=False)
            keep_u = np.ones(len(X_u), dtype=bool)
            keep_u[take] = False
            X_u = X_u[keep_u]
            row_id_u = row_id_u[keep_u]

    if model is None:
        raise RuntimeError("Stage-2 model training failed.")

    val_prob = model.predict_proba(split.X_val).astype(np.float32, copy=False)
    if cfg.dataset == "cic2024" and len(stage_names) >= 3:
        repr_weight_train = _balanced_sample_weight(y_l.astype(np.int64, copy=False), num_classes=len(stage_names))
        if w_l is not None and len(w_l) == len(y_l):
            repr_weight_train = w_l.astype(np.float32, copy=False)
        repr_model, repr_val_prob, _ = _train_repr_mlp_semi(
            X_l=X_l,
            y_l=y_l,
            X_u=X_u,
            X_val=split.X_val,
            y_val=split.y_val,
            num_classes=len(stage_names),
            cfg=cfg,
            sample_weight=repr_weight_train,
            hidden_dims=(512, 256, 256),
            dropout=0.12,
        )
        base_val_pred = val_prob.argmax(axis=1).astype(np.int64, copy=False)
        base_val_metric = compute_metrics(
            y_true=split.y_val,
            y_pred=base_val_pred,
            y_prob=val_prob,
            num_classes=len(stage_names),
        )
        best_mix = None
        for w_repr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            mix_prob = (
                (1.0 - float(w_repr)) * val_prob + float(w_repr) * repr_val_prob
            ).astype(np.float32, copy=False)
            y_val_pred = mix_prob.argmax(axis=1).astype(np.int64, copy=False)
            met = compute_metrics(
                y_true=split.y_val,
                y_pred=y_val_pred,
                y_prob=mix_prob,
                num_classes=len(stage_names),
            )
            key = (float(met.f1), float(met.recall), float(met.precision), float(met.acc))
            if best_mix is None or key > best_mix[0]:
                best_mix = (key, float(w_repr), mix_prob, met)
        if best_mix is not None:
            _, best_w_repr, mix_prob, mix_metric = best_mix
            better_repr = (
                float(mix_metric.f1) > float(base_val_metric.f1) + 0.0005
                or (
                    float(mix_metric.f1) >= float(base_val_metric.f1) - 0.0005
                    and float(mix_metric.recall) > float(base_val_metric.recall) + 0.002
                )
            )
            if better_repr and best_w_repr > 0.0:
                model = _AvgProbaEnsemble(models=[model, repr_model], weights=[1.0 - float(best_w_repr), float(best_w_repr)])
                val_prob = mix_prob.astype(np.float32, copy=False)

    class_mult = np.ones(len(stage_names), dtype=np.float32)
    if cfg.dataset == "cic2024":
        mult_grid = [0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2, 2.8, 3.5]
        base_pred = (val_prob * class_mult).argmax(axis=1).astype(np.int64)
        base_m = compute_metrics(
            y_true=split.y_val,
            y_pred=base_pred,
            y_prob=val_prob,
            num_classes=len(stage_names),
        )
        best_key = (float(base_m.f1), float(base_m.recall), float(base_m.precision))
        for _ in range(2):
            improved = False
            for cid in range(len(stage_names)):
                cur = float(class_mult[int(cid)])
                best_local = (best_key, cur)
                for mult in mult_grid:
                    m = class_mult.copy()
                    m[int(cid)] = float(mult)
                    y_val_pred = (val_prob * m).argmax(axis=1).astype(np.int64)
                    met = compute_metrics(
                        y_true=split.y_val,
                        y_pred=y_val_pred,
                        y_prob=val_prob,
                        num_classes=len(stage_names),
                    )
                    key = (float(met.f1), float(met.recall), float(met.precision))
                    if key > best_local[0]:
                        best_local = (key, float(mult))
                if best_local[0] > best_key:
                    best_key, best_mult = best_local
                    class_mult[int(cid)] = float(best_mult)
                    improved = True
            if not improved:
                break
    if "Lateral Movement" in stage_names:
        lat_id = stage_names.index("Lateral Movement")
        best_m = None
        for mult in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0]:
            m = class_mult.copy()
            m[lat_id] = float(mult)
            y_val_pred = (val_prob * m).argmax(axis=1).astype(np.int64)
            met = compute_metrics(
                y_true=split.y_val,
                y_pred=y_val_pred,
                y_prob=val_prob,
                num_classes=len(stage_names),
            )
            key = (float(met.f1), float(met.recall)) if cfg.dataset == "cic2024" else (min(met.precision, met.recall, met.f1), met.acc)
            if best_m is None or key > best_m[0]:
                best_m = (key, float(mult))
        if best_m is not None:
            class_mult[lat_id] = float(best_m[1])

    guard_min_prob = np.zeros(len(stage_names), dtype=np.float32)
    if "Data Exfiltration" in stage_names:
        ex_id = stage_names.index("Data Exfiltration")
        if cfg.dataset == "dapt2020":
            best = None
            for mult in [1.0, 1.2, 1.5, 2.0, 3.0, 4.0]:
                m = class_mult.copy()
                m[ex_id] = float(mult)
                score = val_prob * m
                for tau in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    masked = score.copy()
                    masked[val_prob[:, ex_id] < float(tau), ex_id] = -1.0
                    y_val_pred = masked.argmax(axis=1).astype(np.int64)
                    met = compute_metrics(
                        y_true=split.y_val,
                        y_pred=y_val_pred,
                        y_prob=val_prob,
                        num_classes=len(stage_names),
                    )
                    obj = float(min(met.precision, met.recall, met.f1))
                    key = (obj, float(met.f1), float(met.recall), float(met.precision), -float(tau), float(mult))
                    if best is None or key > best[0]:
                        best = (key, float(mult), float(tau))
            if best is not None:
                _, mult, tau = best
                class_mult[ex_id] = float(mult)
                guard_min_prob[ex_id] = float(tau)
        else:
            best_g = None
            for tau in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                score = val_prob * class_mult
                masked = score.copy()
                masked[val_prob[:, ex_id] < float(tau), ex_id] = -1.0
                y_val_pred = masked.argmax(axis=1).astype(np.int64)
                met = compute_metrics(
                    y_true=split.y_val,
                    y_pred=y_val_pred,
                    y_prob=val_prob,
                    num_classes=len(stage_names),
                )
                obj = float(met.f1) if cfg.dataset == "cic2024" else float(min(met.precision, met.recall, met.f1))
                if best_g is None or obj > best_g[0]:
                    best_g = (obj, float(tau))
            if best_g is not None:
                guard_min_prob[ex_id] = float(best_g[1])

    extra: dict = {
        "class_multiplier": class_mult.astype(np.float32),
        "guard_min_prob": guard_min_prob.astype(np.float32),
    }

    extra_no_moe = extra.copy()

    if cfg.dataset == "cic2024":
        base_pred_val, _ = _predict_multiclass(model, split.X_val, extra)
        cur_pred = base_pred_val
        override_thr = np.full(len(stage_names), -1.0, dtype=np.float32)
        priority: list[int] = []
        candidates = [int(c) for c in range(len(stage_names)) if 0 < int(counts[c]) <= 8000]
        candidates.sort(key=lambda c: int(counts[c]))
        thresholds = np.linspace(0.05, 0.9, 18)
        best_m = compute_metrics(
            y_true=split.y_val,
            y_pred=cur_pred,
            y_prob=val_prob,
            num_classes=len(stage_names),
        )
        best_f1 = float(best_m.f1)
        for cid in candidates:
            pos_val = int((split.y_val.astype(np.int64, copy=False) == int(cid)).sum())
            if pos_val <= 0:
                continue
            best_local = None
            for thr in thresholds:
                mask = val_prob[:, int(cid)] >= float(thr)
                n_over = int(mask.sum())
                if n_over > max(5000, pos_val * 30):
                    continue
                y_pred = cur_pred.copy()
                y_pred[mask] = int(cid)
                m = compute_metrics(
                    y_true=split.y_val,
                    y_pred=y_pred,
                    y_prob=val_prob,
                    num_classes=len(stage_names),
                )
                key = (float(m.f1), float(m.recall))
                if best_local is None or key > best_local[0]:
                    best_local = (key, float(thr), y_pred)
            if best_local is None:
                continue
            (f1, _), thr, y_pred_best = best_local
            if float(f1) >= best_f1:
                best_f1 = float(f1)
                cur_pred = y_pred_best
                override_thr[int(cid)] = float(thr)
                priority.append(int(cid))
        if len(priority) > 0:
            extra["prob_override_thresholds"] = override_thr.astype(np.float32, copy=False)
            extra["prob_override_priority"] = np.asarray(priority, dtype=np.int64)

    ovr_candidates: list[int] = []
    if len(counts) == len(stage_names):
        for c in range(len(stage_names)):
            limit = 8000 if cfg.dataset == "cic2024" else 3000
            if 0 < int(counts[c]) <= limit:
                ovr_candidates.append(int(c))
    ovr_candidates.sort(key=lambda c: int(counts[c]) if c < len(counts) else 10**9)

    if cfg.dataset != "cic2024" and len(ovr_candidates) > 0:
        base_pred_val, _ = _predict_multiclass(model, split.X_val, extra)
        best_key = None
        best_key_m = compute_metrics(
            y_true=split.y_val,
            y_pred=base_pred_val,
            y_prob=val_prob,
            num_classes=len(stage_names),
        )
        best_key = (float(best_key_m.f1), float(best_key_m.recall)) if cfg.dataset == "cic2024" else (float(min(best_key_m.precision, best_key_m.recall, best_key_m.f1)), float(best_key_m.acc))
        cur_pred = base_pred_val

        ovr_models: list[object] = []
        ovr_thresholds: list[float] = []
        ovr_class_ids: list[int] = []
        thresholds = np.linspace(0.2, 0.99, 17)

        for cid in ovr_candidates:
            y_bin = (y_train_i64 == int(cid)).astype(np.int64)
            n_pos = int(y_bin.sum())
            if n_pos <= 0 or n_pos >= len(y_bin):
                continue
            w_pos = float((len(y_bin) - n_pos) / max(1, n_pos))
            sw = np.where(y_bin == 1, w_pos, 1.0).astype(np.float32)
            sw = sw / float(np.mean(sw))

            if _can_use_xgb_cuda():
                det = _fit_xgb_binary(
                    X_train=split.X_train,
                    y_train=y_bin.astype(np.int64, copy=False),
                    w_train=sw,
                    X_val=split.X_val,
                    y_val=(split.y_val == int(cid)).astype(np.int64, copy=False),
                    cfg=cfg,
                )
            else:
                det = HistGradientBoostingClassifier(
                    learning_rate=cfg.hgb_learning_rate,
                    max_iter=max(300, int(cfg.hgb_max_iter // 2)),
                    random_state=cfg.seed,
                )
                det.fit(split.X_train, y_bin, sample_weight=sw)

            det_prob_val = det.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
            best_local = None
            for thr in thresholds:
                y_pred = cur_pred.copy()
                y_pred[det_prob_val >= float(thr)] = int(cid)
                m = compute_metrics(
                    y_true=split.y_val,
                    y_pred=y_pred,
                    y_prob=val_prob,
                    num_classes=len(stage_names),
                )
                key = (float(m.f1), float(m.recall)) if cfg.dataset == "cic2024" else (float(min(m.precision, m.recall, m.f1)), float(m.acc))
                if best_local is None or key > best_local[0]:
                    best_local = (key, float(thr), y_pred)
            if best_local is None:
                continue
            key, thr, y_pred_best = best_local
            if key > best_key:
                best_key = key
                cur_pred = y_pred_best
                ovr_models.append(det)
                ovr_thresholds.append(float(thr))
                ovr_class_ids.append(int(cid))

        if len(ovr_models) > 0:
            extra["ovr_models"] = ovr_models
            extra["ovr_thresholds"] = np.asarray(ovr_thresholds, dtype=np.float32)
            extra["ovr_class_ids"] = np.asarray(ovr_class_ids, dtype=np.int64)

    if cfg.dataset == "cic2024":
        def _f1_bin(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
            y_true_bin = y_true_bin.astype(np.int64, copy=False)
            y_pred_bin = y_pred_bin.astype(np.int64, copy=False)
            tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
            if tp == 0:
                return 0.0
            fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
            fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
            return float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))

        ovr_all_models: list[object] = []
        ovr_all_thresholds: list[float] = []
        ovr_all_class_ids: list[int] = []
        thr_grid = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99], dtype=np.float32)

        small_candidates: list[int] = []
        for cid in range(len(stage_names)):
            y_bin = (y_train_i64 == int(cid)).astype(np.int64, copy=False)
            n_pos = int(y_bin.sum())
            if n_pos <= 0 or n_pos >= int(len(y_bin)):
                continue
            if n_pos <= 5000:
                small_candidates.append(int(cid))
        small_candidates.sort(key=lambda c: int(counts[int(c)]) if int(c) < len(counts) else 10**9)
        small_candidates = small_candidates[: min(6, len(small_candidates))]

        def _precision_recall_f1(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> tuple[float, float, float]:
            y_true_bin = y_true_bin.astype(np.int64, copy=False)
            y_pred_bin = y_pred_bin.astype(np.int64, copy=False)
            tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
            fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
            fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
            precision = float(tp / max(1, tp + fp))
            recall = float(tp / max(1, tp + fn))
            f1 = 0.0 if tp == 0 else float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))
            return precision, recall, f1

        for cid in small_candidates:
            y_bin = (y_train_i64 == int(cid)).astype(np.int64, copy=False)
            n_pos = int(y_bin.sum())
            if n_pos <= 0 or n_pos >= int(len(y_bin)):
                continue
            w_pos = float((len(y_bin) - n_pos) / max(1, n_pos))
            sw = np.where(y_bin == 1, w_pos, 1.0).astype(np.float32)
            sw = sw / float(np.mean(sw))

            if _can_use_xgb_cuda():
                det = _fit_xgb_binary(
                    X_train=split.X_train,
                    y_train=y_bin,
                    w_train=sw,
                    X_val=split.X_val,
                    y_val=(split.y_val.astype(np.int64, copy=False) == int(cid)).astype(np.int64, copy=False),
                    cfg=cfg,
                )
            else:
                det = HistGradientBoostingClassifier(
                    learning_rate=cfg.hgb_learning_rate,
                    max_iter=max(500, int(cfg.hgb_max_iter)),
                    random_state=cfg.seed,
                )
                det.fit(split.X_train, y_bin, sample_weight=sw)

            y_val_bin = (split.y_val.astype(np.int64, copy=False) == int(cid)).astype(np.int64, copy=False)
            det_prob_val = det.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
            best = None
            for thr in thr_grid.tolist():
                y_pred_bin = (det_prob_val >= float(thr)).astype(np.int64, copy=False)
                precision, recall, f1 = _precision_recall_f1(y_val_bin, y_pred_bin)
                fp = int(((y_val_bin == 0) & (y_pred_bin == 1)).sum())
                if fp > max(2000, int(n_pos * 6)):
                    continue
                key = (float(f1), float(precision), float(recall))
                if best is None or key > best[0]:
                    best = (key, float(thr))
            if best is None:
                continue
            _, thr = best
            ovr_all_models.append(det)
            ovr_all_thresholds.append(float(thr))
            ovr_all_class_ids.append(int(cid))

        if len(ovr_all_models) > 0:
            extra["ovr_all_models"] = ovr_all_models
            extra["ovr_all_thresholds"] = np.asarray(ovr_all_thresholds, dtype=np.float32)
            extra["ovr_all_class_ids"] = np.asarray(ovr_all_class_ids, dtype=np.int64)

        ovr_full_models: list[object] = []
        ovr_full_thresholds: list[float] = []
        ovr_full_class_ids: list[int] = []
        earlycrow_tail_ovr_models: list[object] = []
        earlycrow_tail_ovr_thresholds: list[float] = []
        earlycrow_tail_ovr_class_ids: list[int] = []
        earlycrow_tail_models: list[object] = []
        earlycrow_tail_thresholds: list[float] = []
        earlycrow_tail_class_ids: list[int] = []
        earlycrow_tail_rescue_names = {"onionduke1", "poisonivy1", "zebrocy1", "zebrocy2", "zebrocy3"}

        for cid in range(len(stage_names)):
            y_bin = (y_train_i64 == int(cid)).astype(np.int64, copy=False)
            n_pos = int(y_bin.sum())
            if n_pos <= 0 or n_pos >= int(len(y_bin)):
                continue
            stage_name = str(stage_names[int(cid)])
            force_tail_rescue = bool(cfg.dataset == "earlycrow" and stage_name in earlycrow_tail_rescue_names)
            pos_idx = np.where(y_bin == 1)[0].astype(np.int64)
            neg_idx = np.where(y_bin == 0)[0].astype(np.int64)

            neg_take = int(min(len(neg_idx), max(50_000, min(200_000, n_pos * 10))))
            if neg_take < len(neg_idx):
                neg_idx = rng.choice(neg_idx, size=neg_take, replace=False).astype(np.int64, copy=False)
            train_idx = np.concatenate([pos_idx, neg_idx], axis=0).astype(np.int64, copy=False)
            rng.shuffle(train_idx)

            y_train_bin = y_bin[train_idx].astype(np.int64, copy=False)
            n_pos_eff = int(y_train_bin.sum())
            n_neg_eff = int(len(y_train_bin) - n_pos_eff)
            if n_pos_eff <= 0 or n_neg_eff <= 0:
                continue
            w_pos = float(n_neg_eff / max(1, n_pos_eff))
            sw = np.where(y_train_bin == 1, w_pos, 1.0).astype(np.float32)
            sw = sw / float(np.mean(sw))

            if _can_use_xgb_cuda():
                det = _fit_xgb_binary(
                    X_train=split.X_train[train_idx],
                    y_train=y_train_bin,
                    w_train=sw,
                    X_val=split.X_val,
                    y_val=(split.y_val.astype(np.int64, copy=False) == int(cid)).astype(np.int64, copy=False),
                    cfg=cfg,
                )
            else:
                det = HistGradientBoostingClassifier(
                    learning_rate=cfg.hgb_learning_rate,
                    max_iter=max(500, int(cfg.hgb_max_iter)),
                    random_state=cfg.seed,
                )
                det.fit(split.X_train[train_idx], y_train_bin, sample_weight=sw)

            det_prob_val = det.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
            y_val_bin = (split.y_val.astype(np.int64, copy=False) == int(cid)).astype(np.int64, copy=False)
            best_thr = None
            best_key = (-1.0, -1.0, -1.0)
            for thr in thr_grid.tolist():
                y_pred_bin = (det_prob_val >= float(thr)).astype(np.int64, copy=False)
                precision, recall, f1 = _precision_recall_f1(y_val_bin, y_pred_bin)
                fp = int(((y_val_bin == 0) & (y_pred_bin == 1)).sum())
                if fp > max(5000, int(n_pos * 8)):
                    continue
                key = (float(f1), float(precision), float(recall))
                if key > best_key:
                    best_key = key
                    best_thr = float(thr)
            if best_thr is None or best_key[0] < 0.05:
                continue
            ovr_full_models.append(det)
            ovr_full_thresholds.append(float(best_thr))
            ovr_full_class_ids.append(int(cid))
            if force_tail_rescue:
                tail_thr = float(min(float(best_thr), 0.30))
                earlycrow_tail_models.append(det)
                earlycrow_tail_thresholds.append(tail_thr)
                earlycrow_tail_class_ids.append(int(cid))

        if len(ovr_full_models) > 0:
            extra["ovr_full_models"] = ovr_full_models
            extra["ovr_full_thresholds"] = np.asarray(ovr_full_thresholds, dtype=np.float32)
            extra["ovr_full_class_ids"] = np.asarray(ovr_full_class_ids, dtype=np.int64)
        if len(earlycrow_tail_models) > 0:
            extra["earlycrow_tail_ovr_models"] = earlycrow_tail_models
            extra["earlycrow_tail_ovr_thresholds"] = np.asarray(earlycrow_tail_thresholds, dtype=np.float32)
            extra["earlycrow_tail_ovr_class_ids"] = np.asarray(earlycrow_tail_class_ids, dtype=np.int64)

        # cic2024 relies on the full OVR bank later for below-gate rescue and
        # class-level refinement. Keep it in `extra` instead of dropping it
        # here, otherwise the downstream rescue path is silently disabled.
        if cfg.dataset != "cic2024":
            extra.pop("ovr_full_models", None)
            extra.pop("ovr_full_thresholds", None)
            extra.pop("ovr_full_class_ids", None)

        use_force_moe = bool(getattr(cfg, "force_moe", False))
        tri = None
        if {"DoS", "Exploits", "Reconnaissance"}.issubset(set(stage_names)):
            tri = np.asarray([stage_names.index(n) for n in ["DoS", "Exploits", "Reconnaissance"]], dtype=np.int64)
        elif use_force_moe and len(stage_names) >= 3:
            base_pred_val, _ = _predict_multiclass(model, split.X_val, extra_no_moe)
            base_pred_val = base_pred_val.astype(np.int64, copy=False)
            k = int(len(stage_names))
            cm = np.zeros((k, k), dtype=np.int64)
            yt = split.y_val.astype(np.int64, copy=False)
            yp = base_pred_val.astype(np.int64, copy=False)
            np.add.at(cm, (yt, yp), 1)
            best = None
            for a in range(k):
                for b in range(a + 1, k):
                    for c in range(b + 1, k):
                        sc = int(cm[a, b] + cm[b, a] + cm[a, c] + cm[c, a] + cm[b, c] + cm[c, b])
                        if best is None or sc > best[0]:
                            best = (sc, a, b, c)
            if best is not None:
                _, a, b, c = best
                tri = np.asarray([int(a), int(b), int(c)], dtype=np.int64)
        if tri is not None and len(tri) == 3:
            tr_mask = np.isin(y_train_i64, tri, assume_unique=False)
            va_mask = np.isin(split.y_val.astype(np.int64, copy=False), tri, assume_unique=False)
            min_tr = 500 if cfg.dataset == "cic2024" else 240
            min_va = 200 if cfg.dataset == "cic2024" else 90
            if int(tr_mask.sum()) >= int(min_tr) and int(va_mask.sum()) >= int(min_va):
                tri_map = {int(c): i for i, c in enumerate(tri.tolist())}
                y_tri_train = np.asarray([tri_map[int(v)] for v in y_train_i64[tr_mask].tolist()], dtype=np.int64)
                y_tri_val = np.asarray(
                    [tri_map[int(v)] for v in split.y_val.astype(np.int64, copy=False)[va_mask].tolist()],
                    dtype=np.int64,
                )
                sw_tri = _balanced_sample_weight(y_tri_train, num_classes=3)
                if _can_use_xgb_cuda():
                    tri_model = _fit_xgb_multiclass(
                        X_train=split.X_train[tr_mask],
                        y_train=y_tri_train,
                        w_train=sw_tri,
                        X_val=split.X_val[va_mask],
                        y_val=y_tri_val,
                        num_classes=3,
                        cfg=cfg,
                    )
                else:
                    tri_model = _fit_extratrees_with_early_stop(
                        X_train=split.X_train[tr_mask],
                        y_train=y_tri_train,
                        X_val=split.X_val[va_mask],
                        y_val=y_tri_val,
                        cfg=cfg,
                    )

                base_pred_val, _ = _predict_multiclass(model, split.X_val, extra_no_moe)
                base_pred_val = base_pred_val.astype(np.int64, copy=False)
                score_val = val_prob * class_mult
                top2 = np.argsort(score_val, axis=1)[:, -2:]
                top1 = top2[:, 1].astype(np.int64, copy=False)
                top2c = top2[:, 0].astype(np.int64, copy=False)
                idx = np.arange(len(top1), dtype=np.int64)
                diff = (score_val[idx, top1] - score_val[idx, top2c]).astype(np.float32, copy=False)
                sum_tri = val_prob[:, tri[0]] + val_prob[:, tri[1]] + val_prob[:, tri[2]]

                best = None
                for margin in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]:
                    for min_sum in ([0.50, 0.55, 0.60, 0.65, 0.70, 0.75] if use_force_moe else [0.55, 0.60, 0.65, 0.70, 0.75]):
                        route = (
                            np.isin(top1, tri, assume_unique=False)
                            & np.isin(top2c, tri, assume_unique=False)
                            & (diff <= np.float32(margin))
                            & (sum_tri >= np.float32(min_sum))
                        )
                        if not route.any():
                            continue
                        y_pred = base_pred_val.copy()
                        p3 = tri_model.predict_proba(split.X_val[route]).astype(np.float32, copy=False)
                        choose = p3.argmax(axis=1).astype(np.int64, copy=False)
                        y_pred[np.where(route)[0]] = tri[choose].astype(np.int64, copy=False)
                        met = compute_metrics(
                            y_true=split.y_val,
                            y_pred=y_pred,
                            y_prob=val_prob,
                            num_classes=len(stage_names),
                        )
                        key = (float(met.f1), float(met.recall), float(met.precision))
                        if best is None or key > best[0]:
                            best = (key, float(margin), float(min_sum))
                if best is not None:
                    _, margin, min_sum = best
                    extra["moe_triple_model"] = tri_model
                    extra["moe_triple_class_ids"] = tri.astype(np.int64, copy=False)
                    extra["moe_triple_margin"] = float(margin)
                    extra["moe_triple_min_sum"] = float(min_sum)

    if bool(getattr(cfg, "force_moe", False)) and "moe_triple_model" not in extra and len(stage_names) >= 3:
        base_pred_val, _ = _predict_multiclass(model, split.X_val, extra_no_moe)
        base_pred_val = base_pred_val.astype(np.int64, copy=False)
        k = int(len(stage_names))
        cm = np.zeros((k, k), dtype=np.int64)
        yt = split.y_val.astype(np.int64, copy=False)
        yp = base_pred_val.astype(np.int64, copy=False)
        np.add.at(cm, (yt, yp), 1)
        cnt_tr = np.bincount(y_train_i64.astype(np.int64, copy=False), minlength=k).astype(np.int64, copy=False)
        cnt_va = np.bincount(split.y_val.astype(np.int64, copy=False), minlength=k).astype(np.int64, copy=False)
        best_tri = None
        min_tr_pc = 30
        min_va_pc = 10
        for a in range(k):
            if int(cnt_tr[a]) < min_tr_pc or int(cnt_va[a]) < min_va_pc:
                continue
            for b in range(a + 1, k):
                if int(cnt_tr[b]) < min_tr_pc or int(cnt_va[b]) < min_va_pc:
                    continue
                for c in range(b + 1, k):
                    if int(cnt_tr[c]) < min_tr_pc or int(cnt_va[c]) < min_va_pc:
                        continue
                    sc = int(cm[a, b] + cm[b, a] + cm[a, c] + cm[c, a] + cm[b, c] + cm[c, b])
                    if best_tri is None or sc > best_tri[0]:
                        best_tri = (sc, a, b, c)
        if best_tri is None:
            top = np.argsort(cnt_tr)[-3:].astype(np.int64, copy=False)
            tri = top[::-1].astype(np.int64, copy=False)
        else:
            _, a, b, c = best_tri
            tri = np.asarray([int(a), int(b), int(c)], dtype=np.int64)
            tr_mask = np.isin(y_train_i64, tri, assume_unique=False)
            va_mask = np.isin(split.y_val.astype(np.int64, copy=False), tri, assume_unique=False)
            if int(tr_mask.sum()) >= 150 and int(va_mask.sum()) >= 60:
                tri_map = {int(cid): i for i, cid in enumerate(tri.tolist())}
                y_tri_train = np.asarray([tri_map[int(v)] for v in y_train_i64[tr_mask].tolist()], dtype=np.int64)
                y_tri_val = np.asarray(
                    [tri_map[int(v)] for v in split.y_val.astype(np.int64, copy=False)[va_mask].tolist()],
                    dtype=np.int64,
                )
                sw_tri = _balanced_sample_weight(y_tri_train, num_classes=3)
                if _can_use_xgb_cuda():
                    tri_model = _fit_xgb_multiclass(
                        X_train=split.X_train[tr_mask],
                        y_train=y_tri_train,
                        w_train=sw_tri,
                        X_val=split.X_val[va_mask],
                        y_val=y_tri_val,
                        num_classes=3,
                        cfg=cfg,
                    )
                else:
                    tri_model = _fit_extratrees_with_early_stop(
                        X_train=split.X_train[tr_mask],
                        y_train=y_tri_train,
                        X_val=split.X_val[va_mask],
                        y_val=y_tri_val,
                        cfg=cfg,
                    )

                score_val = val_prob * class_mult
                top2 = np.argsort(score_val, axis=1)[:, -2:]
                top1 = top2[:, 1].astype(np.int64, copy=False)
                top2c = top2[:, 0].astype(np.int64, copy=False)
                idx = np.arange(len(top1), dtype=np.int64)
                diff = (score_val[idx, top1] - score_val[idx, top2c]).astype(np.float32, copy=False)
                sum_tri = val_prob[:, tri[0]] + val_prob[:, tri[1]] + val_prob[:, tri[2]]

                best = None
                for margin in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]:
                    for min_sum in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                        route = (
                            np.isin(top1, tri, assume_unique=False)
                            & np.isin(top2c, tri, assume_unique=False)
                            & (diff <= np.float32(margin))
                            & (sum_tri >= np.float32(min_sum))
                        )
                        if not route.any():
                            continue
                        y_pred = base_pred_val.copy()
                        p3 = tri_model.predict_proba(split.X_val[route]).astype(np.float32, copy=False)
                        choose = p3.argmax(axis=1).astype(np.int64, copy=False)
                        y_pred[np.where(route)[0]] = tri[choose].astype(np.int64, copy=False)
                        met = compute_metrics(
                            y_true=split.y_val,
                            y_pred=y_pred,
                            y_prob=val_prob,
                            num_classes=len(stage_names),
                        )
                        key = (float(met.f1), float(met.recall), float(met.precision))
                        if best is None or key > best[0]:
                            best = (key, float(margin), float(min_sum))
                if best is None:
                    margin = 0.12
                    min_sum = 0.60
                else:
                    _, margin, min_sum = best
                extra["moe_triple_model"] = tri_model
                extra["moe_triple_class_ids"] = tri.astype(np.int64, copy=False)
                extra["moe_triple_margin"] = float(margin)
                extra["moe_triple_min_sum"] = float(min_sum)

    if cfg.dataset != "cic2024" and "ovr_full_models" not in extra and len(stage_names) >= 3:
        thr_grid = np.array(
            [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99],
            dtype=np.float32,
        )
        ovr_full_models: list[object] = []
        ovr_full_thresholds: list[float] = []
        ovr_full_class_ids: list[int] = []
        earlycrow_tail_ovr_models: list[object] = []
        earlycrow_tail_ovr_thresholds: list[float] = []
        earlycrow_tail_ovr_class_ids: list[int] = []
        earlycrow_tail_rescue_names = {"onionduke1", "poisonivy1", "zebrocy1", "zebrocy2", "zebrocy3"}

        def _precision_recall_f1(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> tuple[float, float, float]:
            y_true_bin = y_true_bin.astype(np.int64, copy=False)
            y_pred_bin = y_pred_bin.astype(np.int64, copy=False)
            tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
            fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
            fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
            precision = float(tp / max(1, tp + fp))
            recall = float(tp / max(1, tp + fn))
            f1 = 0.0 if tp == 0 else float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))
            return precision, recall, f1

        for cid in range(len(stage_names)):
            y_bin = (y_train_i64 == int(cid)).astype(np.int64, copy=False)
            n_pos = int(y_bin.sum())
            if n_pos <= 0 or n_pos >= int(len(y_bin)):
                continue
            stage_name = str(stage_names[int(cid)])
            force_tail_rescue = bool(cfg.dataset == "earlycrow" and stage_name in earlycrow_tail_rescue_names)
            w_pos = float((len(y_bin) - n_pos) / max(1, n_pos))
            sw = np.where(y_bin == 1, w_pos, 1.0).astype(np.float32)
            if force_tail_rescue:
                sw[y_bin == 1] = sw[y_bin == 1] * np.float32(1.75)
            sw = sw / float(np.mean(sw))

            if _can_use_xgb_cuda():
                det = _fit_xgb_binary(
                    X_train=split.X_train,
                    y_train=y_bin,
                    w_train=sw,
                    X_val=split.X_val,
                    y_val=(split.y_val.astype(np.int64, copy=False) == int(cid)).astype(np.int64, copy=False),
                    cfg=cfg,
                )
            else:
                det = HistGradientBoostingClassifier(
                    learning_rate=cfg.hgb_learning_rate,
                    max_iter=max(400, int(cfg.hgb_max_iter)),
                    random_state=cfg.seed,
                )
                det.fit(split.X_train, y_bin, sample_weight=sw)

            det_prob_val = det.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
            y_val_bin = (split.y_val.astype(np.int64, copy=False) == int(cid)).astype(np.int64, copy=False)

            best_thr = None
            best_key = (-1.0, -1.0, -1.0)
            recall_first = bool((cfg.dataset == "dapt2020" and n_pos <= 50) or force_tail_rescue)
            floor_thr = 0.0
            if cfg.dataset == "dapt2020" and n_pos <= 50:
                det_prob_train = det.predict_proba(split.X_train)[:, 1].astype(np.float32, copy=False)
                neg_train = det_prob_train[y_bin == 0]
                pos_train = det_prob_train[y_bin == 1]
                base_min = 0.6
                if str(stage_names[int(cid)]) == "Data Exfiltration":
                    base_min = 0.8
                if len(neg_train) > 0:
                    floor_thr = float(max(base_min, float(np.quantile(neg_train, 0.9995))))
                if len(pos_train) > 0 and floor_thr > 0.0:
                    floor_thr = float(min(floor_thr, float(np.quantile(pos_train, 0.05))))
            fp_cap = int(max(50, min(5000, n_pos * 25)))
            for thr in thr_grid.tolist():
                if float(thr) < float(floor_thr):
                    continue
                y_pred_bin = (det_prob_val >= float(thr)).astype(np.int64, copy=False)
                precision, recall, f1 = _precision_recall_f1(y_val_bin, y_pred_bin)
                fp = int(((y_val_bin == 0) & (y_pred_bin == 1)).sum())
                if fp > fp_cap:
                    continue
                if force_tail_rescue:
                    key = (float(recall), float(f1), float(precision), -float(thr))
                else:
                    key = (float(recall), float(f1), float(precision)) if recall_first else (float(f1), float(precision), float(recall))
                if key > best_key:
                    best_key = key
                    best_thr = float(thr)
            if best_thr is None or best_key[0] < 0.05:
                continue
            ovr_full_models.append(det)
            ovr_full_thresholds.append(float(best_thr))
            ovr_full_class_ids.append(int(cid))
            if force_tail_rescue:
                earlycrow_tail_ovr_models.append(det)
                earlycrow_tail_ovr_thresholds.append(float(min(float(best_thr), 0.30)))
                earlycrow_tail_ovr_class_ids.append(int(cid))

        if len(ovr_full_models) > 0:
            extra["ovr_full_models"] = ovr_full_models
            extra["ovr_full_thresholds"] = np.asarray(ovr_full_thresholds, dtype=np.float32)
            extra["ovr_full_class_ids"] = np.asarray(ovr_full_class_ids, dtype=np.int64)
        if len(earlycrow_tail_ovr_models) > 0:
            extra["earlycrow_tail_ovr_models"] = earlycrow_tail_ovr_models
            extra["earlycrow_tail_ovr_thresholds"] = np.asarray(earlycrow_tail_ovr_thresholds, dtype=np.float32)
            extra["earlycrow_tail_ovr_class_ids"] = np.asarray(earlycrow_tail_ovr_class_ids, dtype=np.int64)

    if cfg.dataset == "earlycrow":
        specialist_names = [
            "PlugX1",
            "Sogou",
            "Zeus",
            "onionduke1",
            "poisonivy1",
            "zebrocy1",
            "zebrocy2",
            "zebrocy3",
        ]
        if set(specialist_names).issubset(set(stage_names)):
            specialist_ids = np.asarray([int(stage_names.index(name)) for name in specialist_names], dtype=np.int64)
            specialist_head_ids = np.asarray(
                [int(stage_names.index(name)) for name in ["PlugX1", "Sogou", "Zeus"]],
                dtype=np.int64,
            )
            specialist_tail_ids = np.asarray(
                [int(cid) for cid in specialist_ids.tolist() if int(cid) not in set(specialist_head_ids.tolist())],
                dtype=np.int64,
            )
            tr_mask = np.isin(y_train_i64, specialist_ids, assume_unique=False)
            va_mask = np.isin(split.y_val.astype(np.int64, copy=False), specialist_ids, assume_unique=False)
            if int(tr_mask.sum()) >= 600 and int(va_mask.sum()) >= 120:
                specialist_map = {int(cid): i for i, cid in enumerate(specialist_ids.tolist())}
                y_spec_train = np.asarray([specialist_map[int(v)] for v in y_train_i64[tr_mask].tolist()], dtype=np.int64)
                y_spec_val = np.asarray(
                    [specialist_map[int(v)] for v in split.y_val.astype(np.int64, copy=False)[va_mask].tolist()],
                    dtype=np.int64,
                )
                spec_w = _balanced_sample_weight(y_spec_train, num_classes=len(specialist_ids)).astype(np.float32, copy=False)
                spec_w = np.power(spec_w, 1.35).astype(np.float32, copy=False)
                for local_id, global_id in enumerate(specialist_ids.tolist()):
                    cls_name = str(stage_names[int(global_id)])
                    cls_mask = y_spec_train == int(local_id)
                    if not cls_mask.any():
                        continue
                    global_count = int(counts[int(global_id)]) if int(global_id) < len(counts) else int(cls_mask.sum())
                    if cls_name in {"onionduke1", "poisonivy1"}:
                        spec_w[cls_mask] = np.float32(spec_w[cls_mask] * 3.2)
                    elif cls_name in {"zebrocy1", "zebrocy2", "zebrocy3"}:
                        spec_w[cls_mask] = np.float32(spec_w[cls_mask] * 3.8)
                    elif global_count <= 2000:
                        spec_w[cls_mask] = np.float32(spec_w[cls_mask] * 1.4)
                spec_w = spec_w / np.float32(max(1e-6, float(np.mean(spec_w))))

                if _can_use_xgb_cuda():
                    specialist_model = _fit_xgb_multiclass(
                        X_train=split.X_train[tr_mask],
                        y_train=y_spec_train,
                        w_train=spec_w,
                        X_val=split.X_val[va_mask],
                        y_val=y_spec_val,
                        num_classes=len(specialist_ids),
                        cfg=cfg,
                    )
                else:
                    specialist_model = _fit_extratrees_with_early_stop(
                        X_train=split.X_train[tr_mask],
                        y_train=y_spec_train,
                        X_val=split.X_val[va_mask],
                        y_val=y_spec_val,
                        cfg=cfg,
                    )

                def _macro_f1_recall_on_ids(cm: np.ndarray, class_ids: np.ndarray) -> tuple[float, float]:
                    if class_ids.size == 0:
                        return 0.0, 0.0
                    f1_vals: list[float] = []
                    rec_vals: list[float] = []
                    for cid in class_ids.astype(np.int64, copy=False).tolist():
                        tp = int(cm[int(cid), int(cid)])
                        fp = int(cm[:, int(cid)].sum() - tp)
                        fn = int(cm[int(cid), :].sum() - tp)
                        rec = float(tp / max(1, tp + fn))
                        f1 = 0.0 if tp == 0 else float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))
                        f1_vals.append(f1)
                        rec_vals.append(rec)
                    return float(np.mean(f1_vals)), float(np.mean(rec_vals))

                base_pred_val, _ = _predict_multiclass(model, split.X_val, extra_no_moe)
                base_pred_val = base_pred_val.astype(np.int64, copy=False)
                base_metric = compute_metrics(
                    y_true=split.y_val,
                    y_pred=base_pred_val,
                    y_prob=val_prob,
                    num_classes=len(stage_names),
                )
                base_tail_f1, base_tail_rec = _macro_f1_recall_on_ids(base_metric.cm, specialist_tail_ids)
                base_cluster_f1, _ = _macro_f1_recall_on_ids(base_metric.cm, specialist_ids)

                score_val = val_prob * class_mult
                top2 = np.argsort(score_val, axis=1)[:, -2:]
                top1 = top2[:, 1].astype(np.int64, copy=False)
                top2c = top2[:, 0].astype(np.int64, copy=False)
                idx = np.arange(len(top1), dtype=np.int64)
                diff = (score_val[idx, top1] - score_val[idx, top2c]).astype(np.float32, copy=False)
                cluster_sum = val_prob[:, specialist_ids].sum(axis=1).astype(np.float32, copy=False)
                head_top1 = np.isin(top1, specialist_head_ids, assume_unique=False)

                best = None
                base_key = (
                    float(base_tail_rec),
                    float(base_tail_f1),
                    float(base_cluster_f1),
                    float(base_metric.f1),
                    float(base_metric.recall),
                )
                for margin in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26]:
                    for min_sum in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                        route = (
                            np.isin(top1, specialist_ids, assume_unique=False)
                            & (cluster_sum >= np.float32(min_sum))
                            & (
                                head_top1
                                | np.isin(top2c, specialist_ids, assume_unique=False)
                                | (diff <= np.float32(margin))
                            )
                        )
                        if int(route.sum()) < 20:
                            continue
                        y_pred = base_pred_val.copy()
                        spec_prob = specialist_model.predict_proba(split.X_val[route]).astype(np.float32, copy=False)
                        choose = spec_prob.argmax(axis=1).astype(np.int64, copy=False)
                        y_pred[np.where(route)[0]] = specialist_ids[choose].astype(np.int64, copy=False)
                        met = compute_metrics(
                            y_true=split.y_val,
                            y_pred=y_pred,
                            y_prob=val_prob,
                            num_classes=len(stage_names),
                        )
                        tail_f1, tail_rec = _macro_f1_recall_on_ids(met.cm, specialist_tail_ids)
                        cluster_f1, _ = _macro_f1_recall_on_ids(met.cm, specialist_ids)
                        if float(met.f1) < float(base_metric.f1) - 0.01:
                            continue
                        key = (
                            float(tail_rec),
                            float(tail_f1),
                            float(cluster_f1),
                            float(met.f1),
                            float(met.recall),
                        )
                        if key > base_key and (best is None or key > best[0]):
                            best = (key, float(margin), float(min_sum))
                if best is not None:
                    _, margin, min_sum = best
                    extra["earlycrow_tail_specialist_model"] = specialist_model
                    extra["earlycrow_tail_specialist_class_ids"] = specialist_ids.astype(np.int64, copy=False)
                    extra["earlycrow_tail_specialist_head_ids"] = specialist_head_ids.astype(np.int64, copy=False)
                    extra["earlycrow_tail_specialist_margin"] = float(margin)
                    extra["earlycrow_tail_specialist_min_sum"] = float(min_sum)

    if cfg.dataset == "cic2024":
        specialist_names = [
            "Analysis",
            "Backdoor",
            "DoS",
            "Exploits",
            "Reconnaissance",
            "Shellcode",
            "Worms",
        ]
        if set(specialist_names).issubset(set(stage_names)):
            specialist_ids = np.asarray([int(stage_names.index(name)) for name in specialist_names], dtype=np.int64)
            specialist_head_ids = np.asarray(
                [int(stage_names.index(name)) for name in ["DoS", "Exploits", "Reconnaissance"]],
                dtype=np.int64,
            )
            specialist_tail_ids = np.asarray(
                [int(cid) for cid in specialist_ids.tolist() if int(cid) not in set(specialist_head_ids.tolist())],
                dtype=np.int64,
            )
            tr_mask = np.isin(y_train_i64, specialist_ids, assume_unique=False)
            va_mask = np.isin(split.y_val.astype(np.int64, copy=False), specialist_ids, assume_unique=False)
            if int(tr_mask.sum()) >= 3000 and int(va_mask.sum()) >= 600:
                specialist_map = {int(cid): i for i, cid in enumerate(specialist_ids.tolist())}
                y_spec_train = np.asarray([specialist_map[int(v)] for v in y_train_i64[tr_mask].tolist()], dtype=np.int64)
                y_spec_val = np.asarray(
                    [specialist_map[int(v)] for v in split.y_val.astype(np.int64, copy=False)[va_mask].tolist()],
                    dtype=np.int64,
                )
                spec_w = _balanced_sample_weight(y_spec_train, num_classes=len(specialist_ids)).astype(np.float32, copy=False)
                spec_w = np.power(spec_w, 1.30).astype(np.float32, copy=False)
                for local_id, global_id in enumerate(specialist_ids.tolist()):
                    cls_name = str(stage_names[int(global_id)])
                    cls_mask = y_spec_train == int(local_id)
                    if not cls_mask.any():
                        continue
                    if cls_name == "Analysis":
                        spec_w[cls_mask] = np.float32(spec_w[cls_mask] * 4.0)
                    elif cls_name in {"Backdoor", "Shellcode", "Worms"}:
                        spec_w[cls_mask] = np.float32(spec_w[cls_mask] * 2.8)
                    elif cls_name in {"DoS", "Reconnaissance"}:
                        spec_w[cls_mask] = np.float32(spec_w[cls_mask] * 1.35)
                spec_w = spec_w / np.float32(max(1e-6, float(np.mean(spec_w))))

                if _can_use_xgb_cuda():
                    specialist_model = _fit_xgb_multiclass(
                        X_train=split.X_train[tr_mask],
                        y_train=y_spec_train,
                        w_train=spec_w,
                        X_val=split.X_val[va_mask],
                        y_val=y_spec_val,
                        num_classes=len(specialist_ids),
                        cfg=cfg,
                    )
                else:
                    specialist_model = _fit_extratrees_with_early_stop(
                        X_train=split.X_train[tr_mask],
                        y_train=y_spec_train,
                        X_val=split.X_val[va_mask],
                        y_val=y_spec_val,
                        cfg=cfg,
                    )

                def _macro_f1_recall_on_ids(cm: np.ndarray, class_ids: np.ndarray) -> tuple[float, float]:
                    if class_ids.size == 0:
                        return 0.0, 0.0
                    f1_vals: list[float] = []
                    rec_vals: list[float] = []
                    for cid in class_ids.astype(np.int64, copy=False).tolist():
                        tp = int(cm[int(cid), int(cid)])
                        fp = int(cm[:, int(cid)].sum() - tp)
                        fn = int(cm[int(cid), :].sum() - tp)
                        rec = float(tp / max(1, tp + fn))
                        f1 = 0.0 if tp == 0 else float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))
                        f1_vals.append(f1)
                        rec_vals.append(rec)
                    return float(np.mean(f1_vals)), float(np.mean(rec_vals))

                base_pred_val, _ = _predict_multiclass(model, split.X_val, extra)
                base_pred_val = base_pred_val.astype(np.int64, copy=False)
                base_metric = compute_metrics(
                    y_true=split.y_val,
                    y_pred=base_pred_val,
                    y_prob=val_prob,
                    num_classes=len(stage_names),
                )
                base_tail_f1, base_tail_rec = _macro_f1_recall_on_ids(base_metric.cm, specialist_tail_ids)
                base_cluster_f1, base_cluster_rec = _macro_f1_recall_on_ids(base_metric.cm, specialist_ids)

                score_val = val_prob * class_mult
                top2 = np.argsort(score_val, axis=1)[:, -2:]
                top1 = top2[:, 1].astype(np.int64, copy=False)
                top2c = top2[:, 0].astype(np.int64, copy=False)
                idx = np.arange(len(top1), dtype=np.int64)
                diff = (score_val[idx, top1] - score_val[idx, top2c]).astype(np.float32, copy=False)
                cluster_sum = val_prob[:, specialist_ids].sum(axis=1).astype(np.float32, copy=False)
                head_top1 = np.isin(top1, specialist_head_ids, assume_unique=False)
                near_cluster = np.isin(top1, specialist_ids, assume_unique=False) | np.isin(top2c, specialist_ids, assume_unique=False)

                best = None
                base_key = (
                    float(base_cluster_f1),
                    float(base_cluster_rec),
                    float(base_tail_rec),
                    float(base_tail_f1),
                    float(base_metric.f1),
                    float(base_metric.recall),
                )
                for margin in [0.10, 0.15, 0.20, 0.28, 0.36, 0.45]:
                    for min_sum in [0.45, 0.55, 0.65, 0.75, 0.85]:
                        route = (
                            near_cluster
                            & (cluster_sum >= np.float32(min_sum))
                            & (
                                head_top1
                                | np.isin(top1, specialist_ids, assume_unique=False)
                                | (diff <= np.float32(margin))
                            )
                        )
                        if int(route.sum()) < 120:
                            continue
                        y_pred = base_pred_val.copy()
                        spec_prob = specialist_model.predict_proba(split.X_val[route]).astype(np.float32, copy=False)
                        choose = spec_prob.argmax(axis=1).astype(np.int64, copy=False)
                        y_pred[np.where(route)[0]] = specialist_ids[choose].astype(np.int64, copy=False)
                        met = compute_metrics(
                            y_true=split.y_val,
                            y_pred=y_pred,
                            y_prob=val_prob,
                            num_classes=len(stage_names),
                        )
                        tail_f1, tail_rec = _macro_f1_recall_on_ids(met.cm, specialist_tail_ids)
                        cluster_f1, cluster_rec = _macro_f1_recall_on_ids(met.cm, specialist_ids)
                        if float(met.f1) < float(base_metric.f1) - 0.01:
                            continue
                        key = (
                            float(cluster_f1),
                            float(cluster_rec),
                            float(tail_rec),
                            float(tail_f1),
                            float(met.f1),
                            float(met.recall),
                        )
                        if key > base_key and (best is None or key > best[0]):
                            best = (key, float(margin), float(min_sum))
                if best is not None:
                    _, margin, min_sum = best
                    extra["cic2024_hard_specialist_model"] = specialist_model
                    extra["cic2024_hard_specialist_class_ids"] = specialist_ids.astype(np.int64, copy=False)
                    extra["cic2024_hard_specialist_head_ids"] = specialist_head_ids.astype(np.int64, copy=False)
                    extra["cic2024_hard_specialist_margin"] = float(margin)
                    extra["cic2024_hard_specialist_min_sum"] = float(min_sum)

    if cfg.dataset == "cic2024":
        head_names = ["DoS", "Exploits", "Reconnaissance"]
        if set(head_names).issubset(set(stage_names)):
            head_ids = np.asarray([int(stage_names.index(name)) for name in head_names], dtype=np.int64)
            tr_mask = np.isin(y_train_i64, head_ids, assume_unique=False)
            va_mask = np.isin(split.y_val.astype(np.int64, copy=False), head_ids, assume_unique=False)
            if int(tr_mask.sum()) >= 1800 and int(va_mask.sum()) >= 300:
                head_map = {int(cid): i for i, cid in enumerate(head_ids.tolist())}
                y_head_train = np.asarray([head_map[int(v)] for v in y_train_i64[tr_mask].tolist()], dtype=np.int64)
                y_head_val = np.asarray(
                    [head_map[int(v)] for v in split.y_val.astype(np.int64, copy=False)[va_mask].tolist()],
                    dtype=np.int64,
                )
                head_w = _balanced_sample_weight(y_head_train, num_classes=len(head_ids)).astype(np.float32, copy=False)
                for local_id, global_id in enumerate(head_ids.tolist()):
                    cls_name = str(stage_names[int(global_id)])
                    cls_mask = y_head_train == int(local_id)
                    if not cls_mask.any():
                        continue
                    if cls_name == "DoS":
                        head_w[cls_mask] = np.float32(head_w[cls_mask] * 2.8)
                    elif cls_name == "Reconnaissance":
                        head_w[cls_mask] = np.float32(head_w[cls_mask] * 2.3)
                    elif cls_name == "Exploits":
                        head_w[cls_mask] = np.float32(head_w[cls_mask] * 0.72)
                head_w = head_w / np.float32(max(1e-6, float(np.mean(head_w))))
                head_model = _fit_xgb_multiclass(
                    X_train=split.X_train[tr_mask],
                    y_train=y_head_train,
                    w_train=head_w,
                    X_val=split.X_val[va_mask],
                    y_val=y_head_val,
                    num_classes=len(head_ids),
                    cfg=cfg,
                )

                y_router_train = np.isin(y_train_i64, head_ids, assume_unique=False).astype(np.int64, copy=False)
                y_router_val = np.isin(split.y_val.astype(np.int64, copy=False), head_ids, assume_unique=False).astype(np.int64, copy=False)
                router_w = _balanced_sample_weight(y_router_train, num_classes=2).astype(np.float32, copy=False)
                router_w[y_router_train == 1] = np.float32(router_w[y_router_train == 1] * 1.8)
                router_w = router_w / np.float32(max(1e-6, float(np.mean(router_w))))
                head_router_model = _fit_xgb_binary(
                    X_train=split.X_train,
                    y_train=y_router_train,
                    w_train=router_w,
                    X_val=split.X_val,
                    y_val=y_router_val,
                    cfg=cfg,
                    row_id_train=split.row_id_train,
                    row_id_val=split.row_id_val,
                )

                def _head_absorb_from_pred(y_pred_local: np.ndarray) -> int:
                    dos_id = int(stage_names.index("DoS"))
                    exp_id = int(stage_names.index("Exploits"))
                    rec_id = int(stage_names.index("Reconnaissance"))
                    y_true_local = split.y_val.astype(np.int64, copy=False)
                    y_pred_local = np.asarray(y_pred_local, dtype=np.int64)
                    return int(
                        ((y_true_local == dos_id) & (y_pred_local == exp_id)).sum()
                        + ((y_true_local == rec_id) & (y_pred_local == exp_id)).sum()
                        + ((y_true_local == exp_id) & (y_pred_local == dos_id)).sum()
                        + ((y_true_local == exp_id) & (y_pred_local == rec_id)).sum()
                    )

                def _macro_f1_recall_head(cm: np.ndarray) -> tuple[float, float]:
                    vals_f1: list[float] = []
                    vals_rec: list[float] = []
                    for cid in head_ids.tolist():
                        tp = int(cm[int(cid), int(cid)])
                        fp = int(cm[:, int(cid)].sum() - tp)
                        fn = int(cm[int(cid), :].sum() - tp)
                        rec = float(tp / max(1, tp + fn))
                        f1 = 0.0 if tp == 0 else float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))
                        vals_f1.append(f1)
                        vals_rec.append(rec)
                    return float(np.mean(vals_f1)), float(np.mean(vals_rec))

                base_pred_val, _ = _predict_multiclass(model, split.X_val, extra)
                base_pred_val = base_pred_val.astype(np.int64, copy=False)
                base_metric = compute_metrics(
                    y_true=split.y_val,
                    y_pred=base_pred_val,
                    y_prob=val_prob,
                    num_classes=len(stage_names),
                )
                base_head_absorb = _head_absorb_from_pred(base_pred_val)
                base_head_f1, base_head_rec = _macro_f1_recall_head(base_metric.cm)
                router_prob_val = head_router_model.predict_proba(split.X_val)[:, 1].astype(np.float32, copy=False)
                score_val = val_prob * class_mult
                top2 = np.argsort(score_val, axis=1)[:, -2:]
                top1 = top2[:, 1].astype(np.int64, copy=False)
                top2c = top2[:, 0].astype(np.int64, copy=False)
                idx = np.arange(len(top1), dtype=np.int64)
                diff = (score_val[idx, top1] - score_val[idx, top2c]).astype(np.float32, copy=False)
                sum_head = val_prob[:, head_ids].sum(axis=1).astype(np.float32, copy=False)
                near_head = np.isin(top1, head_ids, assume_unique=False) | np.isin(top2c, head_ids, assume_unique=False)

                best = None
                fallback_best = None
                base_key = (
                    -int(base_head_absorb),
                    float(base_head_f1),
                    float(base_head_rec),
                    float(base_metric.f1),
                    float(base_metric.recall),
                )
                for hi in [0.45, 0.55, 0.65, 0.75]:
                    for lo in [0.15, 0.25, 0.35, 0.45]:
                        if float(lo) > float(hi):
                            continue
                        for sum_thr in [0.25, 0.35, 0.45, 0.55, 0.65]:
                            for margin in [0.0, 0.05, 0.10, 0.18, 0.28]:
                                route = (router_prob_val >= np.float32(float(hi))) | (
                                    (router_prob_val >= np.float32(float(lo)))
                                    & (sum_head >= np.float32(float(sum_thr)))
                                    & (near_head | (diff <= np.float32(float(margin))))
                                )
                                if int(route.sum()) < 180:
                                    continue
                                y_pred = base_pred_val.copy()
                                head_prob = head_model.predict_proba(split.X_val[route]).astype(np.float32, copy=False)
                                choose = head_prob.argmax(axis=1).astype(np.int64, copy=False)
                                y_pred[np.where(route)[0]] = head_ids[choose].astype(np.int64, copy=False)
                                met = compute_metrics(
                                    y_true=split.y_val,
                                    y_pred=y_pred,
                                    y_prob=val_prob,
                                    num_classes=len(stage_names),
                                )
                                head_absorb = _head_absorb_from_pred(y_pred)
                                head_f1, head_rec = _macro_f1_recall_head(met.cm)
                                if float(met.f1) < float(base_metric.f1) - 0.010:
                                    continue
                                key = (
                                    -int(head_absorb),
                                    float(head_f1),
                                    float(head_rec),
                                    float(met.f1),
                                    float(met.recall),
                                    -float(hi),
                                    -float(sum_thr),
                                )
                                if key > base_key and (best is None or key > best[0]):
                                    best = (key, float(hi), float(lo), float(sum_thr), float(margin))
                                fallback_key = (
                                    -int(head_absorb),
                                    float(head_f1),
                                    float(head_rec),
                                    float(met.f1),
                                    float(met.recall),
                                    -float(hi),
                                    -float(sum_thr),
                                )
                                if (
                                    int(head_absorb) <= int(base_head_absorb) - 20
                                    and float(met.f1) >= float(base_metric.f1) - 0.030
                                    and (fallback_best is None or fallback_key > fallback_best[0])
                                ):
                                    fallback_best = (
                                        fallback_key,
                                        float(hi),
                                        float(lo),
                                        float(sum_thr),
                                        float(margin),
                                    )
                chosen_head = best if best is not None else fallback_best
                if chosen_head is not None:
                    _, hi, lo, sum_thr, margin = chosen_head
                    extra["cic2024_head_cluster_model"] = head_model
                    extra["cic2024_head_router_model"] = head_router_model
                    extra["cic2024_head_cluster_class_ids"] = head_ids.astype(np.int64, copy=False)
                    extra["cic2024_head_router_hi"] = float(hi)
                    extra["cic2024_head_router_lo"] = float(lo)
                    extra["cic2024_head_cluster_sum_threshold"] = float(sum_thr)
                    extra["cic2024_head_cluster_margin"] = float(margin)

    if "ovr_full_models" in extra and cfg.dataset not in ("dapt2020", "cic2024"):
        base_pred_val, _ = _predict_multiclass(model, split.X_val, extra)
        base_m = compute_metrics(
            y_true=split.y_val,
            y_pred=base_pred_val,
            y_prob=val_prob,
            num_classes=len(stage_names),
        )
        extra_try = extra.copy()
        extra_try["ovr_full_always"] = True
        y_val_pred, _ = _predict_multiclass(model, split.X_val, extra_try)
        m_try = compute_metrics(
            y_true=split.y_val,
            y_pred=y_val_pred,
            y_prob=val_prob,
            num_classes=len(stage_names),
        )
        if float(m_try.f1) > float(base_m.f1):
            extra["ovr_full_always"] = True

    if "Lateral Movement" in stage_names:
        lat_id = stage_names.index("Lateral Movement")
        y_lat = (split.y_train == lat_id).astype(np.int64)
        n_pos = int(y_lat.sum())
        if n_pos > 0 and n_pos < len(y_lat):
            w_pos = float((len(y_lat) - n_pos) / max(1, n_pos))
            sw = np.where(y_lat == 1, w_pos, 1.0).astype(np.float32)
            if _can_use_xgb_cuda():
                lat_model = _fit_xgb_binary(
                    X_train=split.X_train,
                    y_train=y_lat.astype(np.int64, copy=False),
                    w_train=sw,
                    X_val=split.X_val,
                    y_val=(split.y_val == lat_id).astype(np.int64, copy=False),
                    cfg=cfg,
                )
            else:
                lat_model = HistGradientBoostingClassifier(
                    learning_rate=cfg.hgb_learning_rate,
                    max_iter=max(200, int(cfg.hgb_max_iter // 2)),
                    random_state=cfg.seed,
                )
                lat_model.fit(split.X_train, y_lat, sample_weight=sw)
            lat_prob_val = lat_model.predict_proba(split.X_val)[:, 1].astype(np.float32)
            base_score = val_prob * class_mult
            if "Data Exfiltration" in stage_names:
                ex_id = stage_names.index("Data Exfiltration")
                if guard_min_prob[ex_id] > 0:
                    base_score = base_score.copy()
                    base_score[val_prob[:, ex_id] < guard_min_prob[ex_id], ex_id] = -1.0
            base_pred = base_score.argmax(axis=1).astype(np.int64)
            best_thr = None
            best_obj = -1.0
            for thr in np.linspace(0.2, 0.99, 17):
                y_val_pred = base_pred.copy()
                y_val_pred[lat_prob_val >= float(thr)] = lat_id
                met = compute_metrics(
                    y_true=split.y_val,
                    y_pred=y_val_pred,
                    y_prob=val_prob,
                    num_classes=len(stage_names),
                )
                obj = float(met.f1) if cfg.dataset == "cic2024" else float(min(met.precision, met.recall, met.f1))
                if obj > best_obj:
                    best_obj = obj
                    best_thr = float(thr)
            if best_thr is not None:
                extra["lateral_model"] = lat_model
                extra["lateral_threshold"] = float(best_thr)
                extra["lateral_id"] = int(lat_id)

    if "Data Exfiltration" in stage_names:
        ex_id = stage_names.index("Data Exfiltration")
        y_ex = (split.y_train == ex_id).astype(np.int64)
        n_pos = int(y_ex.sum())
        if n_pos > 0 and n_pos < len(y_ex):
            w_pos = float((len(y_ex) - n_pos) / max(1, n_pos))
            sw = np.where(y_ex == 1, w_pos, 1.0).astype(np.float32)
            if _can_use_xgb_cuda():
                ex_model = _fit_xgb_binary(
                    X_train=split.X_train,
                    y_train=y_ex.astype(np.int64, copy=False),
                    w_train=sw,
                    X_val=split.X_val,
                    y_val=(split.y_val == ex_id).astype(np.int64, copy=False),
                    cfg=cfg,
                )
            else:
                ex_model = HistGradientBoostingClassifier(
                    learning_rate=cfg.hgb_learning_rate,
                    max_iter=max(200, int(cfg.hgb_max_iter // 2)),
                    random_state=cfg.seed,
                )
                ex_model.fit(split.X_train, y_ex, sample_weight=sw)
            ex_prob_val = ex_model.predict_proba(split.X_val)[:, 1].astype(np.float32)
            pos_mask = split.y_val == ex_id
            if pos_mask.any():
                min_pos = float(ex_prob_val[pos_mask].min())
                neg = ex_prob_val[~pos_mask]
                thr_hi = float(min_pos)
                if len(neg) > 0:
                    q = float(np.quantile(neg, 0.9995)) if cfg.dataset == "dapt2020" else float(np.quantile(neg, 0.999))
                    base = 0.93 if cfg.dataset == "dapt2020" else 0.9
                    thr_hi = float(min(min_pos, max(q, base)))

                if cfg.dataset == "dapt2020":
                    prob_train = model.predict_proba(split.X_train).astype(np.float32, copy=False)[:, ex_id]
                    prob_val = val_prob[:, ex_id].astype(np.float32, copy=False)
                    ex_prob_train = ex_model.predict_proba(split.X_train)[:, 1].astype(np.float32, copy=False)
                    y_ex_train = (split.y_train.astype(np.int64, copy=False) == int(ex_id)).astype(np.int64, copy=False)
                    y_ex_val = (split.y_val.astype(np.int64, copy=False) == int(ex_id)).astype(np.int64, copy=False)
                    prob_all = np.concatenate([prob_train, prob_val], axis=0)
                    ex_prob_all = np.concatenate([ex_prob_train, ex_prob_val.astype(np.float32, copy=False)], axis=0)
                    y_all = np.concatenate([y_ex_train, y_ex_val], axis=0)

                    best = None
                    for req in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
                        req = float(req)
                        pos = y_all == 1
                        if not pos.any():
                            continue
                        if float(prob_all[pos].min()) < req:
                            continue
                        neg = (y_all == 0) & (prob_all >= np.float32(req))
                        max_neg = float(ex_prob_all[neg].max()) if neg.any() else -1.0
                        min_pos_all = float(ex_prob_all[pos].min())
                        if min_pos_all <= max_neg:
                            continue
                        thr_lo = float(max(0.0, max_neg + 1e-6))

                        extra_try = extra.copy()
                        extra_try["dataex_model"] = ex_model
                        extra_try["dataex_threshold_hi"] = float(thr_hi)
                        extra_try["dataex_threshold_lo"] = float(thr_lo)
                        extra_try["dataex_guard_min_prob"] = float(0.0)
                        extra_try["dataex_require_base_prob"] = float(req)
                        extra_try["dataex_id"] = int(ex_id)
                        y_val_pred, _ = _predict_multiclass(model, split.X_val, extra_try)
                        fp = int(((~pos_mask) & (y_val_pred == int(ex_id))).sum())
                        if fp != 0:
                            continue
                        if not np.all(y_val_pred[pos_mask] == int(ex_id)):
                            continue
                        met = compute_metrics(
                            y_true=split.y_val,
                            y_pred=y_val_pred,
                            y_prob=val_prob,
                            num_classes=len(stage_names),
                        )
                        key = (float(min(met.precision, met.recall, met.f1)), float(met.f1), -float(req))
                        if best is None or key > best[0]:
                            best = (key, float(thr_lo), float(req))

                    if best is not None:
                        _, thr_lo, req = best
                        extra["dataex_model"] = ex_model
                        extra["dataex_threshold_hi"] = float(thr_hi)
                        extra["dataex_threshold_lo"] = float(thr_lo)
                        extra["dataex_guard_min_prob"] = float(0.0)
                        extra["dataex_require_base_prob"] = float(req)
                        extra["dataex_id"] = int(ex_id)
                    else:
                        extra["dataex_model"] = ex_model
                        extra["dataex_threshold_hi"] = float(thr_hi)
                        extra["dataex_threshold_lo"] = float(thr_hi)
                        extra["dataex_guard_min_prob"] = float(0.0)
                        extra["dataex_require_base_prob"] = float(0.0)
                        extra["dataex_id"] = int(ex_id)
                else:
                    extra["dataex_model"] = ex_model
                    extra["dataex_threshold_hi"] = float(thr_hi)
                    extra["dataex_threshold_lo"] = float(thr_hi)
                    extra["dataex_guard_min_prob"] = float(0.0)
                    extra["dataex_id"] = int(ex_id)

    ovo_models: list[object] = []
    ovo_pairs: list[tuple[int, int]] = []
    if cfg.dataset == "cic2024" or bool(getattr(cfg, "force_moe", False)):
        base_pred_val, _ = _predict_multiclass(model, split.X_val, extra)
        k = int(len(stage_names))
        cm = np.zeros((k, k), dtype=np.int64)
        yt = split.y_val.astype(np.int64, copy=False)
        yp = base_pred_val.astype(np.int64, copy=False)
        np.add.at(cm, (yt, yp), 1)
        pairs: list[tuple[int, int, int]] = []
        min_pair_sc = 80 if cfg.dataset == "cic2024" else 10
        for a in range(k):
            for b in range(a + 1, k):
                sc = int(cm[a, b] + cm[b, a])
                if sc >= int(min_pair_sc):
                    pairs.append((sc, a, b))
        pairs.sort(reverse=True)
        top_pairs = [(a, b) for _, a, b in pairs[: min(10, len(pairs))]]
        for a, b in top_pairs:
            a = int(a)
            b = int(b)
            tr_mask = (split.y_train == a) | (split.y_train == b)
            va_mask = (split.y_val == a) | (split.y_val == b)
            min_tr = 500 if cfg.dataset == "cic2024" else 180
            min_va = 120 if cfg.dataset == "cic2024" else 60
            if int(tr_mask.sum()) < int(min_tr) or int(va_mask.sum()) < int(min_va):
                continue
            y_tr_bin = (split.y_train[tr_mask].astype(np.int64, copy=False) == b).astype(np.int64, copy=False)
            n_pos = int(y_tr_bin.sum())
            if n_pos <= 0 or n_pos >= int(len(y_tr_bin)):
                continue
            w_pos = float((len(y_tr_bin) - n_pos) / max(1, n_pos))
            sw = np.where(y_tr_bin == 1, w_pos, 1.0).astype(np.float32)
            sw = sw / float(np.mean(sw))
            if _can_use_xgb_cuda():
                m = _fit_xgb_binary(
                    X_train=split.X_train[tr_mask],
                    y_train=y_tr_bin,
                    w_train=sw,
                    X_val=split.X_val[va_mask],
                    y_val=(split.y_val[va_mask].astype(np.int64, copy=False) == b).astype(np.int64, copy=False),
                    cfg=cfg,
                )
            else:
                m = HistGradientBoostingClassifier(
                    learning_rate=cfg.hgb_learning_rate,
                    max_iter=max(400, int(cfg.hgb_max_iter)),
                    random_state=cfg.seed,
                )
                m.fit(split.X_train[tr_mask], y_tr_bin, sample_weight=sw)
            ovo_models.append(m)
            ovo_pairs.append((a, b))
        if len(ovo_models) > 0:
            extra["ovo_models"] = ovo_models
            extra["ovo_pairs"] = np.asarray(ovo_pairs, dtype=np.int64)
            extra["ovo_margin"] = float(0.45)

        if "DoS" in stage_names and "Exploits" in stage_names:
            dos_id = int(stage_names.index("DoS"))
            exp_id = int(stage_names.index("Exploits"))
            tr_mask = (split.y_train == dos_id) | (split.y_train == exp_id)
            va_mask = (split.y_val == dos_id) | (split.y_val == exp_id)
            if int(tr_mask.sum()) >= 500 and int(va_mask.sum()) >= 120:
                y_tr_bin = (split.y_train[tr_mask].astype(np.int64, copy=False) == exp_id).astype(np.int64, copy=False)
                n_pos = int(y_tr_bin.sum())
                if 0 < n_pos < int(len(y_tr_bin)):
                    w_pos = float((len(y_tr_bin) - n_pos) / max(1, n_pos))
                    sw = np.where(y_tr_bin == 1, w_pos, 1.0).astype(np.float32)
                    sw = sw / float(np.mean(sw))
                    if _can_use_xgb_cuda():
                        det = _fit_xgb_binary(
                            X_train=split.X_train[tr_mask],
                            y_train=y_tr_bin,
                            w_train=sw,
                            X_val=split.X_val[va_mask],
                            y_val=(split.y_val[va_mask].astype(np.int64, copy=False) == exp_id).astype(np.int64, copy=False),
                            cfg=cfg,
                        )
                    else:
                        det = HistGradientBoostingClassifier(
                            learning_rate=cfg.hgb_learning_rate,
                            max_iter=max(500, int(cfg.hgb_max_iter)),
                            random_state=cfg.seed,
                        )
                        det.fit(split.X_train[tr_mask], y_tr_bin, sample_weight=sw)

                    best = None
                    for mode in ["top2", "top1"]:
                        for margin in [0.25, 0.35, 0.5, 0.7, 1.1]:
                            for min_sum in [0.0, 0.6, 0.75, 0.85]:
                                extra_try = extra.copy()
                                extra_try["dos_ex_model"] = det
                                extra_try["dos_ex_a"] = dos_id
                                extra_try["dos_ex_b"] = exp_id
                                extra_try["dos_ex_mode"] = mode
                                extra_try["dos_ex_margin"] = float(margin)
                                extra_try["dos_ex_min_sum"] = float(min_sum)
                                y_val_pred, _ = _predict_multiclass(model, split.X_val, extra_try)
                                met = compute_metrics(
                                    y_true=split.y_val,
                                    y_pred=y_val_pred,
                                    y_prob=val_prob,
                                    num_classes=len(stage_names),
                                )
                                key = (float(met.f1), float(met.recall), float(met.precision), -float(min_sum), float(margin))
                                if best is None or key > best[0]:
                                    best = (key, mode, float(margin), float(min_sum))
                    if best is not None:
                        _, mode, margin, min_sum = best
                        extra["dos_ex_model"] = det
                        extra["dos_ex_a"] = dos_id
                        extra["dos_ex_b"] = exp_id
                        extra["dos_ex_mode"] = mode
                        extra["dos_ex_margin"] = float(margin)
                        extra["dos_ex_min_sum"] = float(min_sum)

        def _cic2024_stage2_head_absorb(y_pred_local: np.ndarray) -> int:
            if cfg.dataset != "cic2024":
                return 0
            needed = {"DoS", "Exploits", "Reconnaissance"}
            if not needed.issubset(set(stage_names)):
                return 0
            dos_id = int(stage_names.index("DoS"))
            exp_id = int(stage_names.index("Exploits"))
            rec_id = int(stage_names.index("Reconnaissance"))
            y_true_local = split.y_val.astype(np.int64, copy=False)
            y_pred_local = np.asarray(y_pred_local, dtype=np.int64)
            return int(
                ((y_true_local == dos_id) & (y_pred_local == exp_id)).sum()
                + ((y_true_local == rec_id) & (y_pred_local == exp_id)).sum()
                + ((y_true_local == exp_id) & (y_pred_local == dos_id)).sum()
                + ((y_true_local == exp_id) & (y_pred_local == rec_id)).sum()
            )

        def _is_cic2024_head_cluster(cids_local: list[int] | np.ndarray) -> bool:
            if cfg.dataset != "cic2024":
                return False
            names = {stage_names[int(cid)] for cid in np.asarray(cids_local, dtype=np.int64).tolist()}
            return names == {"DoS", "Exploits", "Reconnaissance"}

        def _boost_cic2024_head_group_weights(
            cids_local: list[int] | np.ndarray,
            y_local: np.ndarray,
            sw_local: np.ndarray,
        ) -> np.ndarray:
            if not _is_cic2024_head_cluster(cids_local):
                return sw_local
            cids_arr = np.asarray(cids_local, dtype=np.int64)
            y_out = np.asarray(y_local, dtype=np.int64)
            sw_out = sw_local.astype(np.float32, copy=True)
            for lid, cid in enumerate(cids_arr.tolist()):
                cls_name = stage_names[int(cid)]
                if cls_name == "DoS":
                    sw_out[y_out == int(lid)] *= np.float32(2.6)
                elif cls_name == "Reconnaissance":
                    sw_out[y_out == int(lid)] *= np.float32(2.2)
                elif cls_name == "Exploits":
                    sw_out[y_out == int(lid)] *= np.float32(0.7)
            sw_out = (sw_out / max(np.float32(1e-6), np.float32(sw_out.mean()))).astype(np.float32, copy=False)
            return sw_out

        accepted_pred_val, _ = _predict_multiclass(model, split.X_val, extra)
        accepted_metric = compute_metrics(
            y_true=split.y_val,
            y_pred=accepted_pred_val,
            y_prob=val_prob,
            num_classes=len(stage_names),
        )
        accepted_head_absorb = _cic2024_stage2_head_absorb(accepted_pred_val)
        group_candidates = [
            ["DoS", "Exploits", "Reconnaissance"],
            ["Analysis", "Backdoor", "Shellcode", "Worms"],
        ]
        for group_names in group_candidates:
            cids = [int(stage_names.index(name)) for name in group_names if name in stage_names]
            if len(cids) < 2:
                continue
            tr_mask = np.isin(y_l.astype(np.int64, copy=False), np.asarray(cids, dtype=np.int64), assume_unique=False)
            va_mask = np.isin(split.y_val.astype(np.int64, copy=False), np.asarray(cids, dtype=np.int64), assume_unique=False)
            if int(tr_mask.sum()) < max(240, len(cids) * 80) or int(va_mask.sum()) < max(60, len(cids) * 18):
                continue
            group_map = {int(cid): i for i, cid in enumerate(cids)}
            y_group_train = np.asarray([group_map[int(v)] for v in y_l[tr_mask].tolist()], dtype=np.int64)
            y_group_val = np.asarray([group_map[int(v)] for v in split.y_val[va_mask].tolist()], dtype=np.int64)
            sw_group = _balanced_sample_weight(y_group_train, num_classes=len(cids))
            if w_l is not None and len(w_l) == len(y_l):
                base_w = w_l[tr_mask].astype(np.float32, copy=False)
                base_w = base_w / max(np.float32(1e-6), np.float32(base_w.mean()))
                sw_group = (sw_group * base_w).astype(np.float32, copy=False)
                sw_group = (sw_group / max(np.float32(1e-6), np.float32(sw_group.mean()))).astype(np.float32, copy=False)
            sw_group = _boost_cic2024_head_group_weights(cids, y_group_train, sw_group)
            group_model = _fit_xgb_multiclass(
                X_train=X_l[tr_mask],
                y_train=y_group_train,
                w_train=sw_group,
                X_val=split.X_val[va_mask],
                y_val=y_group_val,
                num_classes=len(cids),
                cfg=cfg,
            )
            target_head_group = bool(
                cfg.dataset == "cic2024"
                and {"DoS", "Exploits", "Reconnaissance"}.issubset({stage_names[int(cid)] for cid in cids})
            )
            best_local = None
            for mode in ["top2", "either", "top1"]:
                for sum_thr in [0.45, 0.55, 0.65, 0.75, 0.85]:
                    for margin_thr in [0.0, 0.12, 0.20, 0.30, 0.45]:
                        extra_try = extra.copy()
                        models_cur = list(extra_try.get("hier_group_models", []))
                        cids_cur = list(extra_try.get("hier_group_class_ids", []))
                        sum_cur = list(extra_try.get("hier_group_sum_thresholds", []))
                        margin_cur = list(extra_try.get("hier_group_margin_thresholds", []))
                        mode_cur = list(extra_try.get("hier_group_modes", []))
                        models_cur.append(group_model)
                        cids_cur.append(np.asarray(cids, dtype=np.int64))
                        sum_cur.append(float(sum_thr))
                        margin_cur.append(float(margin_thr))
                        mode_cur.append(str(mode))
                        extra_try["hier_group_models"] = models_cur
                        extra_try["hier_group_class_ids"] = cids_cur
                        extra_try["hier_group_sum_thresholds"] = np.asarray(sum_cur, dtype=np.float32)
                        extra_try["hier_group_margin_thresholds"] = np.asarray(margin_cur, dtype=np.float32)
                        extra_try["hier_group_modes"] = mode_cur
                        y_val_pred, _ = _predict_multiclass(model, split.X_val, extra_try)
                        met = compute_metrics(
                            y_true=split.y_val,
                            y_pred=y_val_pred,
                            y_prob=val_prob,
                            num_classes=len(stage_names),
                        )
                        head_absorb = _cic2024_stage2_head_absorb(y_val_pred)
                        key = (float(met.f1), float(met.recall), float(met.precision), -float(sum_thr), -float(margin_thr))
                        if target_head_group:
                            local_key = (-int(head_absorb), float(met.f1), float(met.recall), float(met.precision), -float(sum_thr), -float(margin_thr))
                        else:
                            local_key = key
                        if best_local is None or local_key > best_local[0]:
                            best_local = (local_key, extra_try, y_val_pred, met)
            if best_local is None:
                continue
            _, extra_best, pred_best, met_best = best_local
            head_absorb_best = _cic2024_stage2_head_absorb(pred_best)
            better_hier = (
                float(met_best.f1) > float(accepted_metric.f1) + 0.0005
                or (
                    float(met_best.f1) >= float(accepted_metric.f1) - 0.0005
                    and float(met_best.recall) > float(accepted_metric.recall) + 0.002
                )
                or (
                    cfg.dataset == "cic2024"
                    and head_absorb_best + 12 < accepted_head_absorb
                    and float(met_best.f1) >= float(accepted_metric.f1) - 0.010
                )
            )
            if better_hier:
                extra = extra_best
                accepted_pred_val = pred_best
                accepted_metric = met_best
                accepted_head_absorb = head_absorb_best

        twolevel_groups = [
            ["DoS", "Exploits", "Reconnaissance"],
            ["Analysis", "Backdoor", "Shellcode", "Worms"],
        ]
        explicit_groups: list[list[int]] = []
        used_classes: set[int] = set()
        for group_names in twolevel_groups:
            cids = [int(stage_names.index(name)) for name in group_names if name in stage_names]
            if len(cids) < 2:
                continue
            tr_mask = np.isin(y_l.astype(np.int64, copy=False), np.asarray(cids, dtype=np.int64), assume_unique=False)
            va_mask = np.isin(split.y_val.astype(np.int64, copy=False), np.asarray(cids, dtype=np.int64), assume_unique=False)
            if int(tr_mask.sum()) < max(320, len(cids) * 100) or int(va_mask.sum()) < max(90, len(cids) * 24):
                continue
            explicit_groups.append(cids)
            used_classes.update(int(v) for v in cids)
        if len(explicit_groups) > 0:
            router_groups: list[list[int]] = [list(g) for g in explicit_groups]
            for cid in range(len(stage_names)):
                if int(cid) not in used_classes:
                    router_groups.append([int(cid)])
            router_label_by_class = np.full(len(stage_names), -1, dtype=np.int64)
            for gid, cids in enumerate(router_groups):
                router_label_by_class[np.asarray(cids, dtype=np.int64)] = int(gid)
            y_router_train = router_label_by_class[y_l.astype(np.int64, copy=False)]
            y_router_val = router_label_by_class[split.y_val.astype(np.int64, copy=False)]
            sw_router = _balanced_sample_weight(y_router_train, num_classes=len(router_groups))
            if w_l is not None and len(w_l) == len(y_l):
                base_w = w_l.astype(np.float32, copy=False)
                base_w = base_w / max(np.float32(1e-6), np.float32(base_w.mean()))
                sw_router = (sw_router * base_w).astype(np.float32, copy=False)
                sw_router = (sw_router / max(np.float32(1e-6), np.float32(sw_router.mean()))).astype(np.float32, copy=False)
            if cfg.dataset == "cic2024":
                for gid, cids in enumerate(router_groups):
                    if _is_cic2024_head_cluster(cids):
                        sw_router[y_router_train == int(gid)] *= np.float32(1.8)
                sw_router = (sw_router / max(np.float32(1e-6), np.float32(sw_router.mean()))).astype(np.float32, copy=False)
            router_model = _fit_xgb_multiclass(
                X_train=X_l,
                y_train=y_router_train,
                w_train=sw_router,
                X_val=split.X_val,
                y_val=y_router_val,
                num_classes=len(router_groups),
                cfg=cfg,
            )
            accepted_pred_val, _ = _predict_multiclass(model, split.X_val, extra)
            accepted_metric = compute_metrics(
                y_true=split.y_val,
                y_pred=accepted_pred_val,
                y_prob=val_prob,
                num_classes=len(stage_names),
            )
            accepted_head_absorb = _cic2024_stage2_head_absorb(accepted_pred_val)
            for gid, cids in enumerate(router_groups):
                if len(cids) < 2:
                    continue
                tr_mask = np.isin(y_l.astype(np.int64, copy=False), np.asarray(cids, dtype=np.int64), assume_unique=False)
                va_mask = np.isin(split.y_val.astype(np.int64, copy=False), np.asarray(cids, dtype=np.int64), assume_unique=False)
                if int(tr_mask.sum()) < max(240, len(cids) * 80) or int(va_mask.sum()) < max(60, len(cids) * 18):
                    continue
                local_map = {int(cid): i for i, cid in enumerate(cids)}
                y_group_train = np.asarray([local_map[int(v)] for v in y_l[tr_mask].tolist()], dtype=np.int64)
                y_group_val = np.asarray([local_map[int(v)] for v in split.y_val[va_mask].tolist()], dtype=np.int64)
                sw_group = _balanced_sample_weight(y_group_train, num_classes=len(cids))
                if w_l is not None and len(w_l) == len(y_l):
                    base_w = w_l[tr_mask].astype(np.float32, copy=False)
                    base_w = base_w / max(np.float32(1e-6), np.float32(base_w.mean()))
                    sw_group = (sw_group * base_w).astype(np.float32, copy=False)
                    sw_group = (sw_group / max(np.float32(1e-6), np.float32(sw_group.mean()))).astype(np.float32, copy=False)
                sw_group = _boost_cic2024_head_group_weights(cids, y_group_train, sw_group)
                expert_model = _fit_xgb_multiclass(
                    X_train=X_l[tr_mask],
                    y_train=y_group_train,
                    w_train=sw_group,
                    X_val=split.X_val[va_mask],
                    y_val=y_group_val,
                    num_classes=len(cids),
                    cfg=cfg,
                )
                target_head_group = bool(
                    cfg.dataset == "cic2024"
                    and {"DoS", "Exploits", "Reconnaissance"}.issubset({stage_names[int(cid)] for cid in cids})
                )
                best_local = None
                for thr in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
                    for margin in [0.0, 0.02, 0.05, 0.08, 0.12, 0.18]:
                        extra_try = extra.copy()
                        models_cur = list(extra_try.get("twolevel_expert_models", []))
                        gids_cur = list(extra_try.get("twolevel_router_group_ids", []))
                        cids_cur = list(extra_try.get("twolevel_group_class_ids", []))
                        thr_cur = list(extra_try.get("twolevel_router_thresholds", []))
                        margin_cur = list(extra_try.get("twolevel_router_margins", []))
                        extra_try["twolevel_router_model"] = router_model
                        models_cur.append(expert_model)
                        gids_cur.append(int(gid))
                        cids_cur.append(np.asarray(cids, dtype=np.int64))
                        thr_cur.append(float(thr))
                        margin_cur.append(float(margin))
                        extra_try["twolevel_expert_models"] = models_cur
                        extra_try["twolevel_router_group_ids"] = np.asarray(gids_cur, dtype=np.int64)
                        extra_try["twolevel_group_class_ids"] = cids_cur
                        extra_try["twolevel_router_thresholds"] = np.asarray(thr_cur, dtype=np.float32)
                        extra_try["twolevel_router_margins"] = np.asarray(margin_cur, dtype=np.float32)
                        y_val_pred, _ = _predict_multiclass(model, split.X_val, extra_try)
                        met = compute_metrics(
                            y_true=split.y_val,
                            y_pred=y_val_pred,
                            y_prob=val_prob,
                            num_classes=len(stage_names),
                        )
                        head_absorb = _cic2024_stage2_head_absorb(y_val_pred)
                        key = (float(met.f1), float(met.recall), float(met.precision), -float(thr), -float(margin))
                        if target_head_group:
                            local_key = (
                                -int(head_absorb),
                                float(met.f1),
                                float(met.recall),
                                float(met.precision),
                                -float(thr),
                                -float(margin),
                            )
                        else:
                            local_key = key
                        if best_local is None or local_key > best_local[0]:
                            best_local = (local_key, extra_try, y_val_pred, met)
                if best_local is None:
                    continue
                _, extra_best, pred_best, met_best = best_local
                head_absorb_best = _cic2024_stage2_head_absorb(pred_best)
                better_twolevel = (
                    float(met_best.f1) > float(accepted_metric.f1) + 0.0005
                    or (
                        float(met_best.f1) >= float(accepted_metric.f1) - 0.0005
                        and float(met_best.recall) > float(accepted_metric.recall) + 0.002
                    )
                    or (
                        cfg.dataset == "cic2024"
                        and head_absorb_best + 20 < accepted_head_absorb
                        and float(met_best.f1) >= float(accepted_metric.f1) - 0.012
                    )
                )
                if better_twolevel:
                    extra = extra_best
                    accepted_pred_val = pred_best
                    accepted_metric = met_best
                    accepted_head_absorb = head_absorb_best

    stage2_pred_val, _ = _predict_multiclass(model, split.X_val, extra)
    best_val = compute_metrics(
        y_true=split.y_val,
        y_pred=stage2_pred_val,
        y_prob=val_prob,
        num_classes=len(stage_names),
    )
    return model, extra, val_prob, best_val


def run_cascade_feedback_search(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    stage1_split,
    p_train: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
    stage1_threshold: float,
    errors_out_path: str | None = None,
    selected_feature_cols: list[str] | None = None,
) -> None:
    if cfg.dataset == "dapt2020":
        use_act = bool(str(getattr(cfg, "stage2_label", "stage")) == "activity")
        df_task, _, feature_cols = make_stage_task(
            df,
            use_activity_as_stage=use_act,
            selected_feature_cols=selected_feature_cols,
        )
    else:
        df_task, _, feature_cols = make_stage_task(df, selected_feature_cols=selected_feature_cols)

    use_activity = bool(cfg.dataset == "dapt2020" and str(getattr(cfg, "stage2_label", "stage")) == "activity")
    malicious = df_task[df_task["Stage"].astype(str) != "Benign"].copy()
    if use_activity:
        act = malicious["Activity"].astype(str).where(malicious["Activity"].astype(str) != "Normal", other="Other")
        if int(getattr(cfg, "min_class_count", 1)) > 1:
            counts = act.value_counts()
            rare = counts[counts < int(getattr(cfg, "min_class_count", 1))].index.tolist()
            if rare:
                act = act.where(~act.isin(rare), other="Other")
        stage = act.astype(str)
    else:
        stage = malicious["Stage"].astype(str)
    stage_names = sorted(stage.unique().tolist())
    if len(stage_names) == 0:
        raise RuntimeError("No malicious samples found after preprocessing.")
    stage_to_id = {s: i for i, s in enumerate(stage_names)}
    y = stage.map(stage_to_id).astype(np.int64).to_numpy()

    row_ids = malicious["__row_id"].to_numpy(dtype=np.int64)
    idx_train_all = stage1_split.idx_train.astype(np.int64, copy=False)
    idx_val_all = stage1_split.idx_val.astype(np.int64, copy=False)
    idx_test_all = stage1_split.idx_test.astype(np.int64, copy=False)
    rare_set: set[str] = set()
    if use_activity and int(getattr(cfg, "min_class_count", 1)) > 1:
        counts = pd.Series(stage).value_counts()
        rare_set = set(counts[counts < int(getattr(cfg, "min_class_count", 1))].index.tolist())

    def _build_full_targets(idx_all: np.ndarray) -> np.ndarray:
        y_true_all = np.zeros(len(idx_all), dtype=np.int64)
        if use_activity:
            rows_all = df_task.iloc[idx_all][["Stage", "Activity"]]
            stage_true_all = rows_all["Stage"].astype(str).to_numpy()
            act_true_all = rows_all["Activity"].astype(str).to_numpy()
            act_true_all = np.where(stage_true_all == "Benign", "Normal", act_true_all).astype(str)
            act_true_all = np.where(act_true_all == "Normal", "Other", act_true_all).astype(str)
            if rare_set:
                act_true_all = np.array(["Other" if a in rare_set else a for a in act_true_all], dtype=object)
            for i, st in enumerate(stage_true_all.tolist()):
                if st != "Benign":
                    y_true_all[i] = 1 + int(stage_to_id.get(str(act_true_all[i]), 0))
        else:
            stage_true_all = df_task.iloc[idx_all]["Stage"].astype(str).to_numpy()
            for i, s in enumerate(stage_true_all):
                if s != "Benign":
                    y_true_all[i] = 1 + int(stage_to_id[s])
        return y_true_all

    y_true_all_train = _build_full_targets(idx_train_all)
    y_true_all_val = _build_full_targets(idx_val_all)
    y_true_all_test = _build_full_targets(idx_test_all)

    if getattr(cfg, "stage1_gate_method", "") == "f1":
        base = float(stage1_threshold)
        cand = [base - 0.12, base - 0.08, base - 0.04, base, base + 0.04, base + 0.08, base + 0.12]
        cand = [min(cfg.cascade_threshold_max, max(cfg.cascade_threshold_min, float(x))) for x in cand]
        thresholds = np.array(sorted(set(round(float(x), 3) for x in cand)), dtype=np.float32)
    else:
        thresholds = np.asarray(
            sorted(
                set(
                    np.round(
                        np.concatenate(
                            [
                                np.linspace(cfg.cascade_threshold_min, cfg.cascade_threshold_max, cfg.cascade_threshold_steps),
                                np.asarray(
                                    [
                                        float(stage1_threshold),
                                        float(stage1_threshold) - 0.10,
                                        float(stage1_threshold) - 0.05,
                                        float(stage1_threshold) + 0.05,
                                    ],
                                    dtype=np.float32,
                                ),
                            ]
                        ),
                        6,
                    ).tolist()
                )
            ),
            dtype=np.float32,
        )
    stage2_min_conf_candidates = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
    stage2_entropy_max_candidates = [1.01]
    if cfg.dataset == "cic2024" or len(df_task) >= 1_000_000:
        stage2_min_conf_candidates = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
        stage2_entropy_max_candidates = [1.01, 0.95, 0.9, 0.85, 0.8, 0.75]

    def _norm_entropy(prob: np.ndarray) -> np.ndarray:
        p = prob.astype(np.float32, copy=False)
        eps = np.float32(1e-9)
        p = np.clip(p, eps, 1.0)
        ent = -(p * np.log(p)).sum(axis=1)
        k = float(p.shape[1])
        denom = float(np.log(max(2.0, k)))
        return (ent / denom).astype(np.float32, copy=False)

    stage2_margin_candidates = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    stage2_joint_candidates = [0.0]
    if cfg.dataset == "dapt2020":
        stage2_joint_candidates = [0.0, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93, 0.95]
    if cfg.dataset == "cic2024" or len(df_task) >= 1_000_000:
        stage2_margin_candidates = [0.0, 0.02, 0.05, 0.08, 0.10]

    records: list[tuple[float, float, float, float, float, float, Metrics, Metrics, int, int, int, int]] = []

    if len(stage_names) == 1:
        for t in thresholds:
            gate_mask_val = p_val >= t
            if not gate_mask_val.any():
                continue
            y_pred_all = np.zeros(len(idx_val_all), dtype=np.int64)
            y_pred_all[gate_mask_val] = 1
            end2end_val = compute_metrics(
                y_true=y_true_all_val,
                y_pred=y_pred_all,
                y_prob=None,
                num_classes=2,
            )
            obj = float(min(end2end_val.acc, end2end_val.f1, end2end_val.precision, end2end_val.recall))
            records.append((obj, float(t), 0.0, end2end_val, end2end_val, 0, int(gate_mask_val.sum())))

        if len(records) == 0:
            raise RuntimeError("No valid cascade threshold found for feedback search.")

        records.sort(key=lambda x: x[0], reverse=True)
        best_obj = float(records[0][0])
        margin = 0.01
        eligible = [r for r in records if float(r[0]) >= best_obj - margin]
        chosen = max(eligible, key=lambda r: (float(r[1]), float(r[0])))
        best_obj, best_t = float(chosen[0]), float(chosen[1])

        idx_test_all = stage1_split.idx_test.astype(np.int64, copy=False)
        y_true_e2e = np.zeros(len(idx_test_all), dtype=np.int64)
        stage_true_test = df_task.iloc[idx_test_all]["Stage"].astype(str).to_numpy()
        for i, s in enumerate(stage_true_test):
            if s != "Benign":
                y_true_e2e[i] = 1

        y_pred_e2e = np.zeros(len(idx_test_all), dtype=np.int64)
        gate_mask_test = p_test >= best_t
        y_pred_e2e[gate_mask_test] = 1
        labels_all = ["Benign", stage_names[0]]
        metric_e2e = compute_metrics(y_true=y_true_e2e, y_pred=y_pred_e2e, y_prob=None, num_classes=2)
        stage2_val_pred = np.zeros(len(idx_val_all), dtype=np.int64)
        stage2_val_pred[p_val >= best_t] = 1
        stage2_metric_val = compute_metrics(y_true=y_true_all_val, y_pred=stage2_val_pred, y_prob=None, num_classes=2)
        if cfg.verbose:
            print("Stage2 Val:", _macro_weighted_summary(y_true_all_val, stage2_val_pred))
            print(format_confusion_matrix(stage2_metric_val.cm, labels=labels_all))
            print("Stage2 Test:", _macro_weighted_summary(y_true_e2e, y_pred_e2e))
            print(format_confusion_matrix(metric_e2e.cm, labels=labels_all))

        e2e = _macro_weighted_summary(y_true_e2e, y_pred_e2e)
        e2e["macro_acc"] = _macro_acc_from_cm(metric_e2e.cm)
        e2e["fpr"] = _benign_fpr_from_cm(metric_e2e.cm)
        e2e["macro_fpr"], e2e["weighted_fpr"] = _fpr_macro_weighted_from_cm(metric_e2e.cm)
        e2e["macro_auc"], e2e["weighted_auc"] = _auc_macro_weighted_ovr(
            y_true_e2e,
            np.stack([1.0 - p_test, p_test], axis=1),
        )
        print("End2End Test:", e2e)
        print(format_confusion_matrix(metric_e2e.cm, labels=labels_all))
        _write_json_silent(
            os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_seed{cfg.seed}_end2end_metrics.json"),
            {
                "dataset": cfg.dataset,
                "seed": int(cfg.seed),
                "drop_stages": str(cfg.drop_stages),
                "labels": labels_all,
                "metrics": e2e,
                "confusion_matrix": metric_e2e.cm.tolist(),
            },
        )

        ensure_dir(cfg.checkpoint_dir)
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_cascade_feedback_seed{cfg.seed}_best.pt")
        _save_checkpoint(
            model=None,
            cfg=cfg,
            split=stage1_split,
            class_names=labels_all,
            best_val=stage2_metric_val,
            out_path=ckpt_path,
            extra={"stage1_threshold": float(best_t), "cascade_objective": cfg.cascade_objective, "single_stage": stage_names[0]},
        )
        return

    idx_train_mal = np.where(np.isin(row_ids, stage1_split.row_id_train))[0].astype(np.int64)
    idx_val_mal = np.where(np.isin(row_ids, stage1_split.row_id_val))[0].astype(np.int64)
    idx_test_mal = np.where(np.isin(row_ids, stage1_split.row_id_test))[0].astype(np.int64)
    if len(idx_train_mal) == 0:
        idx_train_mal = np.arange(len(row_ids), dtype=np.int64)
    if len(idx_val_mal) == 0:
        idx_val_mal = idx_train_mal[: min(1, len(idx_train_mal))]
    if len(idx_test_mal) == 0:
        idx_test_mal = idx_val_mal[: min(1, len(idx_val_mal))]

    rng_stage2 = np.random.default_rng(cfg.seed)
    stage2_val_ratio = float(np.clip(cfg.val_size, 0.05, 0.3))
    idx_train_pool = idx_train_mal.astype(np.int64, copy=False)
    idx_val_from_train: list[int] = []
    if len(idx_train_pool) > 0:
        for c in range(len(stage_names)):
            idx_c = idx_train_pool[y[idx_train_pool] == int(c)]
            if len(idx_c) <= 1:
                continue
            idx_c = idx_c.copy()
            rng_stage2.shuffle(idx_c)
            k = int(np.ceil(len(idx_c) * stage2_val_ratio))
            k = max(1, min(len(idx_c) - 1, k))
            idx_val_from_train.extend(idx_c[:k].tolist())

    if len(idx_val_from_train) > 0:
        idx_val_mal = np.asarray(sorted(set(idx_val_from_train)), dtype=np.int64)
        val_set = set(int(i) for i in idx_val_mal.tolist())
        idx_train_mal = np.asarray([int(i) for i in idx_train_pool.tolist() if int(i) not in val_set], dtype=np.int64)
        if len(idx_train_mal) == 0:
            idx_train_mal = idx_train_pool

    if len(set(y[idx_train_mal].tolist())) != len(stage_names):
        missing = [stage_names[c] for c in range(len(stage_names)) if c not in set(y[idx_train_mal].tolist())]
        if cfg.verbose and missing:
            print("Stage2 warning: classes absent from malicious train split:", missing)

    split = scale_by_indices(
        df=malicious,
        y=y,
        feature_cols=feature_cols,
        idx_train=idx_train_mal,
        idx_val=idx_val_mal,
        idx_test=idx_test_mal,
    )

    stage2_model, stage2_extra, _, stage2_best_val = _train_stage2_semi(split, stage_names, cfg)
    cic2024_head_branch_forced_ablation = bool(
        cfg.dataset == "cic2024" and stage2_extra.get("cic2024_head_cluster_model") is not None
    )
    if cic2024_head_branch_forced_ablation:
        fixed_t = float(stage1_threshold)
        fixed_min_conf = float(stage2_extra.get("stage2_min_conf", 0.5) or 0.5)
        fixed_ent_thr = float(stage2_extra.get("stage2_entropy_max", 0.75) or 0.75)
        fixed_margin_thr = float(stage2_extra.get("stage2_margin_min", 0.08) or 0.08)
        fixed_joint_thr = float(stage2_extra.get("stage2_joint_min", 0.0) or 0.0)
        thresholds = np.asarray([fixed_t], dtype=np.float32)
        stage2_min_conf_candidates = [fixed_min_conf]
        stage2_entropy_max_candidates = [fixed_ent_thr]
        stage2_margin_candidates = [fixed_margin_thr]
        stage2_joint_candidates = [fixed_joint_thr]

    class_accept_min_prob = None
    class_accept_stage1_min = None
    class_accept_ovr_thresholds: dict[int, float] = {}
    class_accept_ovr_scores_val: dict[int, np.ndarray] = {}
    class_accept_ovr_scores_test: dict[int, np.ndarray] = {}
    ovr_full_scores_val: dict[int, np.ndarray] = {}
    ovr_full_scores_test: dict[int, np.ndarray] = {}
    class_accept_joint_thresholds: dict[int, float] = {}
    class_rescue_joint_thresholds: dict[int, float] = {}
    class_rescue_stage1_mins: dict[int, float] = {}
    pair_refine_pairs: list[tuple[int, int]] = []
    pair_refine_thresholds: list[float] = []
    pair_refine_models: list[object] = []
    if cfg.dataset in {"weekdata", "earlycrow", "cic2024"} and len(stage_names) >= 2:
        earlycrow_tail_rescue_names = {"onionduke1", "poisonivy1", "zebrocy1", "zebrocy2", "zebrocy3"}
        cic2024_tail_rescue_names = {"Backdoor", "Worms", "Shellcode", "Analysis", "Reconnaissance", "DoS"}
        base_stage2_pred_val, _ = _predict_multiclass(stage2_model, split.X_val, stage2_extra)
        pair_specs: list[tuple[str, str]] = []
        if cfg.dataset == "weekdata":
            pair_specs = [
                (
                    "Active Scanning: Scanning IP Blocks",
                    "Active Scanning: Vulnerability Scanning",
                ),
                (
                    "Encrypted Channel: Symmetric Cryptography",
                    "Exfiltration over C2 channel",
                ),
            ]
        elif cfg.dataset == "earlycrow":
            pair_specs = [
                ("PlugX1", "onionduke1"),
                ("Zeus", "onionduke1"),
                ("PlugX1", "poisonivy1"),
                ("Zeus", "poisonivy1"),
                ("Sogou", "zebrocy1"),
                ("Zeus", "zebrocy1"),
                ("Zeus", "zebrocy2"),
                ("Zeus", "zebrocy3"),
            ]
        elif cfg.dataset == "cic2024":
            pair_specs = [
                # Rescue under-recalled classes that are being absorbed into Exploits.
                ("Exploits", "DoS"),
                ("Exploits", "Reconnaissance"),
                ("Exploits", "Backdoor"),
                ("Exploits", "Worms"),
                ("Backdoor", "Shellcode"),
            ]
        pair_thr_grid = np.array(
            [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95],
            dtype=np.float32,
        )
        for name_a, name_b in pair_specs:
            if name_a not in stage_names or name_b not in stage_names:
                continue
            cid_a = int(stage_names.index(name_a))
            cid_b = int(stage_names.index(name_b))
            train_mask = np.isin(split.y_train.astype(np.int64, copy=False), [cid_a, cid_b])
            val_mask = np.isin(split.y_val.astype(np.int64, copy=False), [cid_a, cid_b])
            min_train = 8
            min_val = 2
            if cfg.dataset == "earlycrow":
                min_train = 24
                min_val = 6
            elif cfg.dataset == "cic2024":
                min_train = 40
                min_val = 12
            if int(train_mask.sum()) < min_train or int(val_mask.sum()) < min_val:
                continue
            y_pair_train = (split.y_train[train_mask].astype(np.int64, copy=False) == cid_b).astype(np.int64, copy=False)
            y_pair_val = (split.y_val[val_mask].astype(np.int64, copy=False) == cid_b).astype(np.int64, copy=False)
            if len(np.unique(y_pair_train)) < 2 or len(np.unique(y_pair_val)) < 2:
                continue
            pos_count = int(y_pair_train.sum())
            neg_count = int(len(y_pair_train) - pos_count)
            if pos_count <= 0 or neg_count <= 0:
                continue
            force_pair_recall = bool(
                cfg.dataset == "earlycrow"
                and (name_a in earlycrow_tail_rescue_names or name_b in earlycrow_tail_rescue_names)
            )
            if cfg.dataset == "cic2024" and (name_a in cic2024_tail_rescue_names or name_b in cic2024_tail_rescue_names):
                force_pair_recall = True
            pair_sw = np.where(y_pair_train == 1, float(neg_count / max(1, pos_count)), 1.0).astype(np.float32)
            if pos_count <= 32:
                pair_sw[y_pair_train == 1] = pair_sw[y_pair_train == 1] * np.float32(1.5)
            if cfg.dataset == "earlycrow":
                if pos_count <= 128:
                    pair_sw[y_pair_train == 1] = pair_sw[y_pair_train == 1] * np.float32(1.75)
                if pos_count <= 32:
                    pair_sw[y_pair_train == 1] = pair_sw[y_pair_train == 1] * np.float32(1.5)
            elif cfg.dataset == "cic2024":
                if name_b in cic2024_tail_rescue_names:
                    pair_sw[y_pair_train == 1] = pair_sw[y_pair_train == 1] * np.float32(1.5)
                if pos_count <= 128:
                    pair_sw[y_pair_train == 1] = pair_sw[y_pair_train == 1] * np.float32(1.35)
            pair_sw = pair_sw / float(np.mean(pair_sw))
            if _can_use_xgb_cuda():
                pair_model = _fit_xgb_binary(
                    X_train=split.X_train[train_mask],
                    y_train=y_pair_train,
                    w_train=pair_sw,
                    X_val=split.X_val[val_mask],
                    y_val=y_pair_val,
                    cfg=cfg,
                )
            else:
                pair_model = HistGradientBoostingClassifier(
                    learning_rate=cfg.hgb_learning_rate,
                    max_iter=max(400, int(cfg.hgb_max_iter)),
                    random_state=cfg.seed,
                )
                pair_model.fit(split.X_train[train_mask], y_pair_train, sample_weight=pair_sw)
            pair_prob_val = pair_model.predict_proba(split.X_val[val_mask])[:, 1].astype(np.float32, copy=False)
            base_pair_pred = base_stage2_pred_val[val_mask].astype(np.int64, copy=False)
            y_pair_val_full = np.where(y_pair_val == 1, cid_b, cid_a).astype(np.int64, copy=False)
            base_macro = float(f1_score(y_pair_val_full, base_pair_pred, average="macro", zero_division=0))
            base_recall_b = float(
                recall_score(y_pair_val, (base_pair_pred == cid_b).astype(np.int64, copy=False), zero_division=0)
            )
            best_local = None
            for thr in pair_thr_grid.tolist():
                pred_bin = (pair_prob_val >= float(thr)).astype(np.int64, copy=False)
                pred_full = np.where(pred_bin == 1, cid_b, cid_a).astype(np.int64, copy=False)
                macro = float(f1_score(y_pair_val_full, pred_full, average="macro", zero_division=0))
                recall_b = float(recall_score(y_pair_val, pred_bin, zero_division=0))
                precision_b = float(precision_score(y_pair_val, pred_bin, zero_division=0))
                if force_pair_recall:
                    key = (recall_b, macro, precision_b, -abs(float(thr) - 0.5))
                else:
                    key = (macro, recall_b, precision_b, -abs(float(thr) - 0.5))
                if best_local is None or key > best_local[0]:
                    best_local = (key, float(thr), float(macro), float(recall_b))
            if best_local is None:
                continue
            allow_equal = bool(pos_count <= 32 or cfg.dataset in {"earlycrow", "cic2024"})
            if (
                force_pair_recall
                and float(best_local[3]) > float(base_recall_b) + 1e-6
                and float(best_local[2]) + 0.05 >= float(base_macro)
            ) or float(best_local[2]) > float(base_macro) + 1e-6 or (
                allow_equal
                and float(best_local[2]) + 1e-6 >= float(base_macro)
                and float(best_local[3]) + 1e-6 >= float(base_recall_b)
            ):
                pair_refine_pairs.append((cid_a, cid_b))
                pair_refine_thresholds.append(float(best_local[1]))
                pair_refine_models.append(pair_model)
        if pair_refine_pairs:
            stage2_extra = stage2_extra.copy()
            stage2_extra["pair_refine_class_pairs"] = np.asarray(pair_refine_pairs, dtype=np.int64)
            stage2_extra["pair_refine_thresholds"] = np.asarray(pair_refine_thresholds, dtype=np.float32)
            stage2_extra["pair_refine_models"] = pair_refine_models
    enable_openworld_class_calibration = bool(
        cfg.dataset in {"dapt2020", "earlycrow", "weekdata"}
        or getattr(cfg, "oversample_rare_classes", False)
    )
    if enable_openworld_class_calibration:
        benign_val_mask = y_true_all_val == 0
        if benign_val_mask.any():
            X_benign_val = df_task.iloc[idx_val_all[benign_val_mask]][feature_cols].to_numpy(dtype=np.float32, copy=True)
            X_benign_val = split.scaler.transform(X_benign_val).astype(np.float32, copy=False)
            _, benign_stage_prob = _predict_multiclass(stage2_model, X_benign_val, stage2_extra)
            _, val_stage_prob = _predict_multiclass(stage2_model, split.X_val, stage2_extra)
            if benign_stage_prob is not None and val_stage_prob is not None:
                class_accept = np.zeros(len(stage_names), dtype=np.float32)
                class_stage1_floor = np.zeros(len(stage_names), dtype=np.float32)
                class_counts = np.bincount(split.y_train.astype(np.int64, copy=False), minlength=len(stage_names)).astype(np.int64)
                thr_grid = _class_accept_prob_grid()
                benign_stage_pred = benign_stage_prob.argmax(axis=1).astype(np.int64, copy=False)
                benign_stage1_scores = p_val[benign_val_mask].astype(np.float32, copy=False)
                for cid in range(len(stage_names)):
                    pos_scores = val_stage_prob[split.y_val.astype(np.int64, copy=False) == int(cid), int(cid)]
                    neg_scores = benign_stage_prob[:, int(cid)]
                    if len(pos_scores) == 0 or len(neg_scores) == 0:
                        continue
                    rare_cls = int(class_counts[int(cid)]) <= max(64, int(getattr(cfg, "oversample_target_count", 64)))
                    best_local = None
                    for thr in thr_grid.tolist():
                        tp = int((pos_scores >= float(thr)).sum())
                        fp = int((neg_scores >= float(thr)).sum())
                        fn = int(len(pos_scores) - tp)
                        precision = float(tp / max(1, tp + fp))
                        recall = float(tp / max(1, tp + fn))
                        f1 = 0.0 if tp == 0 else float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))
                        fp_rate = float(fp / max(1, len(neg_scores)))
                        if rare_cls:
                            key = (float(f1), float(precision), -float(fp_rate), float(recall), float(thr))
                        else:
                            key = (float(f1), -float(fp_rate), float(precision), float(recall), float(thr))
                        if best_local is None or key > best_local[0]:
                            best_local = (key, float(thr))
                    if best_local is not None:
                        class_accept[int(cid)] = float(best_local[1])

                    # Risky classes that attract benigns need a stricter Stage-I floor.
                    ben_cls_scores = benign_stage1_scores[benign_stage_pred == int(cid)]
                    pos_stage1_scores = p_val[y_true_all_val == (1 + int(cid))].astype(np.float32, copy=False)
                    if len(pos_stage1_scores) == 0 or len(ben_cls_scores) == 0:
                        continue
                    rare_cls = int(class_counts[int(cid)]) <= max(64, int(getattr(cfg, "oversample_target_count", 64)))
                    stage1_grid = sorted(
                        set(
                            np.round(
                                np.concatenate(
                                    [
                                        np.linspace(
                                            float(max(0.05, stage1_threshold)),
                                            float(min(0.99, max(stage1_threshold, float(np.max(pos_stage1_scores)) + 0.02))),
                                            24,
                                        ),
                                        np.quantile(pos_stage1_scores, [0.05, 0.1, 0.2, 0.5, 0.8, 0.9]),
                                        np.quantile(ben_cls_scores, [0.8, 0.9, 0.95, 0.99]),
                                    ]
                                ),
                                3,
                            ).tolist()
                        )
                    )
                    best_stage1_local = None
                    for thr_s1 in stage1_grid:
                        thr_s1 = float(max(stage1_threshold, min(0.99, thr_s1)))
                        tp = int((pos_stage1_scores >= thr_s1).sum())
                        fp = int((ben_cls_scores >= thr_s1).sum())
                        fn = int(len(pos_stage1_scores) - tp)
                        precision = float(tp / max(1, tp + fp))
                        recall = float(tp / max(1, tp + fn))
                        f1 = 0.0 if tp == 0 else float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))
                        if rare_cls:
                            key = (float(min(f1, precision)), float(precision), -int(fp), float(recall), float(thr_s1))
                        else:
                            key = (float(f1), float(precision), -int(fp), float(recall), float(thr_s1))
                        if best_stage1_local is None or key > best_stage1_local[0]:
                            best_stage1_local = (key, float(thr_s1))
                    if best_stage1_local is not None and float(best_stage1_local[1]) > float(stage1_threshold) + 0.01:
                        class_stage1_floor[int(cid)] = float(best_stage1_local[1])
                    if cfg.dataset == "weekdata" and str(stage_names[int(cid)]) == "Hijack Execution Flow: Path Interception by PATH Environment Variable":
                        pos_joint_scores = (pos_scores.astype(np.float32, copy=False) * pos_stage1_scores).astype(np.float32, copy=False)
                        benign_joint_scores = (neg_scores.astype(np.float32, copy=False) * benign_stage1_scores).astype(
                            np.float32,
                            copy=False,
                        )
                        if len(pos_joint_scores) > 0 and len(benign_joint_scores) > 0:
                            joint_grid = sorted(
                                set(
                                    np.round(
                                        np.concatenate(
                                            [
                                                np.linspace(
                                                    float(max(0.50, min(np.min(pos_joint_scores), np.max(benign_joint_scores)))),
                                                    float(min(0.999, max(np.max(pos_joint_scores), np.max(benign_joint_scores)))),
                                                    24,
                                                ),
                                                np.quantile(pos_joint_scores, [0.0, 0.05, 0.1, 0.2, 0.5]),
                                                np.quantile(benign_joint_scores, [0.9, 0.95, 0.99, 0.999]),
                                            ]
                                        ),
                                        4,
                                    ).tolist()
                                )
                            )
                            best_joint_local = None
                            for thr_joint in joint_grid:
                                thr_joint = float(max(0.0, min(0.9999, thr_joint)))
                                tp = int((pos_joint_scores >= thr_joint).sum())
                                fp = int((benign_joint_scores >= thr_joint).sum())
                                fn = int(len(pos_joint_scores) - tp)
                                precision = float(tp / max(1, tp + fp))
                                recall = float(tp / max(1, tp + fn))
                                f1 = 0.0 if tp == 0 else float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))
                                key = (float(min(f1, precision)), float(precision), -int(fp), float(recall), float(thr_joint))
                                if best_joint_local is None or key > best_joint_local[0]:
                                    best_joint_local = (key, float(thr_joint))
                            if best_joint_local is not None:
                                class_accept_joint_thresholds[int(cid)] = float(best_joint_local[1])
                if np.any(class_accept > 0.0):
                    class_accept_min_prob = class_accept.astype(np.float32, copy=False)
                    stage2_extra = stage2_extra.copy()
                    stage2_extra["class_accept_min_prob"] = class_accept_min_prob
                if np.any(class_stage1_floor > 0.0):
                    class_accept_stage1_min = class_stage1_floor.astype(np.float32, copy=False)
                    stage2_extra = stage2_extra.copy()
                    stage2_extra["class_accept_stage1_min"] = class_accept_stage1_min
                if class_accept_joint_thresholds:
                    stage2_extra = stage2_extra.copy()
                    stage2_extra["class_accept_joint_class_ids"] = np.asarray(
                        sorted(class_accept_joint_thresholds.keys()),
                        dtype=np.int64,
                    )
                    stage2_extra["class_accept_joint_thresholds"] = np.asarray(
                        [float(class_accept_joint_thresholds[int(cid)]) for cid in sorted(class_accept_joint_thresholds.keys())],
                        dtype=np.float32,
                    )

        ovr_full_models = stage2_extra.get("ovr_full_models", None)
        ovr_full_thresholds = stage2_extra.get("ovr_full_thresholds", None)
        ovr_full_class_ids = stage2_extra.get("ovr_full_class_ids", None)
        if ovr_full_models is not None and ovr_full_thresholds is not None and ovr_full_class_ids is not None:
            cids = np.asarray(ovr_full_class_ids, dtype=np.int64)
            base_thr = np.asarray(ovr_full_thresholds, dtype=np.float32)
            if cids.ndim == 1 and base_thr.ndim == 1 and len(cids) == len(base_thr) and len(ovr_full_models) == len(cids):
                X_train_all = df_task.iloc[idx_train_all][feature_cols].to_numpy(dtype=np.float32, copy=True)
                X_val_all = df_task.iloc[idx_val_all][feature_cols].to_numpy(dtype=np.float32, copy=True)
                X_test_all = df_task.iloc[idx_test_all][feature_cols].to_numpy(dtype=np.float32, copy=True)
                X_train_all = split.scaler.transform(X_train_all).astype(np.float32, copy=False)
                X_val_all = split.scaler.transform(X_val_all).astype(np.float32, copy=False)
                X_test_all = split.scaler.transform(X_test_all).astype(np.float32, copy=False)
                class_counts = np.bincount(split.y_train.astype(np.int64, copy=False), minlength=len(stage_names)).astype(np.int64)
                thr_grid = _class_accept_prob_grid()
                selected_ids: list[int] = []
                selected_thr: list[float] = []
                benign_train_mask = y_true_all_train == 0
                benign_val_mask = y_true_all_val == 0
                earlycrow_tail_rescue_names = {"onionduke1", "poisonivy1", "zebrocy1", "zebrocy2", "zebrocy3"}
                cic2024_force_rescue_names = {"Analysis", "Backdoor", "DoS", "Reconnaissance", "Shellcode", "Worms"}
                for model_ovr, cid_raw, thr_raw in zip(ovr_full_models, cids.tolist(), base_thr.tolist()):
                    cid = int(cid_raw)
                    if cid < 0 or cid >= len(stage_names):
                        continue
                    stage_name = str(stage_names[cid])
                    force_tail_rescue = bool(
                        (cfg.dataset == "earlycrow" and stage_name in earlycrow_tail_rescue_names)
                        or (cfg.dataset == "cic2024" and stage_name in cic2024_force_rescue_names)
                    )
                    cls_count = int(class_counts[cid]) if cid < len(class_counts) else 0
                    if (not force_tail_rescue) and cls_count > max(128, int(getattr(cfg, "oversample_target_count", 64)) * 2):
                        continue
                    train_scores = model_ovr.predict_proba(X_train_all)[:, 1].astype(np.float32, copy=False)
                    val_scores = model_ovr.predict_proba(X_val_all)[:, 1].astype(np.float32, copy=False)
                    test_scores = model_ovr.predict_proba(X_test_all)[:, 1].astype(np.float32, copy=False)
                    ovr_full_scores_val[cid] = val_scores
                    ovr_full_scores_test[cid] = test_scores
                    pos_train = train_scores[y_true_all_train == (1 + cid)]
                    ben_train = train_scores[benign_train_mask]
                    pos_val = val_scores[y_true_all_val == (1 + cid)]
                    ben_val = val_scores[benign_val_mask]
                    if len(pos_train) == 0 or len(pos_val) == 0:
                        continue

                    if cfg.dataset == "cic2024":
                        # CIC2024 often ends up with a relatively conservative final gate
                        # (around 0.7-0.8). Restricting below-gate rescue to <0.55 leaves
                        # many borderline malicious flows, especially Analysis, with no
                        # rescue path at all.
                        rescue_gate_high = float(min(float(stage1_threshold), 0.80))
                        rescue_gate_low = float(max(0.30, rescue_gate_high - 0.40))
                    else:
                        rescue_gate_high = float(min(float(stage1_threshold), 0.55))
                        rescue_gate_low = float(max(0.20, rescue_gate_high - 0.25))
                    below_gate_pos_train = (
                        (y_true_all_train == (1 + cid))
                        & (p_train >= np.float32(rescue_gate_low))
                        & (p_train < np.float32(rescue_gate_high))
                    )
                    below_gate_benign_train = (
                        (y_true_all_train == 0)
                        & (p_train >= np.float32(rescue_gate_low))
                        & (p_train < np.float32(rescue_gate_high))
                    )
                    if int(below_gate_pos_train.sum()) >= 4:
                        pos_joint_train = (
                            p_train[below_gate_pos_train].astype(np.float32, copy=False)
                            * train_scores[below_gate_pos_train]
                        ).astype(np.float32, copy=False)
                        benign_joint_train = (
                            p_train[below_gate_benign_train].astype(np.float32, copy=False)
                            * train_scores[below_gate_benign_train]
                        ).astype(np.float32, copy=False)
                        pos_stage1_train = p_train[below_gate_pos_train].astype(np.float32, copy=False)
                        pos_floor_q = 0.05 if len(pos_joint_train) >= 10 else 0.0
                        neg_cap_q = 0.9995
                        allow_gap = 0.005
                        if cfg.dataset == "cic2024" and force_tail_rescue:
                            pos_floor_q = 0.02 if len(pos_joint_train) >= 20 else 0.0
                            neg_cap_q = 0.9998
                            allow_gap = -0.01
                        pos_floor_train = float(np.quantile(pos_joint_train, pos_floor_q))
                        neg_cap_train = float(np.quantile(benign_joint_train, neg_cap_q)) if len(benign_joint_train) > 0 else 0.0
                        if pos_floor_train > neg_cap_train + allow_gap:
                            rescue_thr = float((pos_floor_train + neg_cap_train) / 2.0)
                            if cfg.dataset == "cic2024" and force_tail_rescue:
                                rescue_thr = float(max(rescue_thr, neg_cap_train))
                            rescue_stage1_min = float(
                                max(
                                    rescue_gate_low,
                                    min(
                                        float(rescue_gate_high) - 1e-4,
                                        float(
                                            np.quantile(
                                                pos_stage1_train,
                                                0.02 if (cfg.dataset == "cic2024" and force_tail_rescue and len(pos_stage1_train) >= 20) else (0.05 if len(pos_stage1_train) >= 10 else 0.0),
                                            )
                                        ),
                                    ),
                                )
                            )
                            class_rescue_joint_thresholds[cid] = rescue_thr
                            class_rescue_stage1_mins[cid] = rescue_stage1_min

                    rare_cls = cls_count <= max(64, int(getattr(cfg, "oversample_target_count", 64)))
                    neg_q = float(np.quantile(ben_train, 0.999 if (rare_cls or force_tail_rescue) else 0.9995))
                    pos_q = float(np.quantile(pos_train, 0.15 if force_tail_rescue else (0.10 if rare_cls else 0.05)))
                    floor_thr = max(float(thr_raw), neg_q, 0.35 if force_tail_rescue else (0.55 if rare_cls else 0.45))
                    if floor_thr > pos_q + 0.05:
                        floor_thr = max(float(thr_raw), min(pos_q, 0.98))
                    candidates = [float(v) for v in thr_grid.tolist() if float(v) + 1e-9 >= float(floor_thr)]
                    if not candidates:
                        candidates = [float(min(0.99, max(float(thr_raw), floor_thr)))]
                    best_local = None
                    y_pos_val = y_true_all_val == (1 + cid)
                    for thr in candidates:
                        pred_pos = val_scores >= float(thr)
                        tp = int(np.logical_and(y_pos_val, pred_pos).sum())
                        fp = int(np.logical_and(~y_pos_val, pred_pos).sum())
                        fn = int(np.logical_and(y_pos_val, ~pred_pos).sum())
                        fp_ben = int((ben_val >= float(thr)).sum())
                        precision = float(tp / max(1, tp + fp))
                        recall = float(tp / max(1, tp + fn))
                        f1 = 0.0 if tp == 0 else float((2.0 * tp) / max(1.0, (2.0 * tp + fp + fn)))
                        if force_tail_rescue:
                            key = (
                                float(recall),
                                float(min(f1, precision)),
                                float(precision),
                                -int(fp_ben),
                                -float(thr),
                            )
                        elif rare_cls:
                            key = (
                                float(min(f1, precision)),
                                float(precision),
                                -int(fp_ben),
                                float(recall),
                                float(thr),
                            )
                        else:
                            key = (
                                float(f1),
                                float(precision),
                                -int(fp_ben),
                                float(recall),
                                float(thr),
                            )
                        if best_local is None or key > best_local[0]:
                            best_local = (key, float(thr))
                    if best_local is None:
                        continue
                    chosen_thr = float(best_local[1])
                    if force_tail_rescue:
                        chosen_thr = float(min(chosen_thr, max(float(thr_raw), 0.20)))
                    selected_ids.append(cid)
                    selected_thr.append(chosen_thr)
                    class_accept_ovr_thresholds[cid] = chosen_thr
                    class_accept_ovr_scores_val[cid] = val_scores
                    class_accept_ovr_scores_test[cid] = test_scores

                if selected_ids:
                    stage2_extra = stage2_extra.copy()
                    stage2_extra["class_accept_ovr_class_ids"] = np.asarray(selected_ids, dtype=np.int64)
                    stage2_extra["class_accept_ovr_thresholds"] = np.asarray(selected_thr, dtype=np.float32)
                if class_rescue_joint_thresholds:
                    stage2_extra = stage2_extra.copy()
                    stage2_extra["class_rescue_ovr_class_ids"] = np.asarray(sorted(class_rescue_joint_thresholds.keys()), dtype=np.int64)
                    stage2_extra["class_rescue_joint_thresholds"] = np.asarray(
                        [float(class_rescue_joint_thresholds[int(cid)]) for cid in sorted(class_rescue_joint_thresholds.keys())],
                        dtype=np.float32,
                    )
                    stage2_extra["class_rescue_stage1_mins"] = np.asarray(
                        [float(class_rescue_stage1_mins[int(cid)]) for cid in sorted(class_rescue_joint_thresholds.keys())],
                        dtype=np.float32,
                    )

    def _apply_class_accept_ovr_mask(
        gate_pos_local: np.ndarray,
        stage_pred_local: np.ndarray,
        keep_local: np.ndarray,
        score_bank: dict[int, np.ndarray],
    ) -> np.ndarray:
        if not class_accept_ovr_thresholds:
            return keep_local
        keep_out = np.asarray(keep_local, dtype=bool).copy()
        pred_ids = stage_pred_local.astype(np.int64, copy=False)
        for cid, thr in class_accept_ovr_thresholds.items():
            mask = pred_ids == int(cid)
            if not mask.any():
                continue
            scores_all = score_bank.get(int(cid))
            if scores_all is None:
                continue
            keep_out[mask] = keep_out[mask] & (
                scores_all[gate_pos_local[mask]].astype(np.float32, copy=False) >= np.float32(thr)
            )
        return keep_out

    def _apply_class_accept_joint_mask(
        gate_pos_local: np.ndarray,
        stage_pred_local: np.ndarray,
        keep_local: np.ndarray,
        p_all_local: np.ndarray,
        pred_stage_prob_local: np.ndarray | None,
    ) -> np.ndarray:
        if not class_accept_joint_thresholds or pred_stage_prob_local is None:
            return keep_local
        keep_out = np.asarray(keep_local, dtype=bool).copy()
        pred_ids = stage_pred_local.astype(np.int64, copy=False)
        joint_scores = (
            p_all_local[gate_pos_local].astype(np.float32, copy=False)
            * pred_stage_prob_local.astype(np.float32, copy=False)
        ).astype(np.float32, copy=False)
        for cid, thr in class_accept_joint_thresholds.items():
            mask = pred_ids == int(cid)
            if not mask.any():
                continue
            keep_out[mask] = keep_out[mask] & (joint_scores[mask] >= np.float32(thr))
        return keep_out

    def _overwrite_stage_prob_subset(
        stage_prob_local: np.ndarray | None,
        route_mask: np.ndarray,
        class_ids_local: np.ndarray | list[int],
        local_prob: np.ndarray,
    ) -> np.ndarray | None:
        if stage_prob_local is None or not route_mask.any():
            return stage_prob_local
        cids = np.asarray(class_ids_local, dtype=np.int64)
        p_local = np.asarray(local_prob, dtype=np.float32)
        if cids.ndim != 1 or p_local.ndim != 2 or p_local.shape[1] != len(cids):
            return stage_prob_local
        denom = np.maximum(p_local.sum(axis=1, keepdims=True), np.float32(1e-12))
        p_local = (p_local / denom).astype(np.float32, copy=False)
        prob_out = stage_prob_local.astype(np.float32, copy=True)
        row_idx = np.where(route_mask)[0].astype(np.int64, copy=False)
        prob_out[row_idx, :] = np.float32(0.0)
        prob_out[np.ix_(row_idx, cids)] = p_local
        return prob_out

    def _apply_pairwise_stage_refine(
        stage_pred_local: np.ndarray,
        stage_prob_local: np.ndarray | None,
        X_gate_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not pair_refine_pairs:
            return stage_pred_local, stage_prob_local
        stage_out = stage_pred_local.astype(np.int64, copy=True)
        prob_out = None if stage_prob_local is None else stage_prob_local.astype(np.float32, copy=True)
        for (cid_a, cid_b), pair_thr, pair_model in zip(pair_refine_pairs, pair_refine_thresholds, pair_refine_models):
            mask = np.isin(stage_out, [int(cid_a), int(cid_b)])
            if not mask.any():
                continue
            pair_prob = pair_model.predict_proba(X_gate_local[mask])[:, 1].astype(np.float32, copy=False)
            stage_out[mask] = np.where(pair_prob >= float(pair_thr), int(cid_b), int(cid_a)).astype(np.int64, copy=False)
            if prob_out is not None:
                pair_prob_2d = np.stack(
                    [np.float32(1.0) - pair_prob, pair_prob],
                    axis=1,
                ).astype(np.float32, copy=False)
                prob_out = _overwrite_stage_prob_subset(prob_out, mask, [int(cid_a), int(cid_b)], pair_prob_2d)
        return stage_out, prob_out

    def _apply_earlycrow_tail_specialist_refine(
        stage_pred_local: np.ndarray,
        stage_prob_local: np.ndarray | None,
        X_gate_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if cfg.dataset != "earlycrow" or stage_prob_local is None:
            return stage_pred_local, stage_prob_local
        specialist_model = stage2_extra.get("earlycrow_tail_specialist_model", None)
        specialist_class_ids = stage2_extra.get("earlycrow_tail_specialist_class_ids", None)
        specialist_head_ids = stage2_extra.get("earlycrow_tail_specialist_head_ids", None)
        specialist_margin = stage2_extra.get("earlycrow_tail_specialist_margin", None)
        specialist_min_sum = stage2_extra.get("earlycrow_tail_specialist_min_sum", None)
        if (
            specialist_model is None
            or specialist_class_ids is None
            or specialist_head_ids is None
            or specialist_margin is None
            or specialist_min_sum is None
        ):
            return stage_pred_local, stage_prob_local
        cluster_ids = np.asarray(specialist_class_ids, dtype=np.int64)
        head_ids = np.asarray(specialist_head_ids, dtype=np.int64)
        if cluster_ids.ndim != 1 or head_ids.ndim != 1 or cluster_ids.size == 0 or head_ids.size == 0:
            return stage_pred_local, stage_prob_local
        score_local = stage_prob_local.astype(np.float32, copy=False)
        top2 = np.argsort(score_local, axis=1)[:, -2:]
        top1 = top2[:, 1].astype(np.int64, copy=False)
        top2c = top2[:, 0].astype(np.int64, copy=False)
        idx = np.arange(len(top1), dtype=np.int64)
        diff = (score_local[idx, top1] - score_local[idx, top2c]).astype(np.float32, copy=False)
        cluster_sum = score_local[:, cluster_ids].sum(axis=1).astype(np.float32, copy=False)
        head_top1 = np.isin(top1, head_ids, assume_unique=False)
        route = (
            np.isin(top1, cluster_ids, assume_unique=False)
            & (cluster_sum >= np.float32(float(specialist_min_sum)))
            & (
                head_top1
                | np.isin(top2c, cluster_ids, assume_unique=False)
                | (diff <= np.float32(float(specialist_margin)))
            )
        )
        if not route.any():
            return stage_pred_local, stage_prob_local
        stage_out = stage_pred_local.astype(np.int64, copy=True)
        specialist_prob = specialist_model.predict_proba(X_gate_local[route]).astype(np.float32, copy=False)
        choose = specialist_prob.argmax(axis=1).astype(np.int64, copy=False)
        stage_out[np.where(route)[0]] = cluster_ids[choose].astype(np.int64, copy=False)
        prob_out = _overwrite_stage_prob_subset(stage_prob_local, route, cluster_ids, specialist_prob)
        return stage_out, prob_out

    def _apply_cic2024_hard_specialist_refine(
        stage_pred_local: np.ndarray,
        stage_prob_local: np.ndarray | None,
        X_gate_local: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if cfg.dataset != "cic2024" or stage_prob_local is None:
            return stage_pred_local, stage_prob_local
        specialist_model = stage2_extra.get("cic2024_hard_specialist_model", None)
        specialist_class_ids = stage2_extra.get("cic2024_hard_specialist_class_ids", None)
        specialist_head_ids = stage2_extra.get("cic2024_hard_specialist_head_ids", None)
        specialist_margin = stage2_extra.get("cic2024_hard_specialist_margin", None)
        specialist_min_sum = stage2_extra.get("cic2024_hard_specialist_min_sum", None)
        if (
            specialist_model is None
            or specialist_class_ids is None
            or specialist_head_ids is None
            or specialist_margin is None
            or specialist_min_sum is None
        ):
            return stage_pred_local, stage_prob_local
        cluster_ids = np.asarray(specialist_class_ids, dtype=np.int64)
        head_ids = np.asarray(specialist_head_ids, dtype=np.int64)
        if cluster_ids.ndim != 1 or head_ids.ndim != 1 or cluster_ids.size == 0 or head_ids.size == 0:
            return stage_pred_local, stage_prob_local
        score_local = stage_prob_local.astype(np.float32, copy=False)
        top2 = np.argsort(score_local, axis=1)[:, -2:]
        top1 = top2[:, 1].astype(np.int64, copy=False)
        top2c = top2[:, 0].astype(np.int64, copy=False)
        idx = np.arange(len(top1), dtype=np.int64)
        diff = (score_local[idx, top1] - score_local[idx, top2c]).astype(np.float32, copy=False)
        cluster_sum = score_local[:, cluster_ids].sum(axis=1).astype(np.float32, copy=False)
        near_cluster = np.isin(top1, cluster_ids, assume_unique=False) | np.isin(top2c, cluster_ids, assume_unique=False)
        head_top1 = np.isin(top1, head_ids, assume_unique=False)
        route = (
            near_cluster
            & (cluster_sum >= np.float32(float(specialist_min_sum)))
            & (
                head_top1
                | np.isin(top1, cluster_ids, assume_unique=False)
                | (diff <= np.float32(float(specialist_margin)))
            )
        )
        if not route.any():
            return stage_pred_local, stage_prob_local
        stage_out = stage_pred_local.astype(np.int64, copy=True)
        specialist_prob = specialist_model.predict_proba(X_gate_local[route]).astype(np.float32, copy=False)
        choose = specialist_prob.argmax(axis=1).astype(np.int64, copy=False)
        stage_out[np.where(route)[0]] = cluster_ids[choose].astype(np.int64, copy=False)
        prob_out = _overwrite_stage_prob_subset(stage_prob_local, route, cluster_ids, specialist_prob)
        return stage_out, prob_out

    def _apply_earlycrow_tail_ovr_refine(
        stage_pred_local: np.ndarray,
        X_gate_local: np.ndarray,
        p_gate_local: np.ndarray,
    ) -> np.ndarray:
        if cfg.dataset != "earlycrow":
            return stage_pred_local
        tail_models = stage2_extra.get("earlycrow_tail_ovr_models", None)
        tail_thresholds = stage2_extra.get("earlycrow_tail_ovr_thresholds", None)
        tail_class_ids = stage2_extra.get("earlycrow_tail_ovr_class_ids", None)
        if tail_models is None or tail_thresholds is None or tail_class_ids is None:
            return stage_pred_local
        cids = np.asarray(tail_class_ids, dtype=np.int64)
        thrs = np.asarray(tail_thresholds, dtype=np.float32)
        if cids.ndim != 1 or thrs.ndim != 1 or len(cids) != len(thrs) or len(tail_models) != len(cids):
            return stage_pred_local
        stage_out = stage_pred_local.astype(np.int64, copy=True)
        best_scores = np.full(len(stage_out), -1.0, dtype=np.float32)
        candidate_pred_ids = {int(stage_names.index(name)) for name in ["PlugX1", "Sogou", "Zeus"] if name in stage_names}
        if not candidate_pred_ids:
            return stage_out
        base_mask = np.isin(stage_out, np.asarray(sorted(candidate_pred_ids), dtype=np.int64), assume_unique=False)
        base_mask = base_mask | (p_gate_local.astype(np.float32, copy=False) < np.float32(max(float(stage1_threshold), 0.50)))
        if not base_mask.any():
            return stage_out
        for model_ovr, cid, thr in zip(tail_models, cids.tolist(), thrs.tolist()):
            scores = model_ovr.predict_proba(X_gate_local)[:, 1].astype(np.float32, copy=False)
            hit = base_mask & (scores >= np.float32(thr)) & (scores > best_scores + np.float32(1e-6))
            if not hit.any():
                continue
            stage_out[hit] = int(cid)
            best_scores[hit] = scores[hit]
        return stage_out

    def _apply_class_accept_stage1_mask(
        gate_pos_local: np.ndarray,
        stage_pred_local: np.ndarray,
        keep_local: np.ndarray,
        p_all_local: np.ndarray,
    ) -> np.ndarray:
        if class_accept_stage1_min is None:
            return keep_local
        keep_out = keep_local.astype(bool, copy=True)
        for cid, thr in enumerate(class_accept_stage1_min.tolist()):
            thr = float(thr)
            if thr <= 0.0:
                continue
            mask = stage_pred_local == int(cid)
            if not mask.any():
                continue
            keep_out[mask] = keep_out[mask] & (
                p_all_local[gate_pos_local[mask]].astype(np.float32, copy=False) >= np.float32(thr)
            )
        return keep_out

    def _apply_below_gate_ovr_rescue(
        y_pred_all_local: np.ndarray,
        p_all_local: np.ndarray,
        gate_mask_local: np.ndarray,
        score_bank: dict[int, np.ndarray],
        gate_thr: float,
    ) -> np.ndarray:
        if not class_rescue_joint_thresholds:
            return y_pred_all_local
        y_out = np.asarray(y_pred_all_local, dtype=np.int64).copy()
        best_joint = np.full(len(y_out), -1.0, dtype=np.float32)
        below_gate = ~np.asarray(gate_mask_local, dtype=bool)
        for cid, thr in class_rescue_joint_thresholds.items():
            scores_all = score_bank.get(int(cid))
            if scores_all is None:
                continue
            stage1_min = float(class_rescue_stage1_mins.get(int(cid), 0.25))
            joint = (p_all_local.astype(np.float32, copy=False) * scores_all.astype(np.float32, copy=False)).astype(np.float32, copy=False)
            hit = below_gate & (y_out == 0) & (p_all_local >= np.float32(stage1_min)) & (p_all_local < np.float32(gate_thr)) & (joint >= np.float32(thr))
            if not hit.any():
                continue
            update = hit & (joint > best_joint)
            y_out[update] = 1 + int(cid)
            best_joint[update] = joint[update]
        return y_out

    if str(getattr(cfg, "inference_policy", "original")) == "suspicious_unknown":
        tau_b = _select_tau_b(stage1_split.y_val, p_val)
        tau_m = _select_tau_m(stage1_split.y_val, p_val, tau_b)
        tau_b_candidates, tau_m_candidates = _stage1_threshold_candidates(tau_b, tau_m)
        min_tau_b = float(min(tau_b_candidates))
        base_conf = float(stage2_extra.get("stage2_min_conf", 0.0) or 0.0)
        base_ent = float(stage2_extra.get("stage2_entropy_max", 1.01) or 1.01)
        base_margin = float(getattr(cfg, "stage2_margin_min", 0.0) or 0.0)
        conf_candidates = _neighbor_values(base_conf, stage2_min_conf_candidates, extra=[0.0, 0.5, 0.6, 0.7, 0.8])
        ent_candidates = _neighbor_values(base_ent, stage2_entropy_max_candidates, extra=[1.01, 0.95, 0.9, 0.85, 0.8, 0.75])
        margin_candidates = _neighbor_values(base_margin, [0.0, 0.02, 0.05, 0.08, 0.10, 0.15], extra=[base_margin])

        gate_mask_val = p_val >= np.float32(min_tau_b)
        if not gate_mask_val.any():
            raise RuntimeError("No validation samples entered Stage-II under suspicious_unknown policy.")
        gate_pos_val = np.where(gate_mask_val)[0].astype(np.int64)
        X_gate_val = df_task.iloc[idx_val_all[gate_pos_val]][feature_cols].to_numpy(dtype=np.float32, copy=True)
        X_gate_val = split.scaler.transform(X_gate_val).astype(np.float32, copy=False)
        stage_pred_val_gate, stage_prob_val_gate = _predict_multiclass(stage2_model, X_gate_val, stage2_extra)
        lat_force_val, ex_force_val = _force_keep(stage2_extra, X_gate_val)

        suspicious_id = 1 + len(stage_names)
        unknown_id = suspicious_id + 1
        best_choice = None
        search_records: list[dict[str, float]] = []
        for tau_b_cand in tau_b_candidates:
            gate_mask_val_cand = p_val >= np.float32(tau_b_cand)
            if not gate_mask_val_cand.any():
                continue
            gate_pos_val_cand = np.where(gate_mask_val_cand)[0].astype(np.int64)
            stage_pred_gate = stage_pred_val_gate[np.searchsorted(gate_pos_val, gate_pos_val_cand)]
            stage_prob_gate = stage_prob_val_gate[np.searchsorted(gate_pos_val, gate_pos_val_cand)]
            lat_force_gate = None if lat_force_val is None else np.asarray(lat_force_val, dtype=bool)[np.searchsorted(gate_pos_val, gate_pos_val_cand)]
            ex_force_gate = None if ex_force_val is None else np.asarray(ex_force_val, dtype=bool)[np.searchsorted(gate_pos_val, gate_pos_val_cand)]
            tau_m_cands_local = [float(v) for v in tau_m_candidates if float(v) > float(tau_b_cand)]
            if not tau_m_cands_local:
                tau_m_cands_local = [float(min(0.999, max(float(tau_b_cand) + 1e-4, float(tau_b_cand) * 1.05)))]
            for tau_m_cand in tau_m_cands_local:
                for min_conf in conf_candidates:
                    for ent_thr in ent_candidates:
                        for margin_thr in margin_candidates:
                            thr_bundle = {
                                "tau_b": float(tau_b_cand),
                                "tau_m": float(tau_m_cand),
                                "tau_c": float(min_conf),
                                "tau_e": float(ent_thr),
                                "tau_delta": float(margin_thr),
                            }
                            final_val = _assemble_final_outputs(
                                prob_pos=p_val,
                                gate_pos=gate_pos_val_cand,
                                stage_pred=stage_pred_gate,
                                stage_prob=stage_prob_gate,
                                lat_force=lat_force_gate,
                                ex_force=ex_force_gate,
                                stage_names=stage_names,
                                inference_policy="suspicious_unknown",
                                thresholds=thr_bundle,
                            )
                            y_pred_all = np.asarray(final_val["y_pred"], dtype=np.int64)
                            # Closed-set metrics stay on the original label space; unknown outputs remain counted as errors.
                            metric_label_ids = _closed_set_metric_label_ids(stage_names, y_true=y_true_all_val, y_pred=y_pred_all)
                            summary = _macro_weighted_summary(y_true_all_val, y_pred_all, labels=metric_label_ids)
                            mal_total = max(1, int((y_true_all_val != 0).sum()))
                            mal_to_benign = float(((y_true_all_val != 0) & (y_pred_all == 0)).sum() / mal_total)
                            known_to_unknown = float(
                                ((y_true_all_val != 0) & np.isin(y_pred_all, np.asarray([suspicious_id, unknown_id], dtype=np.int64))).sum()
                                / mal_total
                            )
                            release_cov = float(((y_pred_all > 0) & (~np.isin(y_pred_all, np.asarray([suspicious_id, unknown_id], dtype=np.int64)))).sum() / len(y_pred_all))
                            obj = float(summary["f1_macro"] + 0.5 * summary["f1_weighted"] - 2.0 * mal_to_benign - 0.2 * known_to_unknown)
                            rec = {
                                "objective": obj,
                                "tau_b": float(tau_b_cand),
                                "tau_m": float(tau_m_cand),
                                "tau_c": float(min_conf),
                                "tau_e": float(ent_thr),
                                "tau_delta": float(margin_thr),
                                "macro_f1": float(summary["f1_macro"]),
                                "weighted_f1": float(summary["f1_weighted"]),
                                "malicious_to_benign_rate": mal_to_benign,
                                "known_to_unknown_rate": known_to_unknown,
                                "release_coverage": release_cov,
                            }
                            search_records.append(rec)
                            key = (
                                float(obj),
                                -float(mal_to_benign),
                                -float(known_to_unknown),
                                float(summary["f1_macro"]),
                                float(summary["f1_weighted"]),
                                float(release_cov),
                                -float(tau_b_cand),
                                float(tau_m_cand),
                                -float(ent_thr),
                                float(min_conf),
                            )
                            if best_choice is None or key > best_choice[0]:
                                best_choice = (key, rec, final_val)

        if best_choice is None:
            raise RuntimeError("Failed to select thresholds for suspicious_unknown policy.")

        _, best_rec, _ = best_choice
        best_t = float(best_rec["tau_b"])
        best_t_m = float(best_rec["tau_m"])
        best_min_conf = float(best_rec["tau_c"])
        best_ent_thr = float(best_rec["tau_e"])
        best_margin_thr = float(best_rec["tau_delta"])

        stage2_extra = stage2_extra.copy()
        stage2_extra["stage1_threshold"] = float(best_t)
        stage2_extra["stage1_tau_b"] = float(best_t)
        stage2_extra["stage1_tau_m"] = float(best_t_m)
        stage2_extra["stage2_min_conf"] = float(best_min_conf)
        stage2_extra["stage2_entropy_max"] = float(best_ent_thr)
        stage2_extra["stage2_margin_min"] = float(best_margin_thr)
        stage2_extra["inference_policy"] = "suspicious_unknown"

        stage2_pred_val, stage2_prob_val = _predict_multiclass(stage2_model, split.X_val, stage2_extra)
        stage2_metric_val = compute_metrics(
            y_true=split.y_val,
            y_pred=stage2_pred_val,
            y_prob=stage2_prob_val,
            num_classes=len(stage_names),
        )
        stage2_pred_test, stage2_prob_test = _predict_multiclass(stage2_model, split.X_test, stage2_extra)
        stage2_metric_test = compute_metrics(
            y_true=split.y_test,
            y_pred=stage2_pred_test,
            y_prob=stage2_prob_test,
            num_classes=len(stage_names),
        )
        stage2_summary = _macro_weighted_summary(split.y_test, stage2_pred_test)
        stage2_summary["macro_acc"] = _macro_acc_from_cm(stage2_metric_test.cm)
        stage2_summary["macro_auc"], stage2_summary["weighted_auc"] = _auc_macro_weighted_ovr(split.y_test, stage2_prob_test)
        _write_json_silent(
            os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_seed{cfg.seed}_stage2_metrics.json"),
            {
                "dataset": cfg.dataset,
                "seed": int(cfg.seed),
                "drop_stages": str(cfg.drop_stages),
                "labels": stage_names,
                "metrics": stage2_summary,
                "confusion_matrix": stage2_metric_test.cm.tolist(),
            },
        )

        idx_test_all = stage1_split.idx_test.astype(np.int64, copy=False)
        gate_mask_test = p_test >= np.float32(best_t)
        y_true_e2e = np.zeros(len(idx_test_all), dtype=np.int64)
        if use_activity:
            rows_test = df_task.iloc[idx_test_all][["Stage", "Activity"]]
            stage_true_test = rows_test["Stage"].astype(str).to_numpy()
            act_true_test = rows_test["Activity"].astype(str).to_numpy()
            act_true_test = np.where(stage_true_test == "Benign", "Normal", act_true_test).astype(str)
            act_true_test = np.where(act_true_test == "Normal", "Other", act_true_test).astype(str)
            if int(getattr(cfg, "min_class_count", 1)) > 1:
                counts = pd.Series(stage).value_counts()
                rare_set = set(counts[counts < int(getattr(cfg, "min_class_count", 1))].index.tolist())
                act_true_test = np.array(["Other" if a in rare_set else a for a in act_true_test], dtype=object)
            true_label_text = act_true_test.astype(object, copy=False)
            for i, st in enumerate(stage_true_test.tolist()):
                if st != "Benign":
                    y_true_e2e[i] = 1 + int(stage_to_id.get(str(act_true_test[i]), 0))
                else:
                    true_label_text[i] = "Benign"
        else:
            true_label_text = df_task.iloc[idx_test_all]["Stage"].astype(str).to_numpy()
            for i, s in enumerate(true_label_text):
                if s != "Benign":
                    y_true_e2e[i] = 1 + int(stage_to_id[s])

        gate_pos_test = np.empty(0, dtype=np.int64)
        stage_pred_test_gate = None
        stage_prob_test_gate = None
        lat_force_test = None
        ex_force_test = None
        if gate_mask_test.any():
            gate_pos_test = np.where(gate_mask_test)[0].astype(np.int64)
            X_gate_test = df_task.iloc[idx_test_all[gate_pos_test]][feature_cols].to_numpy(dtype=np.float32, copy=True)
            X_gate_test = split.scaler.transform(X_gate_test).astype(np.float32, copy=False)
            stage_pred_test_gate, stage_prob_test_gate = _predict_multiclass(stage2_model, X_gate_test, stage2_extra)
            lat_force_test, ex_force_test = _force_keep(stage2_extra, X_gate_test)

        final_test = _assemble_final_outputs(
            prob_pos=p_test,
            gate_pos=gate_pos_test,
            stage_pred=stage_pred_test_gate,
            stage_prob=stage_prob_test_gate,
            lat_force=lat_force_test,
            ex_force=ex_force_test,
            stage_names=stage_names,
            inference_policy="suspicious_unknown",
            thresholds={
                "tau_b": float(best_t),
                "tau_m": float(best_t_m),
                "tau_c": float(best_min_conf),
                "tau_e": float(best_ent_thr),
                "tau_delta": float(best_margin_thr),
            },
        )
        y_pred_e2e = np.asarray(final_test["y_pred"], dtype=np.int64)
        labels_all = list(final_test["labels_all"])
        metric_e2e = compute_metrics(y_true=y_true_e2e, y_pred=y_pred_e2e, y_prob=None, num_classes=len(labels_all))
        metric_label_ids = _closed_set_metric_label_ids(stage_names, y_true=y_true_e2e, y_pred=y_pred_e2e)
        e2e = _macro_weighted_summary(y_true_e2e, y_pred_e2e, labels=metric_label_ids)
        e2e["macro_acc"] = _macro_acc_from_cm(metric_e2e.cm)
        e2e["fpr"] = _benign_fpr_from_cm(metric_e2e.cm)
        e2e["macro_fpr"], e2e["weighted_fpr"] = _fpr_macro_weighted_from_cm(metric_e2e.cm)
        prob_full_auc = _build_closed_set_auc_prob(
            prob_pos=p_test,
            gate_pos=gate_pos_test,
            stage_prob=stage_prob_test_gate,
            labels_all_closed=["Benign"] + stage_names,
        )
        e2e["macro_auc"], e2e["weighted_auc"] = _auc_macro_weighted_ovr(y_true_e2e, prob_full_auc)
        e2e["malicious_to_benign_rate"] = float(((y_true_e2e != 0) & (y_pred_e2e == 0)).sum() / max(1, int((y_true_e2e != 0).sum())))
        e2e["known_to_unknown_rate"] = float(
            ((y_true_e2e != 0) & np.isin(y_pred_e2e, np.asarray([suspicious_id, unknown_id], dtype=np.int64))).sum()
            / max(1, int((y_true_e2e != 0).sum()))
        )
        e2e["selection_objective"] = float(best_rec["objective"])
        print("End2End Test:", e2e)
        print(format_confusion_matrix(metric_e2e.cm, labels=labels_all))

        _write_json_silent(
            os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_seed{cfg.seed}_end2end_metrics.json"),
            {
                "dataset": cfg.dataset,
                "seed": int(cfg.seed),
                "drop_stages": str(cfg.drop_stages),
                "inference_policy": "suspicious_unknown",
                "labels": labels_all,
                "metric_labels": [(["Benign"] + stage_names)[i] for i in metric_label_ids],
                "thresholds": {
                    "tau_b": float(best_t),
                    "tau_m": float(best_t_m),
                    "tau_c": float(best_min_conf),
                    "tau_e": float(best_ent_thr),
                    "tau_delta": float(best_margin_thr),
                },
                "metrics": e2e,
                "confusion_matrix": metric_e2e.cm.tolist(),
            },
        )
        _write_json_silent(
            os.path.join(cfg.artifacts_dir, "thresholds", f"{cfg.dataset}_{cfg.seed}.json"),
            {
                "dataset": cfg.dataset,
                "seed": int(cfg.seed),
                "inference_policy": "suspicious_unknown",
                "selected": {
                    "tau_b": float(best_t),
                    "tau_m": float(best_t_m),
                    "tau_c": float(best_min_conf),
                    "tau_e": float(best_ent_thr),
                    "tau_delta": float(best_margin_thr),
                },
                "validation_search": search_records,
            },
        )
        _write_prediction_trace(
            out_path=os.path.join(cfg.artifacts_dir, "predictions", f"{cfg.dataset}_seed{cfg.seed}_closed_set.csv"),
            df_rows=df_task.iloc[idx_test_all],
            cfg=cfg,
            true_label=true_label_text,
            stage1_score=p_test,
            final_outputs=final_test,
            gateway_id="centralized",
        )
        _print_and_collect_errors(
            df_task=df_task,
            idx_test=idx_test_all,
            y_true=y_true_e2e,
            y_pred=y_pred_e2e,
            label_names=labels_all,
            max_print=cfg.max_print_errors,
            do_print=False,
            phase="end2end_test",
            out_path=errors_out_path,
            cfg=cfg,
        )
        ensure_dir(cfg.checkpoint_dir)
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_cascade_feedback_seed{cfg.seed}_best.pt")
        _save_checkpoint(
            model=stage2_model,
            cfg=cfg,
            split=split,
            class_names=stage_names,
            best_val=stage2_metric_val,
            out_path=ckpt_path,
            extra={
                "stage1_threshold": float(best_t),
                "stage1_tau_b": float(best_t),
                "stage1_tau_m": float(best_t_m),
                "stage2_min_conf": float(best_min_conf),
                "stage2_entropy_max": float(best_ent_thr),
                "stage2_margin_min": float(best_margin_thr),
                "cascade_objective": cfg.cascade_objective,
                "inference_policy": "suspicious_unknown",
            } | (stage2_extra or {}),
        )
        return

    dapt_relaxed_gate_search = bool(cfg.dataset == "dapt2020" and _use_federated_dask_backend(cfg))
    dapt_stage1_floor = None
    dapt_stage1_cap = None
    cic_stage1_floor = None
    cic_stage1_cap = None
    if cfg.dataset == "dapt2020":
        # Keep the end-to-end gate near the Stage-I operating point. For dapt2020,
        # raising the final gate above the Stage-I validation threshold mostly turns
        # tail malicious classes back into Benign without bringing meaningful gains.
        floor_margin = 0.2 if (dapt_relaxed_gate_search or class_accept_ovr_thresholds) else 0.02
        dapt_stage1_floor = min(
            float(stage1_threshold),
            max(float(cfg.cascade_threshold_min), float(stage1_threshold) - float(floor_margin)),
        )
        dapt_stage1_cap = float(stage1_threshold)
    elif cfg.dataset == "cic2024":
        # CIC2024 is sensitive to over-conservative end-to-end gating: once the final
        # threshold drifts above the Stage-I operating point, Exploits/Analysis flows
        # are pushed back to Benign faster than Stage-II quality improves.
        cic_stage1_floor = min(
            float(stage1_threshold),
            max(float(cfg.cascade_threshold_min), float(stage1_threshold) - 0.20),
        )
        cic_stage1_cap = float(stage1_threshold)

    for t in thresholds:
        if (
            str(getattr(cfg, "stage1_gate_method", "threshold")) == "conformal_fpr"
            and float(t) + 1e-12 < float(stage1_threshold)
            and not (cfg.dataset == "dapt2020" and class_accept_ovr_thresholds)
        ):
            continue
        if dapt_stage1_floor is not None and float(t) + 1e-12 < float(dapt_stage1_floor):
            continue
        if dapt_stage1_cap is not None and float(t) - 1e-12 > float(dapt_stage1_cap):
            continue
        if cic_stage1_floor is not None and float(t) + 1e-12 < float(cic_stage1_floor):
            continue
        if cic_stage1_cap is not None and float(t) - 1e-12 > float(cic_stage1_cap):
            continue
        row_id_train_gate = stage1_split.row_id_train[p_train >= t]
        row_id_val_gate = stage1_split.row_id_val[p_val >= t]

        gate_mask_val = p_val >= t
        if not gate_mask_val.any():
            continue
        gate_pos = np.where(gate_mask_val)[0].astype(np.int64)
        X_gate = df_task.iloc[idx_val_all[gate_pos]][feature_cols].to_numpy(dtype=np.float32, copy=True)
        X_gate = split.scaler.transform(X_gate).astype(np.float32, copy=False)

        stage_pred, stage_prob = _predict_multiclass(stage2_model, X_gate, stage2_extra)
        stage_pred, stage_prob = _apply_earlycrow_tail_specialist_refine(stage_pred, stage_prob, X_gate)
        has_cic2024_head_branch = bool(cfg.dataset == "cic2024" and stage2_extra.get("cic2024_head_cluster_model") is not None)
        if not has_cic2024_head_branch:
            stage_pred, stage_prob = _apply_cic2024_hard_specialist_refine(stage_pred, stage_prob, X_gate)
            stage_pred, stage_prob = _apply_pairwise_stage_refine(stage_pred, stage_prob, X_gate)
        stage_pred = _apply_earlycrow_tail_ovr_refine(stage_pred, X_gate, p_val[gate_pos].astype(np.float32, copy=False))
        lat_force, ex_force = _force_keep(stage2_extra, X_gate)
        pred_stage_prob = None
        if stage_prob is not None:
            pred_stage_prob = stage_prob[np.arange(len(stage_pred)), stage_pred.astype(np.int64, copy=False)].astype(np.float32, copy=False)
        stage1_gate_prob = p_val[gate_pos].astype(np.float32, copy=False)

        best_local = None
        ent = _norm_entropy(stage_prob) if stage_prob is not None else None
        for min_conf in stage2_min_conf_candidates:
            for ent_thr in stage2_entropy_max_candidates:
                for margin_thr in stage2_margin_candidates:
                    for joint_thr in stage2_joint_candidates:
                        keep = np.ones(len(stage_pred), dtype=bool)
                        if stage_prob is not None and float(min_conf) > 0.0:
                            keep = stage_prob.max(axis=1) >= float(min_conf)
                        if pred_stage_prob is not None and class_accept_min_prob is not None:
                            keep = keep & (
                                pred_stage_prob >= class_accept_min_prob[stage_pred.astype(np.int64, copy=False)]
                            )
                        keep = _apply_class_accept_stage1_mask(
                            gate_pos_local=gate_pos,
                            stage_pred_local=stage_pred,
                            keep_local=keep,
                            p_all_local=p_val,
                        )
                        keep = _apply_class_accept_ovr_mask(gate_pos, stage_pred, keep, class_accept_ovr_scores_val)
                        keep = _apply_class_accept_joint_mask(
                            gate_pos_local=gate_pos,
                            stage_pred_local=stage_pred,
                            keep_local=keep,
                            p_all_local=p_val,
                            pred_stage_prob_local=pred_stage_prob,
                        )
                        if pred_stage_prob is not None and float(joint_thr) > 0.0:
                            keep = keep & ((stage1_gate_prob * pred_stage_prob) >= float(joint_thr))
                        if ent is not None and float(ent_thr) < 1.0:
                            keep = keep & (ent <= float(ent_thr))
                        if stage_prob is not None and float(margin_thr) > 0.0:
                            keep = keep & (_top2_margin(stage_prob) >= float(margin_thr))
                        if lat_force is not None:
                            keep = keep | lat_force
                        if ex_force is not None:
                            keep = keep | ex_force

                        y_pred_all = np.zeros(len(idx_val_all), dtype=np.int64)
                        if keep.any():
                            y_pred_all[gate_pos[keep]] = 1 + stage_pred[keep].astype(np.int64, copy=False)
                        y_pred_all = _apply_below_gate_ovr_rescue(
                            y_pred_all_local=y_pred_all,
                            p_all_local=p_val,
                            gate_mask_local=gate_mask_val,
                            score_bank=ovr_full_scores_val,
                            gate_thr=float(t),
                        )
                        benign_fp = int(((y_true_all_val == 0) & (y_pred_all != 0)).sum())
                        malicious_fn = int(((y_true_all_val != 0) & (y_pred_all == 0)).sum())

                        end2end_val = compute_metrics(
                            y_true=y_true_all_val,
                            y_pred=y_pred_all,
                            y_prob=None,
                            num_classes=1 + len(stage_names),
                        )
                        metric_label_ids = _closed_set_metric_label_ids(stage_names, y_true=y_true_all_val)
                        summary_val = _macro_weighted_summary(y_true_all_val, y_pred_all, labels=metric_label_ids)
                        obj = float(summary_val["f1_macro"])
                        if cfg.cascade_objective == "min":
                            obj = float(
                                min(
                                    float(summary_val["precision_macro"]),
                                    float(summary_val["recall_macro"]),
                                    float(summary_val["f1_macro"]),
                                )
                            )
                        if best_local is None:
                            best_local = (
                                obj,
                                float(min_conf),
                                float(ent_thr),
                                float(margin_thr),
                                float(joint_thr),
                                end2end_val,
                                int(benign_fp),
                                int(malicious_fn),
                            )
                        else:
                            cur = (
                                float(best_local[0]),
                                -int(best_local[6]),
                                -int(best_local[7]),
                                float(best_local[3]),
                                float(best_local[4]),
                                float(best_local[1]),
                                -float(best_local[2]),
                            )
                            cand = (
                                float(obj),
                                -int(benign_fp),
                                -int(malicious_fn),
                                float(margin_thr),
                                float(joint_thr),
                                float(min_conf),
                                -float(ent_thr),
                            )
                            if cand > cur:
                                best_local = (
                                    obj,
                                    float(min_conf),
                                    float(ent_thr),
                                    float(margin_thr),
                                    float(joint_thr),
                                    end2end_val,
                                    int(benign_fp),
                                    int(malicious_fn),
                                )

        if best_local is None:
            continue
        obj, best_min_conf, best_ent_thr, best_margin_thr, best_joint_thr, end2end_val, benign_fp, malicious_fn = best_local
        cic_head_absorb = 0
        if cfg.dataset == "cic2024":
            try:
                cm_val_e2e = np.asarray(end2end_val.cm, dtype=np.int64)
                if cm_val_e2e.ndim == 2 and cm_val_e2e.shape[0] >= 7 and cm_val_e2e.shape[1] >= 7:
                    # End-to-end labels: 0=Benign, 1=Analysis, 2=Backdoor, 3=DoS,
                    # 4=Exploits, 5=Generic, 6=Reconnaissance, 7=Shellcode, 8=Worms.
                    cic_head_absorb = int(cm_val_e2e[3, 4] + cm_val_e2e[6, 4] + cm_val_e2e[4, 3] + cm_val_e2e[4, 6])
            except Exception:
                cic_head_absorb = 0
        records.append(
            (
                obj,
                float(t),
                float(best_min_conf),
                float(best_ent_thr),
                float(best_margin_thr),
                float(best_joint_thr),
                stage2_best_val,
                end2end_val,
                int(len(row_id_train_gate)),
                int(len(row_id_val_gate)),
                int(benign_fp),
                int(malicious_fn),
                int(cic_head_absorb),
            )
        )

    if len(records) == 0:
        fallback_t = float(stage1_threshold)
        if dapt_stage1_floor is not None:
            fallback_t = max(float(dapt_stage1_floor), fallback_t)
        if dapt_stage1_cap is not None:
            fallback_t = min(float(dapt_stage1_cap), fallback_t)
        if cic_stage1_floor is not None:
            fallback_t = max(float(cic_stage1_floor), fallback_t)
        if cic_stage1_cap is not None:
            fallback_t = min(float(cic_stage1_cap), fallback_t)
        if len(p_val) > 0 and not bool((p_val >= np.float32(fallback_t)).any()):
            fallback_t = float(np.max(p_val))

        gate_mask_val = p_val >= np.float32(fallback_t)
        if not gate_mask_val.any():
            raise RuntimeError("No valid cascade threshold found for feedback search.")

        gate_pos = np.where(gate_mask_val)[0].astype(np.int64)
        X_gate = df_task.iloc[idx_val_all[gate_pos]][feature_cols].to_numpy(dtype=np.float32, copy=True)
        X_gate = split.scaler.transform(X_gate).astype(np.float32, copy=False)

        stage_pred, stage_prob = _predict_multiclass(stage2_model, X_gate, stage2_extra)
        stage_pred, stage_prob = _apply_earlycrow_tail_specialist_refine(stage_pred, stage_prob, X_gate)
        has_cic2024_head_branch = bool(cfg.dataset == "cic2024" and stage2_extra.get("cic2024_head_cluster_model") is not None)
        if not has_cic2024_head_branch:
            stage_pred, stage_prob = _apply_cic2024_hard_specialist_refine(stage_pred, stage_prob, X_gate)
            stage_pred, stage_prob = _apply_pairwise_stage_refine(stage_pred, stage_prob, X_gate)
        stage_pred = _apply_earlycrow_tail_ovr_refine(stage_pred, X_gate, p_val[gate_pos].astype(np.float32, copy=False))
        lat_force, ex_force = _force_keep(stage2_extra, X_gate)
        pred_stage_prob = None
        if stage_prob is not None:
            pred_stage_prob = stage_prob[np.arange(len(stage_pred)), stage_pred.astype(np.int64, copy=False)].astype(np.float32, copy=False)
        stage1_gate_prob = p_val[gate_pos].astype(np.float32, copy=False)
        keep = np.ones(len(stage_pred), dtype=bool)
        fallback_min_conf = float(stage2_extra.get("stage2_min_conf", stage2_min_conf_candidates[0] if stage2_min_conf_candidates else 0.0) or 0.0)
        fallback_ent_thr = float(stage2_extra.get("stage2_entropy_max", stage2_entropy_max_candidates[0] if stage2_entropy_max_candidates else 1.01) or 1.01)
        fallback_margin_thr = float(stage2_extra.get("stage2_margin_min", stage2_margin_candidates[0] if stage2_margin_candidates else 0.0) or 0.0)
        fallback_joint_thr = float(stage2_extra.get("stage2_joint_min", stage2_joint_candidates[0] if stage2_joint_candidates else 0.0) or 0.0)
        ent = _norm_entropy(stage_prob) if stage_prob is not None else None
        if stage_prob is not None and float(fallback_min_conf) > 0.0:
            keep = stage_prob.max(axis=1) >= float(fallback_min_conf)
        if pred_stage_prob is not None and class_accept_min_prob is not None:
            keep = keep & (
                pred_stage_prob >= class_accept_min_prob[stage_pred.astype(np.int64, copy=False)]
            )
        keep = _apply_class_accept_stage1_mask(
            gate_pos_local=gate_pos,
            stage_pred_local=stage_pred,
            keep_local=keep,
            p_all_local=p_val,
        )
        keep = _apply_class_accept_ovr_mask(gate_pos, stage_pred, keep, class_accept_ovr_scores_val)
        keep = _apply_class_accept_joint_mask(
            gate_pos_local=gate_pos,
            stage_pred_local=stage_pred,
            keep_local=keep,
            p_all_local=p_val,
            pred_stage_prob_local=pred_stage_prob,
        )
        if pred_stage_prob is not None and float(fallback_joint_thr) > 0.0:
            keep = keep & ((stage1_gate_prob * pred_stage_prob) >= float(fallback_joint_thr))
        if ent is not None and float(fallback_ent_thr) < 1.0:
            keep = keep & (ent <= float(fallback_ent_thr))
        if stage_prob is not None and float(fallback_margin_thr) > 0.0:
            keep = keep & (_top2_margin(stage_prob) >= float(fallback_margin_thr))
        if lat_force is not None:
            keep = keep | lat_force
        if ex_force is not None:
            keep = keep | ex_force

        y_pred_all = np.zeros(len(idx_val_all), dtype=np.int64)
        if keep.any():
            y_pred_all[gate_pos[keep]] = 1 + stage_pred[keep].astype(np.int64, copy=False)
        y_pred_all = _apply_below_gate_ovr_rescue(
            y_pred_all_local=y_pred_all,
            p_all_local=p_val,
            gate_mask_local=gate_mask_val,
            score_bank=ovr_full_scores_val,
            gate_thr=float(fallback_t),
        )
        benign_fp = int(((y_true_all_val == 0) & (y_pred_all != 0)).sum())
        malicious_fn = int(((y_true_all_val != 0) & (y_pred_all == 0)).sum())
        end2end_val = compute_metrics(
            y_true=y_true_all_val,
            y_pred=y_pred_all,
            y_prob=None,
            num_classes=1 + len(stage_names),
        )
        metric_label_ids = _closed_set_metric_label_ids(stage_names, y_true=y_true_all_val)
        summary_val = _macro_weighted_summary(y_true_all_val, y_pred_all, labels=metric_label_ids)
        obj = float(summary_val["f1_macro"])
        if cfg.cascade_objective == "min":
            obj = float(
                min(
                    float(summary_val["precision_macro"]),
                    float(summary_val["recall_macro"]),
                    float(summary_val["f1_macro"]),
                )
            )
        cic_head_absorb = 0
        if cfg.dataset == "cic2024":
            try:
                cm_val_e2e = np.asarray(end2end_val.cm, dtype=np.int64)
                if cm_val_e2e.ndim == 2 and cm_val_e2e.shape[0] >= 7 and cm_val_e2e.shape[1] >= 7:
                    cic_head_absorb = int(cm_val_e2e[3, 4] + cm_val_e2e[6, 4] + cm_val_e2e[4, 3] + cm_val_e2e[4, 6])
            except Exception:
                cic_head_absorb = 0
        records.append(
            (
                obj,
                float(fallback_t),
                float(fallback_min_conf),
                float(fallback_ent_thr),
                float(fallback_margin_thr),
                float(fallback_joint_thr),
                stage2_best_val,
                end2end_val,
                int((p_train >= np.float32(fallback_t)).sum()),
                int(gate_mask_val.sum()),
                int(benign_fp),
                int(malicious_fn),
                int(cic_head_absorb),
            )
        )

    records.sort(key=lambda x: x[0], reverse=True)
    best_obj = float(records[0][0])
    margin = 0.01
    if cfg.dataset == "earlycrow":
        margin = 0.03
    eligible = [r for r in records if float(r[0]) >= best_obj - margin]
    if cfg.dataset == "earlycrow":
        eligible_balanced = [r for r in eligible if float(r[1]) >= 0.12]
        if eligible_balanced:
            eligible = eligible_balanced
    if cfg.dataset == "dapt2020":
        eligible_hi = [r for r in records if float(r[0]) >= 0.96]
        if len(eligible_hi) > 0:
            eligible = eligible_hi
        if dapt_relaxed_gate_search:
            chosen = min(eligible, key=lambda r: (int(r[11]), int(r[10]), float(r[1]), -float(r[0]), -float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        else:
            chosen = min(eligible, key=lambda r: (int(r[10]), int(r[11]), -float(r[1]), -float(r[2]), float(r[3]), float(r[4]), float(r[5]), -float(r[0])))
    elif cfg.dataset == "earlycrow":
        chosen = min(
            eligible,
            key=lambda r: (
                int(r[10]) + int(r[11]),
                int(r[11]),
                int(r[10]),
                -float(r[0]),
                float(r[1]),
                float(r[2]),
                float(r[3]),
                float(r[4]),
                float(r[5]),
            ),
        )
    elif cfg.dataset == "cic2024":
        chosen = min(
            eligible,
            key=lambda r: (
                int(r[11]),
                int(r[12]),
                int(r[10]),
                -float(r[0]),
                float(r[1]),
                float(r[2]),
                float(r[3]),
                float(r[4]),
                float(r[5]),
            ),
        )
    else:
        chosen = max(eligible, key=lambda r: (float(r[1]), float(r[2]), -float(r[3]), -float(r[4]), -float(r[5]), float(r[0])))
    best_obj, best_t, best_min_conf, best_ent_thr, best_margin_thr, best_joint_thr = (
        float(chosen[0]),
        float(chosen[1]),
        float(chosen[2]),
        float(chosen[3]),
        float(chosen[4]),
        float(chosen[5]),
    )
    topk = records[: min(6, len(records))]

    stage2_extra = stage2_extra.copy()
    stage2_extra["stage1_threshold"] = float(best_t)
    stage2_extra["stage2_min_conf"] = float(best_min_conf)
    stage2_extra["stage2_entropy_max"] = float(best_ent_thr)
    stage2_extra["stage2_margin_min"] = float(best_margin_thr)
    stage2_extra["stage2_joint_min"] = float(best_joint_thr)
    if cic2024_head_branch_forced_ablation:
        stage2_extra["cic2024_head_branch_forced_ablation"] = True

    stage2_pred_val, stage2_prob_val = _predict_multiclass(stage2_model, split.X_val, stage2_extra)
    stage2_metric_val = compute_metrics(
        y_true=split.y_val,
        y_pred=stage2_pred_val,
        y_prob=stage2_prob_val,
        num_classes=len(stage_names),
    )
    stage2_pred_test, stage2_prob_test = _predict_multiclass(stage2_model, split.X_test, stage2_extra)
    stage2_metric_test = compute_metrics(
        y_true=split.y_test,
        y_pred=stage2_pred_test,
        y_prob=stage2_prob_test,
        num_classes=len(stage_names),
    )

    if cfg.verbose:
        print("Stage2 Val:", _macro_weighted_summary(split.y_val, stage2_pred_val))
        print(format_confusion_matrix(stage2_metric_val.cm, labels=stage_names))

        print("Stage2 Test:", _macro_weighted_summary(split.y_test, stage2_pred_test))
        print(format_confusion_matrix(stage2_metric_test.cm, labels=stage_names))
    stage2_summary = _macro_weighted_summary(split.y_test, stage2_pred_test)
    stage2_summary["macro_acc"] = _macro_acc_from_cm(stage2_metric_test.cm)
    stage2_summary["macro_auc"], stage2_summary["weighted_auc"] = _auc_macro_weighted_ovr(split.y_test, stage2_prob_test)
    _write_json_silent(
        os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_seed{cfg.seed}_stage2_metrics.json"),
        {
            "dataset": cfg.dataset,
            "seed": int(cfg.seed),
            "drop_stages": str(cfg.drop_stages),
            "labels": stage_names,
            "metrics": stage2_summary,
            "confusion_matrix": stage2_metric_test.cm.tolist(),
        },
    )
    _print_and_collect_errors(
        df_task=malicious,
        idx_test=split.idx_test,
        y_true=split.y_test,
        y_pred=stage2_pred_test,
        label_names=stage_names,
        max_print=cfg.max_print_errors,
        do_print=False,
        phase="stage2_test",
        out_path=errors_out_path,
        cfg=cfg,
    )

    gate_mask_test = p_test >= best_t
    y_true_e2e = y_true_all_test.copy()

    y_pred_e2e = np.zeros(len(idx_test_all), dtype=np.int64)
    if gate_mask_test.any():
        gate_pos_test = np.where(gate_mask_test)[0].astype(np.int64)
        X_gate = df_task.iloc[idx_test_all[gate_pos_test]][feature_cols].to_numpy(dtype=np.float32, copy=True)
        X_gate = split.scaler.transform(X_gate).astype(np.float32, copy=False)
        stage_pred, stage_prob = _predict_multiclass(stage2_model, X_gate, stage2_extra)
        stage_pred, stage_prob = _apply_earlycrow_tail_specialist_refine(stage_pred, stage_prob, X_gate)
        has_cic2024_head_branch = bool(cfg.dataset == "cic2024" and stage2_extra.get("cic2024_head_cluster_model") is not None)
        if not has_cic2024_head_branch:
            stage_pred, stage_prob = _apply_cic2024_hard_specialist_refine(stage_pred, stage_prob, X_gate)
            stage_pred, stage_prob = _apply_pairwise_stage_refine(stage_pred, stage_prob, X_gate)
        stage_pred = _apply_earlycrow_tail_ovr_refine(stage_pred, X_gate, p_test[gate_pos_test].astype(np.float32, copy=False))
        lat_force, ex_force = _force_keep(stage2_extra, X_gate)
        pred_stage_prob = None
        if stage_prob is not None:
            pred_stage_prob = stage_prob[np.arange(len(stage_pred)), stage_pred.astype(np.int64, copy=False)].astype(np.float32, copy=False)
        stage1_gate_prob = p_test[gate_pos_test].astype(np.float32, copy=False)

        keep = np.ones(len(stage_pred), dtype=bool)
        if stage_prob is not None and best_min_conf > 0.0:
            keep = keep & (stage_prob.max(axis=1) >= float(best_min_conf))
        if pred_stage_prob is not None and class_accept_min_prob is not None:
            keep = keep & (pred_stage_prob >= class_accept_min_prob[stage_pred.astype(np.int64, copy=False)])
        keep = _apply_class_accept_stage1_mask(
            gate_pos_local=gate_pos_test,
            stage_pred_local=stage_pred,
            keep_local=keep,
            p_all_local=p_test,
        )
        keep = _apply_class_accept_ovr_mask(gate_pos_test, stage_pred, keep, class_accept_ovr_scores_test)
        keep = _apply_class_accept_joint_mask(
            gate_pos_local=gate_pos_test,
            stage_pred_local=stage_pred,
            keep_local=keep,
            p_all_local=p_test,
            pred_stage_prob_local=pred_stage_prob,
        )
        if pred_stage_prob is not None and float(best_joint_thr) > 0.0:
            keep = keep & ((stage1_gate_prob * pred_stage_prob) >= float(best_joint_thr))
        if stage_prob is not None and float(best_ent_thr) < 1.0:
            keep = keep & (_norm_entropy(stage_prob) <= float(best_ent_thr))
        if stage_prob is not None and float(best_margin_thr) > 0.0:
            keep = keep & (_top2_margin(stage_prob) >= float(best_margin_thr))
        if lat_force is not None:
            keep = keep | lat_force
        if ex_force is not None:
            keep = keep | ex_force
        if keep.any():
            y_pred_e2e[gate_pos_test[keep]] = 1 + stage_pred[keep].astype(np.int64, copy=False)
    y_pred_e2e = _apply_below_gate_ovr_rescue(
        y_pred_all_local=y_pred_e2e,
        p_all_local=p_test,
        gate_mask_local=gate_mask_test,
        score_bank=ovr_full_scores_test,
        gate_thr=float(best_t),
    )

    labels_all = ["Benign"] + stage_names
    metric_e2e = compute_metrics(y_true=y_true_e2e, y_pred=y_pred_e2e, y_prob=None, num_classes=len(labels_all))
    metric_label_ids = _closed_set_metric_label_ids(stage_names, y_true=y_true_e2e)
    e2e = _macro_weighted_summary(y_true_e2e, y_pred_e2e, labels=metric_label_ids)
    e2e["macro_acc"] = _macro_acc_from_cm(metric_e2e.cm)
    e2e["fpr"] = _benign_fpr_from_cm(metric_e2e.cm)
    e2e["macro_fpr"], e2e["weighted_fpr"] = _fpr_macro_weighted_from_cm(metric_e2e.cm)
    prob_full = None
    if gate_mask_test.any() and "stage_prob" in locals() and stage_prob is not None:
        prob_full = np.zeros((len(idx_test_all), len(labels_all)), dtype=np.float32)
        prob_full[:, 0] = 1.0
        if "keep" in locals() and keep.any():
            pos = gate_pos_test[keep].astype(np.int64, copy=False)
            p_mal = p_test[pos].astype(np.float32, copy=False)
            mal_prob = stage_prob[keep].astype(np.float32, copy=False)
            denom = mal_prob.sum(axis=1, keepdims=True)
            mal_prob = mal_prob / np.maximum(denom, 1e-12)
            prob_full[pos, 0] = 1.0 - p_mal
            prob_full[pos, 1:] = mal_prob * p_mal.reshape(-1, 1)
    if prob_full is not None:
        e2e["macro_auc"], e2e["weighted_auc"] = _auc_macro_weighted_ovr(y_true_e2e, prob_full)
    else:
        e2e["macro_auc"], e2e["weighted_auc"] = float("nan"), float("nan")
    print("End2End Test:", e2e)
    print(format_confusion_matrix(metric_e2e.cm, labels=labels_all))
    _write_json_silent(
        os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_seed{cfg.seed}_end2end_metrics.json"),
        {
            "dataset": cfg.dataset,
            "seed": int(cfg.seed),
            "drop_stages": str(cfg.drop_stages),
            "labels": labels_all,
            "metrics": e2e,
            "confusion_matrix": metric_e2e.cm.tolist(),
        },
    )
    _print_and_collect_errors(
        df_task=df_task,
        idx_test=idx_test_all,
        y_true=y_true_e2e,
        y_pred=y_pred_e2e,
        label_names=labels_all,
        max_print=cfg.max_print_errors,
        do_print=False,
        phase="end2end_test",
        out_path=errors_out_path,
        cfg=cfg,
    )

    ensure_dir(cfg.checkpoint_dir)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_cascade_feedback_seed{cfg.seed}_best.pt")
    _save_checkpoint(
        model=stage2_model,
        cfg=cfg,
        split=split,
        class_names=stage_names,
        best_val=stage2_metric_val,
        out_path=ckpt_path,
        extra={"stage1_threshold": float(best_t), "cascade_objective": cfg.cascade_objective} | (stage2_extra or {}),
    )


def parse_args() -> tuple[ExperimentConfig, argparse.Namespace]:
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default=None)
    p.add_argument("--replay", action="store_true", default=False)
    p.add_argument("--stage1_ckpt", type=str, default=None)
    p.add_argument("--stage2_ckpt", type=str, default=None)
    p.add_argument("--cic_stage2_with_benign", action="store_true", default=False)
    p.add_argument("--cic_benign_return_min_conf", type=float, default=0.9)
    p.add_argument("--allow_context_leakage", action="store_true", default=False)
    p.add_argument("--split_path", type=str, default=None)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--split_mode", type=str, default="random", choices=["random", "flowkey"])
    p.add_argument("--feature_subset_path", type=str, default=None)
    p.add_argument("--force_moe", action="store_true", default=False)

    p.add_argument("--dataset", type=str, default=argparse.SUPPRESS)
    p.add_argument("--model_type", type=str, default=argparse.SUPPRESS)
    p.add_argument("--malicious_method", type=str, default=argparse.SUPPRESS)
    p.add_argument("--inference_policy", type=str, default=argparse.SUPPRESS, choices=["original", "suspicious_unknown"])
    p.add_argument("--data_dir", type=str, default=argparse.SUPPRESS)
    p.add_argument("--task", type=str, default=argparse.SUPPRESS, choices=["all", "stage1"])
    p.add_argument("--stage2_label", type=str, default=argparse.SUPPRESS, choices=["stage", "activity"])
    p.add_argument("--drop_stages", type=str, default=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS)

    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--test_size", type=float, default=argparse.SUPPRESS)
    p.add_argument("--val_size", type=float, default=argparse.SUPPRESS)
    p.add_argument("--max_print_errors", type=int, default=argparse.SUPPRESS)
    p.add_argument("--checkpoint_dir", type=str, default=argparse.SUPPRESS)
    p.add_argument("--min_class_count", type=int, default=argparse.SUPPRESS)

    p.add_argument("--labeled_ratio", type=float, default=argparse.SUPPRESS)
    p.add_argument("--pseudo_label_threshold", type=float, default=argparse.SUPPRESS)
    p.add_argument("--tree_n_estimators", type=int, default=argparse.SUPPRESS)
    p.add_argument("--tree_max_features", type=str, default=argparse.SUPPRESS)
    p.add_argument("--hgb_max_iter", type=int, default=argparse.SUPPRESS)
    p.add_argument("--hgb_learning_rate", type=float, default=argparse.SUPPRESS)

    p.add_argument("--xgb_num_boost_round", type=int, default=argparse.SUPPRESS)
    p.add_argument("--xgb_early_stopping_rounds", type=int, default=argparse.SUPPRESS)
    p.add_argument("--xgb_eta", type=float, default=argparse.SUPPRESS)
    p.add_argument("--xgb_max_depth", type=int, default=argparse.SUPPRESS)
    p.add_argument("--xgb_subsample", type=float, default=argparse.SUPPRESS)
    p.add_argument("--xgb_colsample_bytree", type=float, default=argparse.SUPPRESS)
    p.add_argument("--xgb_reg_lambda", type=float, default=argparse.SUPPRESS)
    p.add_argument("--xgb_min_child_weight", type=float, default=argparse.SUPPRESS)
    p.add_argument("--xgb_max_bin", type=int, default=argparse.SUPPRESS)
    p.add_argument("--xgb_scale_pos_weight", type=float, default=argparse.SUPPRESS)

    p.add_argument("--cascade_threshold_min", type=float, default=argparse.SUPPRESS)
    p.add_argument("--cascade_threshold_max", type=float, default=argparse.SUPPRESS)
    p.add_argument("--cascade_threshold_steps", type=int, default=argparse.SUPPRESS)
    p.add_argument("--cascade_objective", type=str, default=argparse.SUPPRESS, choices=["min"])
    p.add_argument("--stage1_threshold_objective", type=str, default=argparse.SUPPRESS, choices=["min", "f1", "precision"])
    p.add_argument("--stage1_min_recall", type=float, default=argparse.SUPPRESS)
    p.add_argument("--fixed_stage1_threshold", type=float, default=argparse.SUPPRESS)
    p.add_argument("--stage1_tau_b", type=float, default=argparse.SUPPRESS)
    p.add_argument("--stage1_tau_m", type=float, default=argparse.SUPPRESS)
    p.add_argument("--stage2_margin_min", type=float, default=argparse.SUPPRESS)

    p.add_argument("--oversample_rare_classes", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--oversample_target_count", type=int, default=argparse.SUPPRESS)
    p.add_argument("--stage1_gate_method", type=str, default=argparse.SUPPRESS, choices=["threshold", "conformal_fpr"])
    p.add_argument("--stage1_fpr_budget", type=float, default=argparse.SUPPRESS)
    p.add_argument("--artifacts_dir", type=str, default=argparse.SUPPRESS)
    args = p.parse_args()

    cfg = ExperimentConfig(
        dataset="dapt2020",
        model_type="tree",
        malicious_method="semi",
        inference_policy="original",
        task="all",
        stage2_label="stage",
        seed=42,
        test_size=0.2,
        val_size=0.1,
        max_print_errors=0,
        checkpoint_dir="checkpoints",
        min_class_count=1,
        labeled_ratio=0.5,
        pseudo_label_threshold=0.9,
        tree_n_estimators=2500,
        tree_max_features="sqrt",
        hgb_max_iter=800,
        hgb_learning_rate=0.06,
        xgb_num_boost_round=8000,
        xgb_early_stopping_rounds=200,
        xgb_eta=0.05,
        xgb_max_depth=8,
        xgb_subsample=0.9,
        xgb_colsample_bytree=0.9,
        xgb_reg_lambda=1.0,
        xgb_min_child_weight=1.0,
        xgb_max_bin=256,
        xgb_scale_pos_weight=1.0,
        cascade_threshold_min=0.05,
        cascade_threshold_max=0.95,
        cascade_threshold_steps=19,
        cascade_objective="min",
        stage1_threshold_objective="min",
        stage1_min_recall=0.9,
        stage1_gate_method="threshold",
        stage1_fpr_budget=0.001,
        stage1_tau_b=None,
        stage1_tau_m=None,
        stage2_margin_min=0.0,
        data_dir="./dataset/dapt2020",
        drop_stages="",
        verbose=False,
        artifacts_dir="artifacts",
    )

    if args.config_path is not None:
        raw_cfg = _load_config_payload(args.config_path)
        cfg_fields = set(asdict(cfg).keys())
        merged = asdict(cfg) | {k: v for k, v in raw_cfg.items() if k in cfg_fields}
        cfg = ExperimentConfig(**merged)

    override = vars(args).copy()
    override.pop("config_path", None)
    override.pop("replay", None)
    override.pop("stage1_ckpt", None)
    override.pop("stage2_ckpt", None)
    override.pop("cic_stage2_with_benign", None)
    override.pop("cic_benign_return_min_conf", None)
    override.pop("allow_context_leakage", None)
    override.pop("split_path", None)
    override.pop("split_seed", None)
    override.pop("split_mode", None)
    override.pop("feature_subset_path", None)
    if override:
        merged = asdict(cfg) | override
        cfg = ExperimentConfig(**merged)

    return cfg, args


def main() -> None:
    cfg, args = parse_args()
    if cfg.model_type != "tree":
        raise NotImplementedError("Only --model_type tree is supported in this script.")
    if cfg.malicious_method != "semi":
        raise NotImplementedError("Only --malicious_method semi is supported in this script.")

    print(f"DEBUG: oversample_rare_classes={getattr(cfg, 'oversample_rare_classes', None)}, oversample_target_count={getattr(cfg, 'oversample_target_count', None)}")

    set_seed(cfg.seed)

    try:
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    except Exception:
        pass
    try:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Could not infer format*")
    except Exception:
        pass
    try:
        warnings.filterwarnings("ignore", category=UserWarning, message=".*If you are loading a serialized model.*")
    except Exception:
        pass

    if bool(getattr(args, "replay", False)):
        stage1_ckpt_path = str(getattr(args, "stage1_ckpt", None) or _default_ckpt_path(cfg.checkpoint_dir, cfg.dataset, cfg.seed, "stage1"))
        stage2_ckpt_path = str(getattr(args, "stage2_ckpt", None) or _default_ckpt_path(cfg.checkpoint_dir, cfg.dataset, cfg.seed, "stage2"))
        if bool(getattr(args, "cic_stage2_with_benign", False)):
            replay_cic_stage2_with_benign_experiment(
                cfg=cfg,
                stage1_ckpt_path=stage1_ckpt_path,
                benign_return_min_conf=float(getattr(args, "cic_benign_return_min_conf", 0.9)),
            )
            return
        replay_from_checkpoints(
            cfg,
            stage1_ckpt_path,
            stage2_ckpt_path,
            split_path=getattr(args, "split_path", None),
            split_seed=int(getattr(args, "split_seed", 42)),
            split_mode=str(getattr(args, "split_mode", "random")),
            allow_context_leakage=bool(getattr(args, "allow_context_leakage", False)),
        )
        return

    if cfg.dataset == "dapt2020":
        df = load_dapt2020_dataset(cfg.data_dir)
    elif cfg.dataset == "zapt":
        df = load_zapt_dataset(cfg.data_dir)
    elif cfg.dataset == "cic2024":
        df = load_cic2024_dataset(cfg.data_dir)
    elif cfg.dataset == "merged_bai":
        df = load_cic2024_dataset(cfg.data_dir)
    elif cfg.dataset == "weekdata":
        df = load_cic2024_dataset(cfg.data_dir)
    elif cfg.dataset == "earlycrow":
        df = load_earlycrow_dataset(cfg.data_dir)
    else:
        raise NotImplementedError(f"Unsupported dataset: {cfg.dataset}")

    if cfg.dataset in {"cic2024", "merged_bai", "weekdata"} and not str(cfg.drop_stages).strip():
        cfg = ExperimentConfig(**(asdict(cfg) | {"drop_stages": "Fuzzers"}))

    drop_stages = [s.strip() for s in str(cfg.drop_stages).split(",") if str(s).strip()]
    if drop_stages:
        before = int(len(df))
        df = df[~df["Stage"].astype(str).isin(drop_stages)].copy()
        if cfg.verbose:
            print("Dropped stages:", drop_stages, f"rows {before} -> {int(len(df))}")

    ensure_dir(cfg.checkpoint_dir)
    errors_out_path = os.path.join(cfg.checkpoint_dir, f"{cfg.dataset}_seed{cfg.seed}_errors.csv")
    if not os.path.exists(errors_out_path):
        meta_cols = [c for c in META_COLS_DAPT if c != "__row_id"]
        pd.DataFrame(
            columns=["dataset", "seed", "drop_stages", "phase", "__row_id", "row_1based"] + meta_cols + ["y_true", "y_pred"]
        ).to_csv(errors_out_path, index=False)

    fixed_split = None
    if not bool(getattr(args, "allow_context_leakage", False)):
        split_path = getattr(args, "split_path", None)
        split_seed = int(getattr(args, "split_seed", 42))
        split_mode = str(getattr(args, "split_mode", "random"))
        stage_series = df["Stage"].astype(str)
        y_bin = (stage_series != "Benign").astype(np.int64).to_numpy()
        y_split = stage_series.to_numpy() if cfg.dataset in {"cic2024", "merged_bai", "weekdata"} else y_bin
        if split_path:
            idx_train, idx_val, idx_test = _load_or_create_split_indices(
                path=str(split_path),
                y=y_split,
                seed=split_seed,
                test_size=cfg.test_size,
                val_size=cfg.val_size,
            )
        elif split_mode == "flowkey":
            idx_train, idx_val, idx_test = _split_indices_flowkey(
                df=df,
                y=y_bin,
                seed=split_seed,
                test_size=cfg.test_size,
                val_size=cfg.val_size,
            )
        else:
            idx_train, idx_val, idx_test = split_indices(
                y=y_split,
                seed=cfg.seed,
                test_size=cfg.test_size,
                val_size=cfg.val_size,
            )
        df = add_network_context_group_stats_train_only(df, idx_train=idx_train)
        df = _refresh_log1p_columns(df)
        fixed_split = (idx_train, idx_val, idx_test)

    selected_feature_cols = _load_selected_feature_cols(getattr(args, "feature_subset_path", None))

    stage1_model, stage_split, stage1_threshold = run_stage_binary(
        df,
        cfg,
        errors_out_path=errors_out_path,
        fixed_split=fixed_split,
        selected_feature_cols=selected_feature_cols,
    )
    if cfg.task == "stage1":
        return
    p_train = _predict_binary_prob_pos(stage1_model, stage_split.X_train)
    p_val = _predict_binary_prob_pos(stage1_model, stage_split.X_val)
    p_test = _predict_binary_prob_pos(stage1_model, stage_split.X_test)
    run_cascade_feedback_search(
        df,
        cfg,
        stage1_split=stage_split,
        p_train=p_train,
        p_val=p_val,
        p_test=p_test,
        stage1_threshold=stage1_threshold,
        errors_out_path=errors_out_path,
        selected_feature_cols=selected_feature_cols,
    )


if __name__ == "__main__":
    main()

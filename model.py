from __future__ import annotations

import json
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import urllib.request

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import dask.array as da
    from dask.distributed import Client, LocalCluster
    import xgboost.dask as dxgb
except Exception:
    da = None
    Client = None
    LocalCluster = None
    dxgb = None

from config import ExperimentConfig

_FED_GATEWAY_MAP_CACHE: dict[str, dict[int, int]] = {}
_XGB_CUDA_PROBE_CACHE: tuple[bool, str] | None = None
_XGB_BACKEND_LOGGED: set[str] = set()


def _debug_report_cic_stage2_shape(hypothesis_id: str, location: str, msg: str, data: dict) -> None:
    # #region debug-point A:shape-report
    try:
        _p = ".dbg/cic-stage2-shape.env"
        _u = "http://127.0.0.1:7777/event"
        _s = "cic-stage2-shape"
        if os.path.exists(_p):
            with open(_p, "r", encoding="utf-8") as _f:
                _c = _f.read()
            for _line in _c.splitlines():
                if _line.startswith("DEBUG_SERVER_URL="):
                    _u = _line.split("=", 1)[1].strip() or _u
                elif _line.startswith("DEBUG_SESSION_ID="):
                    _s = _line.split("=", 1)[1].strip() or _s
        _payload = {
            "sessionId": _s,
            "runId": "pre-fix",
            "hypothesisId": str(hypothesis_id),
            "location": str(location),
            "msg": f"[DEBUG] {msg}",
            "data": data,
        }
        urllib.request.urlopen(
            urllib.request.Request(
                _u,
                data=json.dumps(_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=1.0,
        ).read()
    except Exception:
        pass
    # #endregion


def _debug_report_torch_cuda_enum(hypothesis_id: str, location: str, msg: str, data: dict) -> None:
    # #region debug-point A:torch-cuda-enum
    try:
        _p = ".dbg/torch-cuda-enum.env"
        _u = "http://127.0.0.1:7777/event"
        _s = "torch-cuda-enum"
        if os.path.exists(_p):
            with open(_p, "r", encoding="utf-8") as _f:
                _c = _f.read()
            for _line in _c.splitlines():
                if _line.startswith("DEBUG_SERVER_URL="):
                    _u = _line.split("=", 1)[1].strip() or _u
                elif _line.startswith("DEBUG_SESSION_ID="):
                    _s = _line.split("=", 1)[1].strip() or _s
        _payload = {
            "sessionId": _s,
            "runId": "pre-fix",
            "hypothesisId": str(hypothesis_id),
            "location": str(location),
            "msg": f"[DEBUG] {msg}",
            "data": data,
        }
        urllib.request.urlopen(
            urllib.request.Request(
                _u,
                data=json.dumps(_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=1.0,
        ).read()
    except Exception:
        pass
    # #endregion


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.head(z)


class _TorchTabularWrapper:
    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
        dropout: float = 0.15,
        batch_size: int = 4096,
    ) -> None:
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.state_dict = {str(k): v.detach().cpu() for k, v in state_dict.items()}

    def _build_model(self) -> TabularMLP:
        model = TabularMLP(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        model.load_state_dict(self.state_dict, strict=True)
        model.eval()
        return model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        model = self._build_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        X_np = np.asarray(X, dtype=np.float32)
        outs: list[np.ndarray] = []
        bs = max(64, int(self.batch_size))
        with torch.no_grad():
            for start in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[start : start + bs]).to(device)
                logits = model(xb)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
                outs.append(prob)
        if not outs:
            return np.zeros((0, self.num_classes), dtype=np.float32)
        return np.concatenate(outs, axis=0).astype(np.float32, copy=False)


def _probe_xgb_cuda() -> tuple[bool, str]:
    global _XGB_CUDA_PROBE_CACHE
    if _XGB_CUDA_PROBE_CACHE is not None:
        return _XGB_CUDA_PROBE_CACHE
    if xgb is None:
        _XGB_CUDA_PROBE_CACHE = (False, "xgboost import failed")
        return _XGB_CUDA_PROBE_CACHE
    _debug_report_torch_cuda_enum(
        "A",
        "model.py:_probe_xgb_cuda:start",
        "starting cuda probe",
        {
            "pid": int(os.getpid()),
            "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
            "nvidia_visible_devices": str(os.environ.get("NVIDIA_VISIBLE_DEVICES", "")),
            "torch_version": str(getattr(torch, "__version__", "unknown")),
            "xgboost_version": str(getattr(xgb, "__version__", "unknown")),
        },
    )
    try:
        _torch_is_available = bool(torch.cuda.is_available())
        _torch_device_count = int(torch.cuda.device_count())
        _debug_report_torch_cuda_enum(
            "A",
            "model.py:_probe_xgb_cuda:torch",
            "torch cuda probe result",
            {
                "is_available": bool(_torch_is_available),
                "device_count": int(_torch_device_count),
            },
        )
    except Exception as exc:
        _debug_report_torch_cuda_enum(
            "A",
            "model.py:_probe_xgb_cuda:torch-exc",
            "torch cuda probe raised",
            {
                "error": repr(exc),
            },
        )
    try:
        dtrain = xgb.DMatrix(np.asarray([[0.0], [1.0]], dtype=np.float32), label=np.asarray([0, 1], dtype=np.float32))
        xgb.train(
            params={
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 1,
                "eta": 1.0,
            },
            dtrain=dtrain,
            num_boost_round=1,
            verbose_eval=False,
        )
    except Exception as exc:
        _debug_report_torch_cuda_enum(
            "B",
            "model.py:_probe_xgb_cuda:xgb-exc",
            "xgboost cuda probe failed",
            {
                "error": repr(exc),
            },
        )
        _XGB_CUDA_PROBE_CACHE = (False, f"xgboost CUDA probe failed: {exc!r}")
        return _XGB_CUDA_PROBE_CACHE
    _debug_report_torch_cuda_enum(
        "B",
        "model.py:_probe_xgb_cuda:xgb-ok",
        "xgboost cuda probe succeeded",
        {
            "result": "cuda",
        },
    )
    _XGB_CUDA_PROBE_CACHE = (True, "cuda")
    return _XGB_CUDA_PROBE_CACHE


def _can_use_xgb_cuda() -> bool:
    ok, reason = _probe_xgb_cuda()
    if not ok:
        raise RuntimeError(
            "XGBoost CUDA backend is unavailable; refusing to fall back to CPU. "
            f"Probe result: {reason}"
        )
    return True


def _resolve_xgb_device(cfg: ExperimentConfig, *, context: str) -> str:
    if _use_federated_dask_backend(cfg):
        if context not in _XGB_BACKEND_LOGGED:
            print(f"[xgb-backend] {context}: backend=xgb_dask_cpu", flush=True)
            _XGB_BACKEND_LOGGED.add(context)
        return "cpu"
    _can_use_xgb_cuda()
    if context not in _XGB_BACKEND_LOGGED:
        print(f"[xgb-backend] {context}: backend=cuda", flush=True)
        _XGB_BACKEND_LOGGED.add(context)
    return "cuda"


def _can_use_dask_xgb() -> bool:
    return bool(xgb is not None and dxgb is not None and da is not None and Client is not None and LocalCluster is not None)


def _use_federated_dask_backend(cfg: ExperimentConfig) -> bool:
    if not bool(getattr(cfg, "federated_enabled", False)):
        return False
    backend = str(getattr(cfg, "federated_backend", "single_process") or "single_process").strip().lower()
    return backend == "dask"


def _load_federated_gateway_map(path: str) -> dict[int, int]:
    norm = os.path.abspath(path)
    cached = _FED_GATEWAY_MAP_CACHE.get(norm)
    if cached is not None:
        return cached
    df = pd.read_csv(norm, usecols=["row_id", "gateway_id"])
    mapping = {
        int(row_id): int(gateway_id)
        for row_id, gateway_id in zip(
            df["row_id"].astype(np.int64, copy=False).tolist(),
            df["gateway_id"].astype(np.int64, copy=False).tolist(),
        )
    }
    _FED_GATEWAY_MAP_CACHE[norm] = mapping
    return mapping


def _partition_ids_from_row_ids(row_ids: np.ndarray | None, cfg: ExperimentConfig) -> np.ndarray | None:
    if row_ids is None:
        return None
    map_path = str(getattr(cfg, "federated_gateway_map_path", "") or "").strip()
    if not map_path:
        return None
    mapping = _load_federated_gateway_map(map_path)
    arr = np.asarray(row_ids, dtype=np.int64)
    part = np.asarray([mapping.get(int(r), -1) for r in arr.tolist()], dtype=np.int64)
    if np.any(part < 0):
        missing = arr[part < 0][:10].tolist()
        raise ValueError(f"Missing gateway mapping for row ids: {missing}")
    return part


def _default_row_chunks(n_rows: int, worker_count: int) -> tuple[int, ...]:
    n_rows = int(n_rows)
    worker_count = max(1, int(worker_count))
    if n_rows <= 0:
        return (0,)
    worker_count = min(worker_count, n_rows)
    base = n_rows // worker_count
    rem = n_rows % worker_count
    return tuple(base + (1 if i < rem else 0) for i in range(worker_count) if base + (1 if i < rem else 0) > 0)


def _reorder_by_partition_ids(
    X: np.ndarray,
    y: np.ndarray,
    part_ids: np.ndarray | None,
    *,
    worker_count: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[int, ...]]:
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    w_arr = None if sample_weight is None else np.asarray(sample_weight)
    if len(X_arr) == 0:
        return X_arr, y_arr, w_arr, (0,)
    if part_ids is None:
        return X_arr, y_arr, w_arr, _default_row_chunks(len(X_arr), worker_count)
    pid = np.asarray(part_ids, dtype=np.int64)
    order = np.argsort(pid, kind="mergesort")
    pid_sorted = pid[order]
    uniq, counts = np.unique(pid_sorted, return_counts=True)
    if np.any(uniq < 0):
        raise ValueError(f"Invalid partition ids encountered: {uniq.tolist()}")
    chunks = tuple(int(c) for c in counts.tolist() if int(c) > 0)
    return (
        X_arr[order],
        y_arr[order],
        None if w_arr is None else w_arr[order],
        chunks if chunks else _default_row_chunks(len(X_arr), worker_count),
    )


def _numpy_to_dask_rows(arr: np.ndarray, row_chunks: tuple[int, ...]):
    arr_np = np.asarray(arr)
    if arr_np.ndim not in {1, 2}:
        raise ValueError(f"Unsupported ndarray rank for Dask conversion: {arr_np.ndim}")
    if len(arr_np) == 0:
        return da.from_array(arr_np, chunks=arr_np.shape if arr_np.shape else (0,))
    parts = []
    start = 0
    for chunk in row_chunks:
        chunk = int(chunk)
        if chunk <= 0:
            continue
        sub = arr_np[start : start + chunk]
        if arr_np.ndim == 1:
            parts.append(da.from_array(sub, chunks=(len(sub),)))
        else:
            parts.append(da.from_array(sub, chunks=(len(sub), arr_np.shape[1])))
        start += chunk
    if start != len(arr_np):
        raise ValueError(f"Row chunking mismatch: consumed={start}, total={len(arr_np)}")
    if len(parts) == 1:
        return parts[0]
    axis = 0
    return da.concatenate(parts, axis=axis)


def _train_xgb_dask(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, object],
    num_boost_round: int,
    early_stopping_rounds: int,
    num_class: int,
    worker_count: int,
    train_part_ids: np.ndarray | None,
    val_part_ids: np.ndarray | None,
) -> _XGBBoosterWrapper:
    if not _can_use_dask_xgb():
        raise RuntimeError("Distributed XGBoost dependencies are unavailable.")
    params = dict(params)
    # Dask multi-worker training cannot share one CUDA device across workers.
    params["device"] = "cpu"
    params["tree_method"] = "hist"
    X_train_ord, y_train_ord, w_train_ord, train_chunks = _reorder_by_partition_ids(
        X_train,
        y_train,
        train_part_ids,
        worker_count=worker_count,
        sample_weight=w_train,
    )
    X_val_ord, y_val_ord, _, val_chunks = _reorder_by_partition_ids(
        X_val,
        y_val,
        val_part_ids,
        worker_count=worker_count,
        sample_weight=None,
    )
    workers = max(1, min(int(worker_count), len(train_chunks)))
    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    )
    client = Client(cluster)
    try:
        dtrain = dxgb.DaskDMatrix(
            client,
            _numpy_to_dask_rows(X_train_ord, train_chunks),
            _numpy_to_dask_rows(y_train_ord.astype(np.float32 if num_class == 2 else np.int64, copy=False), train_chunks),
            weight=None if w_train_ord is None else _numpy_to_dask_rows(w_train_ord.astype(np.float32, copy=False), train_chunks),
        )
        dval = dxgb.DaskDMatrix(
            client,
            _numpy_to_dask_rows(X_val_ord, val_chunks),
            _numpy_to_dask_rows(y_val_ord.astype(np.float32 if num_class == 2 else np.int64, copy=False), val_chunks),
        )
        result = dxgb.train(
            client,
            params=params,
            dtrain=dtrain,
            num_boost_round=int(num_boost_round),
            evals=[(dval, "val")],
            early_stopping_rounds=int(early_stopping_rounds),
            verbose_eval=False,
        )
        booster = result["booster"]
    finally:
        client.close()
        cluster.close()
    return _XGBBoosterWrapper(booster=booster, num_class=num_class)


class _XGBBoosterWrapper:
    def __init__(self, booster, num_class: int) -> None:
        self.booster = booster
        self.num_class = int(num_class)
        self.best_iteration = int(getattr(booster, "best_iteration", -1) or -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d = xgb.DMatrix(X)
        pred = self.booster.predict(d)
        if self.num_class == 2:
            p = np.asarray(pred, dtype=np.float32)
            if p.ndim == 2 and p.shape[1] == 2:
                return p
            return np.stack([1.0 - p, p], axis=1)
        return np.asarray(pred, dtype=np.float32)


class _AvgProbaEnsemble:
    def __init__(self, models: list[object], weights: list[float] | None = None) -> None:
        self.models = list(models)
        if weights is None:
            self.weights = [1.0 for _ in self.models]
        else:
            self.weights = [float(w) for w in weights]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Empty ensemble")
        w = np.asarray(self.weights, dtype=np.float32)
        w = w / float(np.sum(w))
        out = None
        for i, m in enumerate(self.models):
            p = m.predict_proba(X).astype(np.float32, copy=False)
            if out is None:
                out = p * w[i]
            else:
                out = out + p * w[i]
        return out.astype(np.float32, copy=False)


def _fit_xgb_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ExperimentConfig,
    row_id_train: np.ndarray | None = None,
    row_id_val: np.ndarray | None = None,
) -> _XGBBoosterWrapper:
    device = _resolve_xgb_device(cfg, context="fit_xgb_binary")
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": device,
        "seed": int(cfg.seed),
        "eta": float(cfg.xgb_eta),
        "max_depth": int(cfg.xgb_max_depth),
        "subsample": float(cfg.xgb_subsample),
        "colsample_bytree": float(cfg.xgb_colsample_bytree),
        "lambda": float(cfg.xgb_reg_lambda),
        "min_child_weight": float(cfg.xgb_min_child_weight),
        "max_bin": int(cfg.xgb_max_bin),
    }
    if float(cfg.xgb_scale_pos_weight) != 1.0:
        params["scale_pos_weight"] = float(cfg.xgb_scale_pos_weight)
    if _use_federated_dask_backend(cfg):
        return _train_xgb_dask(
            X_train=X_train,
            y_train=y_train.astype(np.int64, copy=False),
            w_train=w_train,
            X_val=X_val,
            y_val=y_val.astype(np.int64, copy=False),
            params=params,
            num_boost_round=int(cfg.xgb_num_boost_round),
            early_stopping_rounds=int(cfg.xgb_early_stopping_rounds),
            num_class=2,
            worker_count=int(getattr(cfg, "federated_num_workers", 1)),
            train_part_ids=_partition_ids_from_row_ids(row_id_train, cfg),
            val_part_ids=_partition_ids_from_row_ids(row_id_val, cfg),
        )

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(cfg.xgb_num_boost_round),
        evals=[(dval, "val")],
        verbose_eval=False,
        early_stopping_rounds=int(cfg.xgb_early_stopping_rounds),
    )
    return _XGBBoosterWrapper(booster=booster, num_class=2)


def _fit_xgb_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    cfg: ExperimentConfig,
    row_id_train: np.ndarray | None = None,
    row_id_val: np.ndarray | None = None,
) -> _XGBBoosterWrapper:
    num_classes = int(num_classes)
    device = _resolve_xgb_device(cfg, context="fit_xgb_multiclass")
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": device,
        "seed": int(cfg.seed),
        "eta": float(cfg.xgb_eta),
        "max_depth": max(2, int(cfg.xgb_max_depth)),
        "subsample": float(cfg.xgb_subsample),
        "colsample_bytree": float(cfg.xgb_colsample_bytree),
        "lambda": float(cfg.xgb_reg_lambda),
        "min_child_weight": float(cfg.xgb_min_child_weight),
        "max_bin": int(cfg.xgb_max_bin),
    }
    if _use_federated_dask_backend(cfg):
        return _train_xgb_dask(
            X_train=X_train,
            y_train=y_train.astype(np.int64, copy=False),
            w_train=w_train,
            X_val=X_val,
            y_val=y_val.astype(np.int64, copy=False),
            params=params,
            num_boost_round=int(cfg.xgb_num_boost_round),
            early_stopping_rounds=int(cfg.xgb_early_stopping_rounds),
            num_class=num_classes,
            worker_count=int(getattr(cfg, "federated_num_workers", 1)),
            train_part_ids=_partition_ids_from_row_ids(row_id_train, cfg),
            val_part_ids=_partition_ids_from_row_ids(row_id_val, cfg),
        )

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(cfg.xgb_num_boost_round),
        evals=[(dval, "val")],
        verbose_eval=False,
        early_stopping_rounds=int(cfg.xgb_early_stopping_rounds),
    )
    return _XGBBoosterWrapper(booster=booster, num_class=num_classes)


def _predict_binary_prob_pos(model: object, X: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(X)[:, 1]
    return prob.astype(np.float32, copy=False)


def _predict_multiclass(
    model: object,
    X: np.ndarray,
    extra: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    prob = model.predict_proba(X).astype(np.float32, copy=False)
    def _overwrite_prob_with_local_subset(
        route_mask: np.ndarray,
        class_ids_raw: np.ndarray | list[int],
        local_prob_raw: np.ndarray,
    ) -> None:
        nonlocal prob
        if not route_mask.any():
            return
        cids = np.asarray(class_ids_raw, dtype=np.int64)
        local_prob = np.asarray(local_prob_raw, dtype=np.float32)
        if cids.ndim != 1 or local_prob.ndim != 2 or local_prob.shape[1] != len(cids):
            return
        denom = np.maximum(local_prob.sum(axis=1, keepdims=True), np.float32(1e-12))
        local_prob = (local_prob / denom).astype(np.float32, copy=False)
        row_idx = np.where(route_mask)[0].astype(np.int64, copy=False)
        prob = prob.copy()
        prob[row_idx, :] = np.float32(0.0)
        prob[np.ix_(row_idx, cids)] = local_prob
    if extra is not None and (extra.get("hier_group_models") is not None or extra.get("twolevel_expert_models") is not None):
        _debug_report_cic_stage2_shape(
            "A",
            "model.py:_predict_multiclass:entry",
            "multiclass entry with local experts",
            {
                "x_shape": list(np.shape(X)),
                "prob_shape": list(np.shape(prob)),
                "has_hier_group_models": extra.get("hier_group_models") is not None,
                "has_twolevel_expert_models": extra.get("twolevel_expert_models") is not None,
            },
        )

    class_mult = None if extra is None else extra.get("class_multiplier", None)
    score = prob
    if class_mult is not None:
        score = prob * np.asarray(class_mult, dtype=np.float32)
    def _refresh_score_from_prob() -> None:
        nonlocal score
        score = prob
        if class_mult is not None:
            score = prob * np.asarray(class_mult, dtype=np.float32)

    guard_min_prob = None if extra is None else extra.get("guard_min_prob", None)
    if guard_min_prob is not None:
        guard_min_prob = np.asarray(guard_min_prob, dtype=np.float32)
        masked = score.copy()
        for c in range(masked.shape[1]):
            masked[prob[:, c] < guard_min_prob[c], c] = -1.0
        all_masked = masked.max(axis=1) < 0.0
        if all_masked.any():
            masked[all_masked] = score[all_masked]
        score = masked

    pred = score.argmax(axis=1).astype(np.int64, copy=False)

    cic_head_model = None if extra is None else extra.get("cic2024_head_cluster_model", None)
    cic_head_router_model = None if extra is None else extra.get("cic2024_head_router_model", None)
    cic_head_class_ids = None if extra is None else extra.get("cic2024_head_cluster_class_ids", None)
    cic_head_router_hi = None if extra is None else extra.get("cic2024_head_router_hi", None)
    cic_head_router_lo = None if extra is None else extra.get("cic2024_head_router_lo", None)
    cic_head_sum_threshold = None if extra is None else extra.get("cic2024_head_cluster_sum_threshold", None)
    cic_head_margin = None if extra is None else extra.get("cic2024_head_cluster_margin", None)
    if (
        cic_head_model is not None
        and cic_head_router_model is not None
        and cic_head_class_ids is not None
        and cic_head_router_hi is not None
        and cic_head_router_lo is not None
        and cic_head_sum_threshold is not None
        and cic_head_margin is not None
    ):
        cids = np.asarray(cic_head_class_ids, dtype=np.int64)
        if cids.ndim == 1 and len(cids) == 3:
            router_prob = cic_head_router_model.predict_proba(X)[:, 1].astype(np.float32, copy=False)
            top2 = np.argsort(score, axis=1)[:, -2:]
            top1 = top2[:, 1].astype(np.int64, copy=False)
            top2c = top2[:, 0].astype(np.int64, copy=False)
            idx = np.arange(len(top1), dtype=np.int64)
            diff = (score[idx, top1] - score[idx, top2c]).astype(np.float32, copy=False)
            sum_head = prob[:, cids].sum(axis=1).astype(np.float32, copy=False)
            hi = np.float32(float(cic_head_router_hi))
            lo = np.float32(float(cic_head_router_lo))
            sum_thr = np.float32(float(cic_head_sum_threshold))
            margin = np.float32(float(cic_head_margin))
            near_head = np.isin(top1, cids, assume_unique=False) | np.isin(top2c, cids, assume_unique=False)
            route = (router_prob >= hi) | (
                (router_prob >= lo)
                & (sum_head >= sum_thr)
                & (near_head | (diff <= margin))
            )
            if route.any():
                p_local = cic_head_model.predict_proba(X[route]).astype(np.float32, copy=False)
                choose = p_local.argmax(axis=1).astype(np.int64, copy=False)
                mapped = cids[choose].astype(np.int64, copy=False)
                _overwrite_prob_with_local_subset(route, cids, p_local)
                _refresh_score_from_prob()
                pred = pred.copy()
                pred[np.where(route)[0]] = mapped

    prob_override_thresholds = None if extra is None else extra.get("prob_override_thresholds", None)
    prob_override_priority = None if extra is None else extra.get("prob_override_priority", None)
    if prob_override_thresholds is not None and prob_override_priority is not None:
        thr = np.asarray(prob_override_thresholds, dtype=np.float32)
        pri = np.asarray(prob_override_priority, dtype=np.int64)
        if thr.ndim == 1 and pri.ndim == 1 and len(thr) == prob.shape[1] and len(pri) > 0:
            pred = pred.copy()
            for cid in pri.tolist():
                cid = int(cid)
                if cid < 0 or cid >= prob.shape[1]:
                    continue
                t = float(thr[cid])
                if t <= 0.0:
                    continue
                mask = prob[:, cid] >= t
                if mask.any():
                    pred[mask] = cid

    ovr_all_models = None if extra is None else extra.get("ovr_all_models", None)
    ovr_all_thresholds = None if extra is None else extra.get("ovr_all_thresholds", None)
    ovr_all_class_ids = None if extra is None else extra.get("ovr_all_class_ids", None)
    if ovr_all_models is not None and ovr_all_thresholds is not None and ovr_all_class_ids is not None:
        thr = np.asarray(ovr_all_thresholds, dtype=np.float32)
        cids = np.asarray(ovr_all_class_ids, dtype=np.int64)
        if thr.ndim == 1 and cids.ndim == 1 and len(thr) == len(cids) and len(ovr_all_models) == len(cids):
            best_score = np.full(len(X), -1e9, dtype=np.float32)
            best_cid = np.full(len(X), -1, dtype=np.int64)
            for m, t, cid in zip(ovr_all_models, thr.tolist(), cids.tolist()):
                cid = int(cid)
                p = m.predict_proba(X)[:, 1].astype(np.float32, copy=False)
                t = float(t)
                mask = p >= t
                if not mask.any():
                    continue
                s = (p - np.float32(t)).astype(np.float32, copy=False)
                upd = mask & (s > best_score)
                if upd.any():
                    best_score[upd] = s[upd]
                    best_cid[upd] = cid
            any_hit = best_cid >= 0
            if any_hit.any():
                pred = pred.copy()
                pred[any_hit] = best_cid[any_hit]

    ovr_full_models = None if extra is None else extra.get("ovr_full_models", None)
    ovr_full_thresholds = None if extra is None else extra.get("ovr_full_thresholds", None)
    ovr_full_class_ids = None if extra is None else extra.get("ovr_full_class_ids", None)
    ovr_full_always = False if extra is None else bool(extra.get("ovr_full_always", False))
    if ovr_full_models is not None and ovr_full_thresholds is not None and ovr_full_class_ids is not None:
        thr = np.asarray(ovr_full_thresholds, dtype=np.float32)
        cids = np.asarray(ovr_full_class_ids, dtype=np.int64)
        if thr.ndim == 1 and cids.ndim == 1 and len(thr) == len(cids) and len(ovr_full_models) == len(cids):
            best_score = np.full(len(X), -1e9, dtype=np.float32)
            best_cid = np.full(len(X), -1, dtype=np.int64)
            for m, t, cid in zip(ovr_full_models, thr.tolist(), cids.tolist()):
                cid = int(cid)
                p = m.predict_proba(X)[:, 1].astype(np.float32, copy=False)
                t = float(t)
                s = (p - np.float32(t)).astype(np.float32, copy=False)
                upd = s > best_score
                if upd.any():
                    best_score[upd] = s[upd]
                    best_cid[upd] = cid
            any_hit = best_score >= np.float32(0.0)
            if ovr_full_always:
                pred = pred.copy()
                pred[:] = best_cid
            elif any_hit.any():
                pred = pred.copy()
                pred[any_hit] = best_cid[any_hit]

    ovr_models = None if extra is None else extra.get("ovr_models", None)
    ovr_thresholds = None if extra is None else extra.get("ovr_thresholds", None)
    ovr_class_ids = None if extra is None else extra.get("ovr_class_ids", None)
    if ovr_models is not None and ovr_thresholds is not None and ovr_class_ids is not None:
        for m, thr, cid in zip(ovr_models, ovr_thresholds, ovr_class_ids):
            p = m.predict_proba(X)[:, 1].astype(np.float32, copy=False)
            override = p >= float(thr)
            if override.any():
                pred = pred.copy()
                pred[override] = int(cid)

    dataex_model = None if extra is None else extra.get("dataex_model", None)
    dataex_threshold = None if extra is None else extra.get("dataex_threshold", None)
    dataex_threshold_hi = None if extra is None else extra.get("dataex_threshold_hi", None)
    dataex_threshold_lo = None if extra is None else extra.get("dataex_threshold_lo", None)
    dataex_guard_min_prob = None if extra is None else extra.get("dataex_guard_min_prob", None)
    dataex_require_base_prob = None if extra is None else extra.get("dataex_require_base_prob", None)
    dataex_id = None if extra is None else extra.get("dataex_id", None)
    if dataex_model is not None and (dataex_threshold is not None or dataex_threshold_hi is not None) and dataex_id is not None:
        ex_prob = dataex_model.predict_proba(X)[:, 1].astype(np.float32, copy=False)
        ex_id = int(dataex_id)
        override = None
        if dataex_threshold_hi is not None:
            hi = float(dataex_threshold_hi)
            lo = hi if dataex_threshold_lo is None else float(dataex_threshold_lo)
            guard = 0.0 if dataex_guard_min_prob is None else float(dataex_guard_min_prob)
            base_ok = np.ones(len(X), dtype=bool)
            if 0 <= ex_id < prob.shape[1]:
                req = float(dataex_require_base_prob) if dataex_require_base_prob is not None else 0.0
                req = float(max(req, guard))
                if req > 0.0:
                    base_ok = prob[:, ex_id].astype(np.float32, copy=False) >= np.float32(req)
            override = (ex_prob >= np.float32(hi)) | ((ex_prob >= np.float32(lo)) & base_ok)
        else:
            override = ex_prob >= float(dataex_threshold)
        if override.any():
            pred = pred.copy()
            pred[override] = ex_id

    lateral_model = None if extra is None else extra.get("lateral_model", None)
    lateral_threshold = None if extra is None else extra.get("lateral_threshold", None)
    lateral_id = None if extra is None else extra.get("lateral_id", None)
    if lateral_model is not None and lateral_threshold is not None and lateral_id is not None:
        lat_prob = lateral_model.predict_proba(X)[:, 1].astype(np.float32, copy=False)
        override = lat_prob >= float(lateral_threshold)
        if override.any():
            pred = pred.copy()
            pred[override] = int(lateral_id)

    moe_triple_model = None if extra is None else extra.get("moe_triple_model", None)
    moe_triple_class_ids = None if extra is None else extra.get("moe_triple_class_ids", None)
    moe_triple_margin = None if extra is None else extra.get("moe_triple_margin", None)
    moe_triple_min_sum = None if extra is None else extra.get("moe_triple_min_sum", None)
    if (
        moe_triple_model is not None
        and moe_triple_class_ids is not None
        and moe_triple_margin is not None
        and moe_triple_min_sum is not None
    ):
        tri = np.asarray(moe_triple_class_ids, dtype=np.int64)
        if tri.ndim == 1 and len(tri) == 3:
            margin = float(moe_triple_margin)
            min_sum = float(moe_triple_min_sum)
            if margin > 0.0:
                top2 = np.argsort(score, axis=1)[:, -2:]
                top1 = top2[:, 1].astype(np.int64, copy=False)
                top2c = top2[:, 0].astype(np.int64, copy=False)
                n = int(len(top1))
                idx = np.arange(n, dtype=np.int64)
                diff = (score[idx, top1] - score[idx, top2c]).astype(np.float32, copy=False)
                sum_tri = prob[:, tri[0]] + prob[:, tri[1]] + prob[:, tri[2]]
                route = (
                    np.isin(top1, tri, assume_unique=False)
                    & np.isin(top2c, tri, assume_unique=False)
                    & (diff <= np.float32(margin))
                    & (sum_tri >= np.float32(min_sum))
                )
                if route.any():
                    p3 = moe_triple_model.predict_proba(X[route]).astype(np.float32, copy=False)
                    choose = p3.argmax(axis=1).astype(np.int64, copy=False)
                    mapped = tri[choose].astype(np.int64, copy=False)
                    pred = pred.copy()
                    pred[np.where(route)[0]] = mapped

    ovo_models = None if extra is None else extra.get("ovo_models", None)
    ovo_pairs = None if extra is None else extra.get("ovo_pairs", None)
    ovo_margin = None if extra is None else extra.get("ovo_margin", None)
    if ovo_models is not None and ovo_pairs is not None:
        pairs = np.asarray(ovo_pairs, dtype=np.int64)
        margin = float(ovo_margin) if ovo_margin is not None else 0.0
        if pairs.ndim == 2 and pairs.shape[1] == 2 and len(ovo_models) == int(pairs.shape[0]) and margin > 0.0:
            top2 = np.argsort(score, axis=1)[:, -2:]
            top1 = top2[:, 1].astype(np.int64, copy=False)
            top2c = top2[:, 0].astype(np.int64, copy=False)
            n = int(len(top1))
            idx = np.arange(n, dtype=np.int64)
            diff = np.abs(prob[idx, top1] - prob[idx, top2c]).astype(np.float32, copy=False)
            for (a, b), m in zip(pairs, ovo_models):
                a = int(a)
                b = int(b)
                mask = (
                    ((top1 == a) & (top2c == b)) | ((top1 == b) & (top2c == a))
                ) & (diff <= np.float32(margin))
                if not mask.any():
                    continue
                p = m.predict_proba(X[mask])[:, 1].astype(np.float32, copy=False)
                choose_b = p >= np.float32(0.5)
                if mask.any():
                    pred = pred.copy()
                    pred_idx = np.where(mask)[0]
                    pred[pred_idx] = np.where(choose_b, b, a).astype(np.int64, copy=False)

    dos_ex_model = None if extra is None else extra.get("dos_ex_model", None)
    dos_ex_a = None if extra is None else extra.get("dos_ex_a", None)
    dos_ex_b = None if extra is None else extra.get("dos_ex_b", None)
    dos_ex_margin = None if extra is None else extra.get("dos_ex_margin", None)
    dos_ex_min_sum = None if extra is None else extra.get("dos_ex_min_sum", None)
    dos_ex_mode = None if extra is None else extra.get("dos_ex_mode", None)
    if dos_ex_model is not None and dos_ex_a is not None and dos_ex_b is not None:
        a = int(dos_ex_a)
        b = int(dos_ex_b)
        if 0 <= a < prob.shape[1] and 0 <= b < prob.shape[1]:
            mode = "top2" if dos_ex_mode is None else str(dos_ex_mode)
            margin = 1.1 if dos_ex_margin is None else float(dos_ex_margin)
            min_sum = 0.0 if dos_ex_min_sum is None else float(dos_ex_min_sum)
            if margin > 0.0:
                top2 = np.argsort(score, axis=1)[:, -2:]
                top1 = top2[:, 1].astype(np.int64, copy=False)
                top2c = top2[:, 0].astype(np.int64, copy=False)
                diff_ab = np.abs(prob[:, a] - prob[:, b]).astype(np.float32, copy=False)
                sum_ab = (prob[:, a] + prob[:, b]).astype(np.float32, copy=False)
                if mode == "top1":
                    route = ((top1 == a) | (top1 == b)) & (diff_ab <= np.float32(margin)) & (sum_ab >= np.float32(min_sum))
                else:
                    route = (
                        (((top1 == a) & (top2c == b)) | ((top1 == b) & (top2c == a)))
                        & (diff_ab <= np.float32(margin))
                        & (sum_ab >= np.float32(min_sum))
                    )
                if route.any():
                    p = dos_ex_model.predict_proba(X[route])[:, 1].astype(np.float32, copy=False)
                    choose_b = p >= np.float32(0.5)
                    pred = pred.copy()
                    pred_idx = np.where(route)[0]
                    pred[pred_idx] = np.where(choose_b, b, a).astype(np.int64, copy=False)

    hier_group_models = None if extra is None else extra.get("hier_group_models", None)
    hier_group_class_ids = None if extra is None else extra.get("hier_group_class_ids", None)
    hier_group_sum_thresholds = None if extra is None else extra.get("hier_group_sum_thresholds", None)
    hier_group_margin_thresholds = None if extra is None else extra.get("hier_group_margin_thresholds", None)
    hier_group_modes = None if extra is None else extra.get("hier_group_modes", None)
    if (
        hier_group_models is not None
        and hier_group_class_ids is not None
        and hier_group_sum_thresholds is not None
        and hier_group_margin_thresholds is not None
        and hier_group_modes is not None
    ):
        n_models = int(len(hier_group_models))
        if (
            len(hier_group_class_ids) == n_models
            and len(hier_group_sum_thresholds) == n_models
            and len(hier_group_margin_thresholds) == n_models
            and len(hier_group_modes) == n_models
            and n_models > 0
        ):
            top2 = np.argsort(score, axis=1)[:, -2:]
            top1 = top2[:, 1].astype(np.int64, copy=False)
            top2c = top2[:, 0].astype(np.int64, copy=False)
            idx = np.arange(len(top1), dtype=np.int64)
            diff = (score[idx, top1] - score[idx, top2c]).astype(np.float32, copy=False)
            for m, cids_raw, sum_thr, margin_thr, mode in zip(
                hier_group_models,
                hier_group_class_ids,
                hier_group_sum_thresholds,
                hier_group_margin_thresholds,
                hier_group_modes,
            ):
                cids = np.asarray(cids_raw, dtype=np.int64)
                if cids.ndim != 1 or len(cids) < 2:
                    continue
                in_top1 = np.isin(top1, cids, assume_unique=False)
                in_top2 = np.isin(top2c, cids, assume_unique=False)
                if str(mode) == "top1":
                    route = in_top1
                elif str(mode) == "either":
                    route = in_top1 | in_top2
                else:
                    route = in_top1 & in_top2
                route = route & ((prob[:, cids].sum(axis=1)) >= np.float32(float(sum_thr)))
                if float(margin_thr) > 0.0:
                    route = route & (diff <= np.float32(float(margin_thr)))
                if not route.any():
                    continue
                p_local = m.predict_proba(X[route]).astype(np.float32, copy=False)
                choose = p_local.argmax(axis=1).astype(np.int64, copy=False)
                mapped = cids[choose].astype(np.int64, copy=False)
                _debug_report_cic_stage2_shape(
                    "B",
                    "model.py:_predict_multiclass:hier_group",
                    "hier_group route triggered",
                    {
                        "route_count": int(route.sum()),
                        "cids": [int(v) for v in cids.tolist()],
                        "p_local_shape": list(np.shape(p_local)),
                        "choose_shape": list(np.shape(choose)),
                        "mapped_shape": list(np.shape(mapped)),
                    },
                )
                _overwrite_prob_with_local_subset(route, cids, p_local)
                pred = pred.copy()
                pred[np.where(route)[0]] = mapped

    twolevel_router_model = None if extra is None else extra.get("twolevel_router_model", None)
    twolevel_expert_models = None if extra is None else extra.get("twolevel_expert_models", None)
    twolevel_router_group_ids = None if extra is None else extra.get("twolevel_router_group_ids", None)
    twolevel_group_class_ids = None if extra is None else extra.get("twolevel_group_class_ids", None)
    twolevel_router_thresholds = None if extra is None else extra.get("twolevel_router_thresholds", None)
    twolevel_router_margins = None if extra is None else extra.get("twolevel_router_margins", None)
    if (
        twolevel_router_model is not None
        and twolevel_expert_models is not None
        and twolevel_router_group_ids is not None
        and twolevel_group_class_ids is not None
        and twolevel_router_thresholds is not None
        and twolevel_router_margins is not None
    ):
        n_experts = int(len(twolevel_expert_models))
        if (
            len(twolevel_router_group_ids) == n_experts
            and len(twolevel_group_class_ids) == n_experts
            and len(twolevel_router_thresholds) == n_experts
            and len(twolevel_router_margins) == n_experts
            and n_experts > 0
        ):
            router_prob = twolevel_router_model.predict_proba(X).astype(np.float32, copy=False)
            router_pred = router_prob.argmax(axis=1).astype(np.int64, copy=False)
            router_conf = router_prob.max(axis=1).astype(np.float32, copy=False)
            if router_prob.shape[1] >= 2:
                router_top2 = np.sort(router_prob, axis=1)[:, -2:]
                router_margin = (router_top2[:, 1] - router_top2[:, 0]).astype(np.float32, copy=False)
            else:
                router_margin = np.ones(len(X), dtype=np.float32)
            for m, gid, cids_raw, thr, margin in zip(
                twolevel_expert_models,
                twolevel_router_group_ids,
                twolevel_group_class_ids,
                twolevel_router_thresholds,
                twolevel_router_margins,
            ):
                cids = np.asarray(cids_raw, dtype=np.int64)
                gid = int(gid)
                if cids.ndim != 1 or len(cids) < 2:
                    continue
                route = (router_pred == gid) & (router_conf >= np.float32(float(thr)))
                if float(margin) > 0.0:
                    route = route & (router_margin >= np.float32(float(margin)))
                if not route.any():
                    continue
                p_local = m.predict_proba(X[route]).astype(np.float32, copy=False)
                choose = p_local.argmax(axis=1).astype(np.int64, copy=False)
                mapped = cids[choose].astype(np.int64, copy=False)
                _debug_report_cic_stage2_shape(
                    "C",
                    "model.py:_predict_multiclass:twolevel",
                    "twolevel route triggered",
                    {
                        "route_count": int(route.sum()),
                        "gid": gid,
                        "cids": [int(v) for v in cids.tolist()],
                        "p_local_shape": list(np.shape(p_local)),
                        "choose_shape": list(np.shape(choose)),
                        "mapped_shape": list(np.shape(mapped)),
                    },
                )
                _overwrite_prob_with_local_subset(route, cids, p_local)
                pred = pred.copy()
                pred[np.where(route)[0]] = mapped

    return pred, prob

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def make_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def supervised_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if class_weights is None:
        return F.cross_entropy(logits, targets)
    return F.cross_entropy(logits, targets, weight=class_weights.to(logits.device))


def fixmatch_unsup_loss(
    logits_weak: torch.Tensor,
    logits_strong: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits_weak.detach(), dim=1)
    conf, pseudo = probs.max(dim=1)
    mask = conf.ge(threshold).float()
    loss_per_sample = F.cross_entropy(logits_strong, pseudo, reduction="none")
    loss = (loss_per_sample * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, mask, pseudo

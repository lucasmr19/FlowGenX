"""
training/lr_scheduler.py
=========================
Factory de schedulers de learning rate para el Trainer.

Soporta warmup lineal combinado con cualquiera de los schedulers
principales: cosine annealing, step decay y reduce on plateau.

Todos los schedulers se envuelven en un ``SequentialLR`` o en un
``LambdaLR`` de warmup + scheduler principal, de forma transparente.

Uso
---
>>> from training.lr_scheduler import build_scheduler
>>> scheduler = build_scheduler(
...     optimizer   = optimizer,
...     config      = training_cfg,
...     steps_per_epoch = len(train_loader),
... )
>>> # En el Trainer:
>>> scheduler.step()        # paso por época (todos salvo "plateau")
>>> scheduler.step(val_loss) # solo para "plateau"
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
    _LRScheduler,
)

from ..utils.logger_config import LOGGER


def build_scheduler(
    optimizer:        Optimizer,
    config,                         # TrainingConfig
    steps_per_epoch:  int   = 1,
) -> Optional[object]:
    """
    Construye y devuelve el scheduler configurado para ``optimizer``.

    Si ``config.lr_scheduler == "none"`` devuelve ``None``.

    Para "plateau" devuelve un ``ReduceLROnPlateau`` cuyo ``.step()``
    requiere la métrica de validación como argumento.

    Para el resto devuelve un scheduler estándar cuyo ``.step()``
    no requiere argumentos.

    El warmup se combina automáticamente cuando ``warmup_epochs > 0``.
    """
    name          = config.lr_scheduler.lower()
    warmup_epochs = max(0, config.warmup_epochs)
    total_epochs  = config.epochs

    if name == "none":
        return None

    # ------------------------------------------------------------------ #
    # ReduceLROnPlateau — no es compatible con SequentialLR              #
    # ------------------------------------------------------------------ #
    if name == "plateau":
        if warmup_epochs > 0:
            LOGGER.warning(
                "[LRScheduler] warmup_epochs=%d ignorado con 'plateau'.",
                warmup_epochs,
            )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode    = "min",
            factor  = config.lr_gamma,
            patience= max(3, config.early_stopping_patience // 4),
            min_lr  = config.lr_min,
        )
        LOGGER.info("[LRScheduler] ReduceLROnPlateau (factor=%.2f)", config.lr_gamma)
        return scheduler

    # ------------------------------------------------------------------ #
    # Scheduler principal                                                 #
    # ------------------------------------------------------------------ #
    if name == "cosine":
        T_max = max(1, total_epochs - warmup_epochs)
        main_sched: _LRScheduler = CosineAnnealingLR(
            optimizer,
            T_max  = T_max,
            eta_min= config.lr_min,
        )
        LOGGER.info("[LRScheduler] CosineAnnealingLR (T_max=%d, eta_min=%.1e)", T_max, config.lr_min)

    elif name == "step":
        main_sched = StepLR(
            optimizer,
            step_size = config.lr_step_size,
            gamma     = config.lr_gamma,
        )
        LOGGER.info(
            "[LRScheduler] StepLR (step_size=%d, gamma=%.2f)",
            config.lr_step_size, config.lr_gamma,
        )
    else:
        raise ValueError(
            f"lr_scheduler desconocido: '{name}'. "
            "Usa 'cosine' | 'step' | 'plateau' | 'none'."
        )

    # ------------------------------------------------------------------ #
    # Warmup lineal (opcional)                                            #
    # ------------------------------------------------------------------ #
    if warmup_epochs <= 0:
        return main_sched

    def _warmup_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0

    warmup_sched = LambdaLR(optimizer, lr_lambda=_warmup_lambda)

    combined = SequentialLR(
        optimizer,
        schedulers  = [warmup_sched, main_sched],
        milestones  = [warmup_epochs],
    )
    LOGGER.info("[LRScheduler] Warmup=%d épocas + %s", warmup_epochs, name)
    return combined


__all__ = ["build_scheduler"]
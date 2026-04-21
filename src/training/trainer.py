"""
training/trainer.py
====================
Trainer principal, completamente agnóstico al modelo generativo concreto.

Protocolos de entrenamiento
----------------------------
El Trainer detecta automáticamente el protocolo correcto consultando el
tipo de respuesta de ``model.configure_optimizers()``:

  ┌─────────────────┬───────────────────────────────────────────────────┐
  │ Modelo          │ Protocolo                                         │
  ├─────────────────┼───────────────────────────────────────────────────┤
  │ Transformer     │ STANDARD: Trainer → zero_grad → loss.backward    │
  │ DDPM            │ → clip → step                                     │
  ├─────────────────┼───────────────────────────────────────────────────┤
  │ GAN (WGAN-GP)   │ ADVERSARIAL: modelo gestiona internamente         │
  │                 │ backward+step de G y D. El Trainer solo asigna   │
  │                 │ _opt_generator / _opt_discriminator antes de      │
  │                 │ llamar a train_step() y loguea los losses.        │
  └─────────────────┴───────────────────────────────────────────────────┘

Ciclo de entrenamiento
----------------------
fit(train_loader, val_loader=None)
  ├── callbacks.on_train_start()
  ├── for epoch in range(epochs):
  │     ├── callbacks.on_epoch_start()
  │     ├── _run_train_epoch()        ← un paso por batch
  │     ├── callbacks.on_epoch_end()
  │     ├── [si val_every] _run_val_epoch()
  │     ├── callbacks.on_validation_end()
  │     ├── scheduler.step()
  │     └── [si state.stop_training] break
  └── callbacks.on_train_end()
"""

from __future__ import annotations

import random
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models_ml.base import GenerativeModel
from ..utils.logger_config import LOGGER

from .callbacks import TrainerCallback, TrainerState
from .config import TrainingConfig
from .lr_scheduler import build_scheduler


# ---------------------------------------------------------------------------
# Constante de protocolo
# ---------------------------------------------------------------------------

_PROTOCOL_STANDARD    = "standard"    # Transformer, DDPM
_PROTOCOL_ADVERSARIAL = "adversarial" # GAN


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Trainer genérico para todos los modelos generativos de FlowGenX.

    Parameters
    ----------
    model        : Instancia de GenerativeModel ya construida (build()).
    config       : TrainingConfig con todos los hiperparámetros.
    callbacks    : Lista de TrainerCallback a ejecutar durante el training.

    Ejemplo de uso
    --------------
    >>> from training.trainer import Trainer
    >>> from training.config import TrainingConfig
    >>> from training.callbacks import CheckpointCallback, EarlyStoppingCallback
    >>>
    >>> cfg = TrainingConfig(epochs=100, lr=1e-4)
    >>> trainer = Trainer(model, cfg, callbacks=[
    ...     CheckpointCallback("checkpoints/", metric="val_loss", ...),
    ...     EarlyStoppingCallback(metric="val_loss"),
    ... ])
    >>> trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model:     GenerativeModel,
        config:    TrainingConfig,
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.model     = model
        self.cfg       = config
        self.callbacks = callbacks or []
        self.state     = TrainerState()

        # Fijar semilla
        self._set_seed(config.seed)

        # AMP scaler (solo CUDA y si config.amp=True)
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None
        if config.amp and torch.cuda.is_available():
            self._scaler = torch.cuda.amp.GradScaler()
            LOGGER.info("[Trainer] AMP activado.")

        # Optimizadores y scheduler se inicializan en fit()
        self._optimizer: Optional[Union[torch.optim.Optimizer, Dict]] = None
        self._scheduler = None
        self._protocol: Optional[str] = None

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader] = None,
    ) -> TrainerState:
        """
        Ejecuta el bucle completo de entrenamiento.

        Returns
        -------
        TrainerState con las métricas y rutas del mejor checkpoint.
        """
        self._setup_optimizer_and_protocol()

        LOGGER.info(
            "[Trainer] Iniciando entrenamiento: modelo=%s  epochs=%d  "
            "protocol=%s  device=%s",
            self.model.config.name,
            self.cfg.epochs,
            self._protocol,
            self.model.device,
        )

        self._fire("on_train_start", self.state)

        for epoch in range(1, self.cfg.epochs + 1):
            self.state.epoch = epoch
            self._fire("on_epoch_start", self.state)

            # ---- entrenamiento ----
            t0 = time.time()
            train_loss = self._run_train_epoch(train_loader)
            elapsed    = time.time() - t0

            self.state.train_loss = train_loss
            self._fire("on_epoch_end", self.state)

            LOGGER.info(
                "[Trainer] epoch=%d/%d  train_loss=%.4f  (%.1fs)",
                epoch, self.cfg.epochs, train_loss, elapsed,
            )

            # ---- validación ----
            if val_loader is not None and epoch % self.cfg.val_every == 0:
                val_metrics = self._run_val_epoch(val_loader)
                self.state.val_metrics = val_metrics
                self._fire("on_validation_end", self.state)

                metrics_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                )
                LOGGER.info("[Trainer] Validación epoch=%d  %s", epoch, metrics_str)

            # ---- scheduler ----
            self._step_scheduler(self.state.val_metrics.get(self.cfg.checkpoint_metric))

            # ---- early stopping ----
            if self.state.stop_training:
                LOGGER.info("[Trainer] Early stopping activado en epoch %d.", epoch)
                break

        self._fire("on_train_end", self.state)
        LOGGER.info(
            "[Trainer] Entrenamiento finalizado. "
            "Mejor %s=%.4f en epoch %d.",
            self.cfg.checkpoint_metric,
            self.state.best_metric,
            self.state.best_epoch,
        )
        return self.state

    # ------------------------------------------------------------------
    # Época de entrenamiento
    # ------------------------------------------------------------------

    def _run_train_epoch(self, loader: DataLoader) -> float:
        self.model.train_mode()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            self._fire("on_batch_start", self.state)
            self.state.global_step += 1

            losses = self._train_step(batch)
            total_loss += losses["loss"].item() if isinstance(losses["loss"], torch.Tensor) \
                          else float(losses["loss"])
            n_batches  += 1

            self._fire("on_batch_end", self.state, losses)

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Paso de entrenamiento (dispatcher por protocolo)
    # ------------------------------------------------------------------

    def _train_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        if self._protocol == _PROTOCOL_ADVERSARIAL:
            return self._train_step_adversarial(batch)
        else:
            return self._train_step_standard(batch)

    def _train_step_standard(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Protocolo estándar: Trainer gestiona zero_grad / backward / step.
        Usado por Transformer y DDPM.
        """
        optimizer = self._optimizer

        amp_ctx = (
            torch.cuda.amp.autocast()
            if self._scaler is not None
            else nullcontext()
        )

        optimizer.zero_grad()

        with amp_ctx:
            losses = self.model.train_step(batch)

        loss = losses["loss"]

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self._scaler.unscale_(optimizer)
                self._clip_gradients()
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            loss.backward()
            if self.cfg.grad_clip:
                self._clip_gradients()
            optimizer.step()

        return losses

    def _train_step_adversarial(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Protocolo adversarial: el modelo GAN gestiona backward+step internamente.
        El Trainer solo llama train_step() y registra los losses.
        """
        losses = self.model.train_step(batch)
        return losses
    
    def _disable_optimizer_updates(self) -> None:
        """
        Desactiva temporalmente optimizer.zero_grad() y optimizer.step().

        Se usa en validación adversarial para permitir backward sin actualizar pesos.
        """
        if not isinstance(self._optimizer, dict):
            return

        self._opt_backup = {}

        for name, opt in self._optimizer.items():
            self._opt_backup[name] = {
                "zero_grad": opt.zero_grad,
                "step": opt.step,
            }

            # Reemplazar por no-ops
            opt.zero_grad = lambda *args, **kwargs: None
            opt.step      = lambda *args, **kwargs: None


    def _restore_optimizer_updates(self) -> None:
        """
        Restaura los métodos originales de los optimizers tras validación.
        """
        if not hasattr(self, "_opt_backup"):
            return

        for name, opt in self._optimizer.items():
            backup = self._opt_backup.get(name)
            if backup:
                opt.zero_grad = backup["zero_grad"]
                opt.step      = backup["step"]

        self._opt_backup = {}

    # ------------------------------------------------------------------
    # Época de validación
    # ------------------------------------------------------------------

    def _run_val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Ejecuta la época de validación y devuelve métricas agregadas.

        Para modelos secuenciales calcula NLL/perplexity promedio.
        Para modelos visuales calcula MSE promedio.
        """
        self.model.eval_mode()
        agg: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        # ---- contexto según protocolo ----
        if self._protocol == _PROTOCOL_ADVERSARIAL:
            ctx = nullcontext()
            self._disable_optimizer_updates()
        else:
            ctx = torch.no_grad()

        try:
            with ctx:
                for batch in loader:
                    losses = self.model.train_step(batch)

                    for k, v in losses.items():
                        val = v.item() if isinstance(v, torch.Tensor) else float(v)
                        agg[k]    = agg.get(k, 0.0) + val
                        counts[k] = counts.get(k, 0) + 1
        finally:
            if self._protocol == _PROTOCOL_ADVERSARIAL:
                self._restore_optimizer_updates()

        # ---- promedios ----
        result: Dict[str, float] = {}
        for k, total in agg.items():
            result[f"val_{k}"] = total / max(counts[k], 1)

        self.model.train_mode()
        return result

    # ------------------------------------------------------------------
    # Setup de optimizadores
    # ------------------------------------------------------------------

    def _setup_optimizer_and_protocol(self) -> None:
        """
        Detecta el protocolo de entrenamiento y configura los optimizadores.
        """
        opts = self.model.configure_optimizers(lr=self.cfg.lr)

        if isinstance(opts, dict):
            # GAN: múltiples optimizadores gestionados por el modelo
            self._protocol  = _PROTOCOL_ADVERSARIAL
            self._optimizer = opts
            # Asignar referencias internas que espera TrafficGAN.train_step()
            for key, opt in opts.items():
                attr = f"_opt_{key}"  # e.g., "_opt_generator", "_opt_discriminator"
                setattr(self.model, attr, opt)
            # Aplicar weight_decay manualmente si > 0
            if self.cfg.weight_decay > 0:
                for opt in opts.values():
                    for pg in opt.param_groups:
                        pg.setdefault("weight_decay", self.cfg.weight_decay)
            # El scheduler solo se construye sobre el optimizador del generador
            # (el del discriminador tiene su propio ritmo controlado por n_critic)
            gen_opt = opts.get("generator", next(iter(opts.values())))
            self._scheduler = build_scheduler(gen_opt, self.cfg)
        else:
            self._protocol  = _PROTOCOL_STANDARD
            self._optimizer = opts
            if self.cfg.weight_decay > 0:
                for pg in opts.param_groups:
                    pg.setdefault("weight_decay", self.cfg.weight_decay)
            self._scheduler = build_scheduler(opts, self.cfg)

        LOGGER.info(
            "[Trainer] Protocolo: %s  |  Scheduler: %s",
            self._protocol,
            type(self._scheduler).__name__ if self._scheduler else "None",
        )

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------

    def _clip_gradients(self) -> None:
        """Recorre todos los módulos del modelo y aplica clip por norma."""
        for net in self.model._networks.values():
            nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)

    # ------------------------------------------------------------------
    # Scheduler step
    # ------------------------------------------------------------------

    def _step_scheduler(self, metric_value: Optional[float] = None) -> None:
        if self._scheduler is None:
            return
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        if isinstance(self._scheduler, ReduceLROnPlateau):
            if metric_value is not None:
                self._scheduler.step(metric_value)
        else:
            self._scheduler.step()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _fire(self, hook: str, *args, **kwargs) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(*args, **kwargs)

    # ------------------------------------------------------------------
    # Reproducibilidad
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Trainer("
            f"model={self.model.config.name!r}, "
            f"epochs={self.cfg.epochs}, "
            f"lr={self.cfg.lr}, "
            f"protocol={self._protocol or 'not_setup'}, "
            f"callbacks=[{', '.join(type(c).__name__ for c in self.callbacks)}])"
        )


__all__ = ["Trainer"]
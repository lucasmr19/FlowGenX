"""
training/callbacks.py
======================
Sistema de callbacks para el Trainer.

Diseño inspirado en PyTorch Lightning y Keras, adaptado al contrato
del framework FlowGenX. Cada callback recibe el estado actual del
Trainer vía un objeto TrainerState, lo que permite acceder a métricas,
epoch, modelo y config sin acoplarse a la clase Trainer directamente.

Callbacks disponibles
---------------------
- CheckpointCallback      : Guarda los mejores y/o último checkpoint.
- EarlyStoppingCallback   : Para el entrenamiento si la métrica se estanca.
- MetricsLoggerCallback   : Escribe métricas en CSV y consola estructurada.
- TimingCallback          : Registra tiempos por época y total en JSON.
- TensorBoardCallback     : Escribe eventos SummaryWriter (TensorBoard).
- EMACallback             : Mantiene y aplica Exponential Moving Average.
- WandbCallback           : Integración opcional con Weights & Biases.
"""

from __future__ import annotations

import copy
import csv
import heapq
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..utils.logger_config import LOGGER


# ---------------------------------------------------------------------------
# Estado compartido del Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainerState:
    """
    Snapshot mutable del estado del Trainer en cada hook.

    El Trainer actualiza este objeto antes de invocar cada hook.
    Los callbacks pueden leer cualquier campo y escribir ``stop_training``.
    """
    epoch:       int   = 0
    global_step: int   = 0
    train_loss:  float = float("inf")
    val_metrics: Dict[str, float] = field(default_factory=dict)
    best_metric: float = float("inf")
    best_epoch:  int   = 0
    stop_training: bool = False
    checkpoint_path: Optional[str] = None  # ruta del último checkpoint guardado
    # Registro de tiempos por época (segundos)
    epoch_times: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Clase base abstracta
# ---------------------------------------------------------------------------

class TrainerCallback(ABC):
    """
    Interfaz de callback del Trainer.

    Todos los hooks tienen implementación vacía por defecto,
    de forma que las subclases solo sobreescriben los que necesitan.

    Orden de llamada por época
    --------------------------
    on_train_start()
      for epoch:
        on_epoch_start(state)
        for batch:
          on_batch_start(state)
          [train_step]
          on_batch_end(state, losses)
        on_epoch_end(state)
        [validación si val_every]
        on_validation_end(state)
    on_train_end(state)
    """

    def on_train_start(self, state: TrainerState) -> None:
        pass

    def on_epoch_start(self, state: TrainerState) -> None:
        pass

    def on_batch_start(self, state: TrainerState) -> None:
        pass

    def on_batch_end(
        self,
        state: TrainerState,
        losses: Dict[str, float],
    ) -> None:
        pass

    def on_epoch_end(self, state: TrainerState) -> None:
        pass

    def on_validation_end(self, state: TrainerState) -> None:
        pass

    def on_train_end(self, state: TrainerState) -> None:
        pass


# ---------------------------------------------------------------------------
# CheckpointCallback
# ---------------------------------------------------------------------------

class CheckpointCallback(TrainerCallback):
    """
    Guarda los mejores checkpoints (top-k) y/o el último.

    Estructura de ficheros
    ----------------------
    checkpoint_dir/
        best_epoch_007_val_loss=0.2341.pt   ← top-k mejores
        last.pt                              ← último (si save_last=True)

    Parameters
    ----------
    checkpoint_dir  : Directorio de salida.
    metric          : Clave en ``state.val_metrics`` a monitorizar.
    mode            : "min" | "max".
    save_last       : Guardar siempre el checkpoint del último epoch.
    save_top_k      : Cuántos mejores checkpoints mantener (1 = solo best).
    model           : Referencia al GenerativeModel para guardar.
    representation  : Referencia a la TrafficRepresentation (para co-guardar).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric:         str,
        mode:           str,
        model,
        representation,
        save_last:  bool = True,
        save_top_k: int  = 1,
        experiment_name: str = "exp",
    ) -> None:
        if save_top_k < 0:
            raise ValueError("save_top_k must be >= 0")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.dir    = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode   = mode
        self.model  = model
        self.rep    = representation
        self.save_last  = save_last
        self.exp_name   = experiment_name
        self.save_top_k = save_top_k
        self._best_value = float("inf") if mode == "min" else -float("inf")
        # Heap de (score, path) — min-heap
        # Para mode="max", guardamos -score
        self._top_k_heap: List[tuple] = []
    
    def _priority(self, value: float) -> float:
        # Más alto = mejor
        return -value if self.mode == "min" else value

    def on_validation_end(self, state: TrainerState) -> None:
        if self.metric not in state.val_metrics:
            return

        value = state.val_metrics[self.metric]

        # ---- best tracking independiente del top-k ----
        is_better = (
            value < self._best_value if self.mode == "min"
            else value > self._best_value
        )
        if is_better:
            self._best_value = value
            state.best_metric = value
            state.best_epoch = state.epoch

        # ---- save_last siempre que proceda ----
        last_path = None
        if self.save_last:
            last_path = self.dir / f"{self.exp_name}_last.pt"
            self._save(last_path, state)

        # ---- top-k desactivado ----
        if self.save_top_k == 0:
            if last_path is not None:
                state.checkpoint_path = str(last_path)
            return

        priority = self._priority(value)

        fname = (
            f"{self.exp_name}_epoch{state.epoch:04d}"
            f"_{self.metric}={value:.4f}.pt"
        )
        path = self.dir / fname

        # ---- guardar si entra en top-k ----
        if len(self._top_k_heap) < self.save_top_k:
            self._save(path, state)
            heapq.heappush(self._top_k_heap, (priority, str(path)))
            state.checkpoint_path = str(path)

        elif priority > self._top_k_heap[0][0]:
            # El peor de los guardados es heap[0]
            _, old_path = heapq.heapreplace(self._top_k_heap, (priority, str(path)))
            _safe_remove(old_path)
            self._save(path, state)
            state.checkpoint_path = str(path)

    def _save(self, path: Path, state: TrainerState) -> None:
        self.model.save(path)
        rep_path = path.with_suffix(".rep.pt")
        self.rep.save(rep_path)
        LOGGER.info(
            "[Checkpoint] epoch=%d  %s=%.4f → %s",
            state.epoch,
            self.metric,
            state.val_metrics.get(self.metric, float("nan")),
            path,
        )


# ---------------------------------------------------------------------------
# EarlyStoppingCallback
# ---------------------------------------------------------------------------

class EarlyStoppingCallback(TrainerCallback):
    """
    Para el entrenamiento cuando la métrica monitoreada deja de mejorar.

    Parameters
    ----------
    metric    : Clave en ``state.val_metrics``.
    mode      : "min" | "max".
    patience  : Épocas sin mejora antes de parar.
    min_delta : Cambio mínimo considerado mejora.
    """

    def __init__(
        self,
        metric:    str,
        mode:      str   = "min",
        patience:  int   = 20,
        min_delta: float = 1e-4,
    ) -> None:
        self.metric    = metric
        self.mode      = mode
        self.patience  = patience
        self.min_delta = min_delta

        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._wait: int   = 0

    def on_validation_end(self, state: TrainerState) -> None:
        if self.metric not in state.val_metrics:
            return

        current = state.val_metrics[self.metric]
        improved = (
            (current < self._best - self.min_delta)
            if self.mode == "min"
            else (current > self._best + self.min_delta)
        )

        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            LOGGER.debug(
                "[EarlyStopping] %s no mejoró (%d/%d).",
                self.metric, self._wait, self.patience,
            )

        if self._wait >= self.patience:
            LOGGER.info(
                "[EarlyStopping] Deteniendo entrenamiento en epoch %d. "
                "Mejor %s=%.4f",
                state.epoch, self.metric, self._best,
            )
            state.stop_training = True


# ---------------------------------------------------------------------------
# MetricsLoggerCallback
# ---------------------------------------------------------------------------

class MetricsLoggerCallback(TrainerCallback):
    """
    Registra todas las métricas por época en un CSV y en consola.

    Fichero de salida: ``{log_dir}/{experiment_name}_metrics.csv``

    El CSV se escribe completo al final del entrenamiento para garantizar
    que el header refleje todas las columnas posibles (incluyendo las
    métricas de validación que solo aparecen a partir de val_every > 1).
    """

    def __init__(
        self,
        log_dir:         str,
        experiment_name: str = "exp",
        log_every:       int = 20,
    ) -> None:
        self.log_dir   = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.exp_name  = experiment_name

        self._csv_path = self.log_dir / f"{experiment_name}_metrics.csv"

        # Buffer: acumulamos filas y columnas hasta on_train_end
        self._rows:     list = []
        self._all_keys: list = ["epoch", "train_loss"]

        # Buffer para promedios de batch dentro de la época
        self._batch_losses: Dict[str, List[float]] = {}

    def on_train_start(self, state: TrainerState) -> None:
        # Reset por si el callback se reutiliza
        self._rows      = []
        self._all_keys  = ["epoch", "train_loss"]
        self._batch_losses = {}

    def on_batch_end(
        self,
        state: TrainerState,
        losses: Dict[str, float],
    ) -> None:
        for k, v in losses.items():
            self._batch_losses.setdefault(k, []).append(
                v.item() if isinstance(v, torch.Tensor) else float(v)
            )

        if state.global_step % self.log_every == 0:
            avg = {k: sum(vs) / len(vs) for k, vs in self._batch_losses.items()}
            self._batch_losses.clear()
            parts = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            LOGGER.info(
                "step=%d  epoch=%d  %s",
                state.global_step, state.epoch, parts,
            )

    def on_epoch_end(self, state: TrainerState) -> None:
        row: Dict[str, str] = {
            "epoch":      state.epoch,
            "train_loss": f"{state.train_loss:.6f}",
        }
        row.update(
            {k: f"{v:.6f}" for k, v in state.val_metrics.items()}
        )

        # Registrar columnas nuevas (mantiene orden de aparición)
        for k in row:
            if k not in self._all_keys:
                self._all_keys.append(k)

        self._rows.append(row)

    def on_train_end(self, state: TrainerState) -> None:
        """Escribe el CSV completo con header correcto al terminar el entrenamiento."""
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames   = self._all_keys,
                extrasaction = "ignore",
                restval      = "",          # columnas ausentes → celda vacía
            )
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)
        LOGGER.info("[Metrics] Log guardado en %s", self._csv_path)


# ---------------------------------------------------------------------------
# TensorBoardCallback
# ---------------------------------------------------------------------------

class TensorBoardCallback(TrainerCallback):
    """
    Escribe métricas de entrenamiento a TensorBoard.

    Requiere ``tensorboard`` instalado (``pip install tensorboard``).
    Los logs se escriben en ``{log_dir}/tb_logs/{experiment_name}/``.
    """

    def __init__(self, log_dir: str, experiment_name: str = "exp") -> None:
        from torch.utils.tensorboard import SummaryWriter
        tb_path = Path(log_dir)
        # Si por alguna razón el path apunta a un fichero, usamos un sufijo
        if tb_path.exists() and not tb_path.is_dir():
            tb_path = tb_path.parent / (tb_path.name + "_tb")
        tb_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_path))
        LOGGER.info("[TensorBoard] Logs en %s", tb_path)

    def on_batch_end(
        self,
        state: TrainerState,
        losses: Dict[str, float],
    ) -> None:
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            self.writer.add_scalar(f"train/{k}", val, state.global_step)

    def on_validation_end(self, state: TrainerState) -> None:
        for k, v in state.val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, state.epoch)

    def on_train_end(self, state: TrainerState) -> None:
        self.writer.close()


# ---------------------------------------------------------------------------
# EMACallback
# ---------------------------------------------------------------------------

class EMACallback(TrainerCallback):
    """
    Mantiene una copia EMA (Exponential Moving Average) de los parámetros
    del modelo. Al final del entrenamiento aplica los pesos EMA al modelo.

    Útil especialmente para GAN y DDPM donde la calidad de generación
    mejora notablemente con EMA (Karras et al., 2022; Song et al., 2020).

    Parameters
    ----------
    model     : GenerativeModel.
    decay     : Factor de decaimiento (0.999 es estándar).
    start_epoch: Época a partir de la cual empezar a acumular EMA.
    apply_on_end: Aplicar los pesos EMA al modelo al final del training.
    """

    def __init__(
        self,
        model,
        decay:         float = 0.999,
        start_epoch:   int   = 10,
        apply_on_end:  bool  = True,
    ) -> None:
        self.model        = model
        self.decay        = decay
        self.start_epoch  = start_epoch
        self.apply_on_end = apply_on_end
        self._shadow: Optional[Dict[str, torch.Tensor]] = None

    def _collect_params(self) -> Dict[str, torch.Tensor]:
        params = {}
        for name, net in self.model._networks.items():
            for pname, param in net.named_parameters():
                params[f"{name}.{pname}"] = param.data.clone()
        return params

    def _update_shadow(self) -> None:
        current = self._collect_params()
        if self._shadow is None:
            self._shadow = {k: v.clone() for k, v in current.items()}
            return
        for k, v in current.items():
            self._shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def on_epoch_end(self, state: TrainerState) -> None:
        if state.epoch >= self.start_epoch:
            self._update_shadow()

    def on_train_end(self, state: TrainerState) -> None:
        if not self.apply_on_end or self._shadow is None:
            return
        LOGGER.info("[EMA] Aplicando pesos EMA al modelo (decay=%.4f).", self.decay)
        for name, net in self.model._networks.items():
            for pname, param in net.named_parameters():
                key = f"{name}.{pname}"
                if key in self._shadow:
                    param.data.copy_(self._shadow[key])


# ---------------------------------------------------------------------------
# WandbCallback
# ---------------------------------------------------------------------------

class WandbCallback(TrainerCallback):
    """
    Integración con Weights & Biases.

    Requiere ``wandb`` instalado y autenticado (``wandb login``).

    Parameters
    ----------
    project    : Nombre del proyecto W&B.
    run_name   : Nombre del run (None = auto).
    config_dict: Dict de hyperparámetros a loguear en W&B.
    tags       : Lista de tags para filtrar runs.
    """

    def __init__(
        self,
        project:     str,
        run_name:    Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        tags:        Optional[List[str]] = None,
    ) -> None:
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError("wandb no encontrado. Instala con: pip install wandb")

        self._wandb.init(
            project = project,
            name    = run_name,
            config  = config_dict or {},
            tags    = tags or [],
        )

    def on_batch_end(
        self,
        state: TrainerState,
        losses: Dict[str, float],
    ) -> None:
        log = {f"train/{k}": (v.item() if isinstance(v, torch.Tensor) else float(v))
               for k, v in losses.items()}
        log["global_step"] = state.global_step
        self._wandb.log(log, step=state.global_step)

    def on_validation_end(self, state: TrainerState) -> None:
        log = {f"val/{k}": v for k, v in state.val_metrics.items()}
        log["epoch"] = state.epoch
        self._wandb.log(log, step=state.global_step)

    def on_train_end(self, state: TrainerState) -> None:
        self._wandb.finish()



# ---------------------------------------------------------------------------
# TimingCallback
# ---------------------------------------------------------------------------

class TimingCallback(TrainerCallback):
    """
    Registra el tiempo de entrenamiento por época y el total acumulado.

    Al finalizar el entrenamiento guarda un JSON estructurado en
    ``{log_dir}/{experiment_name}_timing.json`` con:
        - epoch_times: lista de segundos por época
        - total_training_s: tiempo total de entrenamiento
        - mean_epoch_s: media de segundos por época
        - median_epoch_s: mediana de segundos por época

    Los tiempos también se guardan en ``state.epoch_times`` para que
    otros callbacks y el ExperimentRunner puedan acceder a ellos.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "exp",
    ) -> None:
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name = experiment_name
        self._json_path = self.log_dir / f"{experiment_name}_timing.json"
        self._epoch_start: float = 0.0

    def on_epoch_start(self, state: TrainerState) -> None:
        import time
        self._epoch_start = time.time()

    def on_epoch_end(self, state: TrainerState) -> None:
        import time
        elapsed = round(time.time() - self._epoch_start, 3)
        state.epoch_times.append(elapsed)

    def on_train_end(self, state: TrainerState) -> None:
        import json, statistics
        times = state.epoch_times
        summary = {
            "epoch_times": times,
            "total_training_s": round(sum(times), 3),
            "mean_epoch_s":   round(statistics.mean(times), 3)   if times else 0,
            "median_epoch_s": round(statistics.median(times), 3) if times else 0,
            "min_epoch_s":    round(min(times), 3)                if times else 0,
            "max_epoch_s":    round(max(times), 3)                if times else 0,
        }
        with open(self._json_path, "w") as f:
            json.dump(summary, f, indent=2)
        LOGGER.info("[Timing] JSON guardado en %s", self._json_path)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "TrainerState",
    "TrainerCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "MetricsLoggerCallback",
    "TensorBoardCallback",
    "EMACallback",
    "WandbCallback",
    "TimingCallback",
]
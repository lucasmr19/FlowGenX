"""
training/experiment.py
=======================
Orquestador de alto nivel: YAML → datos → modelo → entrenamiento → artefactos.

``ExperimentRunner`` une todos los componentes del framework:
  1. Parsea el YAML de experimento.
  2. Construye la representación y valida su compatibilidad con el modelo.
  3. Instancia el TrafficDataModule y materializa los splits.
  4. Construye el Trainer con los callbacks adecuados.
  5. Lanza el entrenamiento.
  6. (Opcional) Ejecuta la suite de evaluación sobre el test set.
  7. Guarda un resumen JSON del experimento.

Uso rápido
----------
>>> from training.experiment import ExperimentRunner
>>> runner = ExperimentRunner("src/configs/exp_ddpm_nprint.yaml", "data/pcap/")
>>> result = runner.run()

Uso programático
----------------
>>> runner = ExperimentRunner.from_dict(
...     config_dict={
...         "representation": {"representation_type": "nprint_image"},
...         "model":          {"model_type": "ddpm"},
...         "training":       {"epochs": 50, "lr": 1e-4},
...         "data":           {"data_dir": "data/pcap/"},
...     }
... )
>>> result = runner.run()
"""

from __future__ import annotations

import json
import time
import yaml
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..core.config_loader import load_yaml
from ..core.builders import build_model_from_dict, build_representation_from_dict
from ..data_utils import TrafficDataModule, build_datamodule_from_dir
from ..models_ml.base import GenerativeModel
from ..representations.base import TrafficRepresentation
from ..utils.logger_config import LOGGER

from .callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    EMACallback,
    MetricsLoggerCallback,
    TensorBoardCallback,
    TrainerCallback,
    WandbCallback,
)
from .config import TrainingConfig
from .trainer import Trainer


# ---------------------------------------------------------------------------
# Resultado del experimento
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    Resumen completo de un experimento de entrenamiento.

    Serializable a JSON vía ``to_dict()`` para reproducibilidad y
    trazabilidad en trabajos de investigación.
    """
    run_dir: str
    experiment_name:   str
    representation:    str
    model:             str
    best_epoch:        int
    best_metric_name:  str
    best_metric_value: float
    total_epochs_ran:  int
    training_time_s:   float
    checkpoint_path:   Optional[str]
    val_metrics:       Dict[str, float] = field(default_factory=dict)
    extra:             Dict[str, Any]   = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info("[ExperimentResult] Guardado en %s", path)


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Orquestador de experimento completo para FlowGenX.

    Parameters
    ----------
    yaml_path       : Ruta al YAML de configuración del experimento.
    data_dir        : Directorio raíz con las carpetas de .pcap por clase
                      (estructura: root/Benign/*.pcap, root/Malware/*.pcap…).
                      Si es None, se lee desde ``data.data_dir`` del YAML.
    experiment_name : Nombre del experimento (default = stem del YAML).
    extra_callbacks : Callbacks adicionales a los que se construyen desde config.
    device          : Dispositivo explícito ("cpu" | "cuda"). None = auto.

    YAML esperado
    -------------
    .. code-block:: yaml

        experiment_name: exp_transformer_flat   # opcional

        representation:
          representation_type: flat_tokenizer
          max_length: 512

        model:
          model_type: transformer
          d_model: 256
          n_layers: 6
          n_heads: 8

        training:
          epochs: 100
          batch_size: 32
          lr: 1e-4
          lr_scheduler: cosine
          warmup_epochs: 5
          checkpoint_dir: checkpoints/
          checkpoint_metric: val_loss
          early_stopping: true
          early_stopping_patience: 15
          use_tensorboard: false
          use_wandb: false
          use_ema: false

        data:
          data_dir: data/pcap/
          max_packets: 500       # opcional, para limitar PCAPs en dev
          protocols: null        # null = todos
    """

    def __init__(
        self,
        yaml_path:        str,
        data_dir:         Optional[str] = None,
        experiment_name:  Optional[str] = None,
        extra_callbacks:  Optional[List[TrainerCallback]] = None,
        device:           Optional[str] = None,
    ) -> None:
        self._raw_cfg = load_yaml(yaml_path)
        self._yaml_stem = Path(yaml_path).stem

        self._data_dir       = data_dir
        self._exp_name       = (
            experiment_name
            or self._raw_cfg.get("experiment_name")
            or self._yaml_stem
        )
        self._extra_callbacks = extra_callbacks or []
        self._device          = device

    @classmethod
    def from_dict(
        cls,
        config_dict:     Dict[str, Any],
        data_dir:        Optional[str] = None,
        experiment_name: str = "exp",
        extra_callbacks: Optional[List[TrainerCallback]] = None,
        device:          Optional[str] = None,
    ) -> "ExperimentRunner":
        """Crea un ExperimentRunner desde un dict en lugar de un YAML."""
        import tempfile, yaml
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f)
            tmp_path = f.name
        runner = cls(
            tmp_path, data_dir, experiment_name, extra_callbacks, device
        )
        runner._raw_cfg = config_dict
        return runner

    # ------------------------------------------------------------------
    # Punto de entrada
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResult:
        """
        Ejecuta el experimento completo.

        Returns
        -------
        ExperimentResult con las métricas finales y la ruta del checkpoint.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path("experiments/runs") / f"{timestamp}_{self._exp_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = run_dir / "metrics"
        logs_dir = run_dir / "logs"
        plots_dir = run_dir / "plots"

        metrics_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)
        
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg_raw, f)
        
        t_start = time.time()

        cfg_raw = self._raw_cfg

        # 1. Parsear configs
        train_cfg = TrainingConfig.from_dict(cfg_raw.get("training", {}))
        train_cfg.checkpoint_dir = str(run_dir / "checkpoints")
        data_cfg  = cfg_raw.get("data", {})
        rep_cfg   = cfg_raw.get("representation", {})
        model_cfg = cfg_raw.get("model", {})

        # 2. Dispositivo
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg.setdefault("device", device)

        # 3. Representación
        LOGGER.info("[ExperimentRunner] Construyendo representación: %s", rep_cfg.get("representation_type"))
        rep: TrafficRepresentation = build_representation_from_dict(rep_cfg)

        # 4. Datos
        data_dir = Path(self._data_dir or data_cfg.get("data_dir", "data/pcap/"))
        LOGGER.info("[ExperimentRunner] Cargando datos desde %s", data_dir)

        dm: TrafficDataModule = build_datamodule_from_dir(
            root_path          = data_dir,
            representation     = rep,
            aggregator         = rep.default_aggregator(),
            max_packets        = data_cfg.get("max_packets", None),
            protocols          = data_cfg.get("protocols", None),
            max_payload_bytes  = data_cfg.get("max_payload_bytes", 20),
            train_ratio        = train_cfg.train_ratio,
            val_ratio          = train_cfg.val_ratio,
            test_ratio         = train_cfg.test_ratio,
            batch_size         = train_cfg.batch_size,
            num_workers        = train_cfg.num_workers,
            seed               = train_cfg.seed,
        )
        dm.setup()

        summary = dm.summary()
        LOGGER.info(
            "[ExperimentRunner] DataModule: train=%d  val=%d  test=%d",
            summary["train_samples"],
            summary["val_samples"],
            summary["test_samples"],
        )

        # 5. Sincronizar vocab_size representación → modelo
        #    (Solo para modelos secuenciales que necesitan vocab_size)
        vocab_size = getattr(rep, "vocab_size", None)
        if vocab_size is not None and "vocab_size" not in model_cfg:
            model_cfg["vocab_size"] = vocab_size
            LOGGER.info(
                "[ExperimentRunner] vocab_size=%d sincronizado desde la representación.",
                vocab_size,
            )

        # 6. Modelo
        LOGGER.info("[ExperimentRunner] Construyendo modelo: %s", model_cfg.get("model_type"))
        model: GenerativeModel = build_model_from_dict(model_cfg)
        model.build()

        params = model.get_num_parameters()
        LOGGER.info(
            "[ExperimentRunner] Parámetros totales: %s",
            f"{params.get('total', 0):,}",
        )

        # 7. Callbacks
        callbacks = self._build_callbacks(train_cfg, model, rep, run_dir)
        callbacks.extend(self._extra_callbacks)

        # 8. Trainer
        trainer = Trainer(model=model, config=train_cfg, callbacks=callbacks)

        LOGGER.info(
            "[ExperimentRunner] Iniciando experimento '%s'  (%s + %s)",
            self._exp_name,
            rep_cfg.get("representation_type"),
            model_cfg.get("model_type"),
        )

        state = trainer.fit(
            train_loader = dm.train_dataloader(),
            val_loader   = dm.val_dataloader(),
        )

        elapsed = time.time() - t_start

        # 9. Resultado
        result = ExperimentResult(
            run_dir=str(run_dir),
            experiment_name   = self._exp_name,
            representation    = rep_cfg.get("representation_type", "unknown"),
            model             = model_cfg.get("model_type", "unknown"),
            best_epoch        = state.best_epoch,
            best_metric_name  = train_cfg.checkpoint_metric,
            best_metric_value = state.best_metric,
            total_epochs_ran  = state.epoch,
            training_time_s   = round(elapsed, 2),
            checkpoint_path   = state.checkpoint_path,
            val_metrics       = state.val_metrics,
            extra             = {"n_params": params.get("total", 0)},
        )

        # Guardar resumen JSON junto a los checkpoints
        ckpt_dir = Path(train_cfg.checkpoint_dir)
        result.save_json(run_dir / "result.json")

        LOGGER.info(
            "[ExperimentRunner] ✓ Experimento '%s' completado en %.1fs.",
            self._exp_name, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Construcción de callbacks
    # ------------------------------------------------------------------

    def _build_callbacks(
        self,
        train_cfg:  TrainingConfig,
        model:      GenerativeModel,
        rep:        TrafficRepresentation,
        run_dir: Path,
    ) -> List[TrainerCallback]:
        callbacks: List[TrainerCallback] = []

        # ---- métricas a CSV ----
        callbacks.append(
            MetricsLoggerCallback(
                log_dir         = str(run_dir / "metrics"),
                experiment_name = self._exp_name,
                log_every       = train_cfg.log_every,
            )
        )

        # ---- checkpoint ----
        callbacks.append(
            CheckpointCallback(
                checkpoint_dir  = str(run_dir / "checkpoints"),
                metric          = train_cfg.checkpoint_metric,
                mode            = train_cfg.checkpoint_mode,
                model           = model,
                representation  = rep,
                save_last       = train_cfg.save_last,
                save_top_k      = train_cfg.save_top_k,
                experiment_name = self._exp_name,
            )
        )

        # ---- early stopping ----
        if train_cfg.early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    metric    = train_cfg.early_stopping_metric,
                    mode      = train_cfg.checkpoint_mode,
                    patience  = train_cfg.early_stopping_patience,
                    min_delta = train_cfg.early_stopping_delta,
                )
            )

        # ---- EMA ----
        if train_cfg.use_ema:
            callbacks.append(
                EMACallback(
                    model       = model,
                    decay       = train_cfg.ema_decay,
                    start_epoch = train_cfg.ema_start,
                )
            )

        # ---- TensorBoard ----
        if train_cfg.use_tensorboard:
            try:
                callbacks.append(
                    TensorBoardCallback(
                        log_dir         = str(run_dir / "logs"),
                        experiment_name = self._exp_name,
                    )
                )
            except ImportError:
                LOGGER.warning("[ExperimentRunner] TensorBoard no disponible.")

        # ---- W&B ----
        if train_cfg.use_wandb:
            try:
                callbacks.append(
                    WandbCallback(
                        project    = train_cfg.wandb_project,
                        run_name   = train_cfg.wandb_run_name or self._exp_name,
                        config_dict= self._raw_cfg,
                        tags       = train_cfg.extra_tags,
                    )
                )
            except ImportError:
                LOGGER.warning("[ExperimentRunner] wandb no disponible.")

        return callbacks


# ---------------------------------------------------------------------------
# Función de conveniencia
# ---------------------------------------------------------------------------

def run_experiment(
    yaml_path:  str,
    data_dir:   Optional[str] = None,
    device:     Optional[str] = None,
) -> ExperimentResult:
    """
    Shortcut para lanzar un experimento desde una sola línea.

    >>> result = run_experiment("src/configs/exp_ddpm_nprint.yaml")
    """
    runner = ExperimentRunner(yaml_path, data_dir=data_dir, device=device)
    return runner.run()


__all__ = ["ExperimentRunner", "ExperimentResult", "run_experiment"]
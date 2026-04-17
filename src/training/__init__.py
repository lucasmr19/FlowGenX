"""
training/
=========
Módulo de entrenamiento de FlowGenX.

Exporta el Trainer, la configuración, los callbacks estándar y
el orquestador de experimentos.

Uso rápido
----------
>>> from src.training import ExperimentRunner
>>> result = ExperimentRunner("src/configs/exp_transformer_flat.yaml").run()

Uso avanzado
------------
>>> from src.training import Trainer, TrainingConfig
>>> from src.training.callbacks import CheckpointCallback, EarlyStoppingCallback
>>>
>>> cfg     = TrainingConfig(epochs=50, lr=3e-4, use_ema=True)
>>> trainer = Trainer(model, cfg, callbacks=[
...     CheckpointCallback("ckpts/", metric="val_loss", mode="min",
...                        model=model, representation=rep),
...     EarlyStoppingCallback(metric="val_loss"),
... ])
>>> state = trainer.fit(train_loader, val_loader)
"""

from .config    import TrainingConfig
from .trainer   import Trainer
from .experiment import ExperimentRunner, ExperimentResult, run_experiment
from .callbacks import (
    TrainerState,
    TrainerCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricsLoggerCallback,
    TensorBoardCallback,
    EMACallback,
    WandbCallback,
)
from .lr_scheduler import build_scheduler

__all__ = [
    # Config
    "TrainingConfig",
    # Trainer
    "Trainer",
    # Experiment
    "ExperimentRunner",
    "ExperimentResult",
    "run_experiment",
    # Callbacks
    "TrainerState",
    "TrainerCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "MetricsLoggerCallback",
    "TensorBoardCallback",
    "EMACallback",
    "WandbCallback",
    # Utils
    "build_scheduler",
]
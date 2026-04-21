"""
training/config.py
==================
Dataclass de configuración para el módulo de entrenamiento.

Todos los hiperparámetros de training se definen aquí y se cargan
directamente desde la sección ``training:`` de los YAMLs de experimento,
manteniendo un contrato unificado entre todos los modelos generativos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    """
    Parámetros de entrenamiento agnósticos al modelo.

    Diseñados para mapear 1:1 con la sección ``training:`` del YAML.
    Los valores por defecto son razonables para un experimento base.

    Parámetros generales
    --------------------
    epochs          : Número total de épocas.
    batch_size      : Tamaño de batch para los DataLoaders.
    lr              : Learning rate inicial.
    weight_decay    : Regularización L2 del optimizador.
    grad_clip       : Norma máxima del gradiente (None = desactivado).
    seed            : Semilla de reproducibilidad.
    num_workers     : Workers del DataLoader.
    amp             : Activar Automatic Mixed Precision (AMP).
    deterministic   : Forzar modo determinista en CUDA (más lento).

    LR Scheduler
    ------------
    lr_scheduler    : "cosine" | "step" | "plateau" | "none"
    warmup_epochs   : Épocas de warmup lineal antes del scheduler principal.
    lr_step_size    : Paso del StepLR (si lr_scheduler="step").
    lr_gamma        : Factor multiplicativo de StepLR / ReduceLROnPlateau.
    lr_min          : LR mínimo para CosineAnnealingLR.

    Checkpointing
    -------------
    checkpoint_dir    : Directorio de salida para checkpoints.
    checkpoint_metric : Métrica a monitorizar (ej: "val_loss", "fid").
    checkpoint_mode   : "min" | "max" según si la métrica es mejor menor.
    save_last         : Guardar siempre el último checkpoint.
    save_top_k        : Cuántos mejores checkpoints conservar.

    Early Stopping
    --------------
    early_stopping           : Activar early stopping.
    early_stopping_patience  : Épocas sin mejora antes de parar.
    early_stopping_delta     : Cambio mínimo considerado mejora.
    early_stopping_metric    : Métrica a monitorizar (None = checkpoint_metric).

    Logging
    -------
    log_every       : Loguear cada N batches.
    val_every       : Ejecutar validación cada N épocas.
    use_tensorboard : Escribir eventos TensorBoard en checkpoint_dir/tb_logs/.
    use_wandb       : Activar integración con Weights & Biases.
    wandb_project   : Nombre del proyecto W&B.
    wandb_run_name  : Nombre del run W&B (None = auto).
    extra_tags      : Tags libres para W&B / metadatos.

    EMA (Exponential Moving Average)
    ---------------------------------
    use_ema     : Activar EMA sobre los pesos del generador.
    ema_decay   : Factor de decaimiento (0.999 es estándar para generación).
    ema_start   : Época desde la que empezar a acumular EMA.

    GAN-específico
    --------------
    Heredado del ``GANConfig``: ``n_critic``, ``lambda_gp``.
    El Trainer detecta automáticamente si el modelo es GAN y aplica
    el protocolo multi-optimizador correspondiente.
    """

    # ------------------------------------------------------------------ #
    # General                                                             #
    # ------------------------------------------------------------------ #
    epochs:        int   = 100
    batch_size:    int   = 32
    lr:            float = 1e-4
    weight_decay:  float = 1e-5
    grad_clip:     Optional[float] = 1.0
    seed:          int   = 42
    num_workers:   int   = 0
    amp:           bool  = False
    deterministic: bool  = False

    # ------------------------------------------------------------------ #
    # LR Scheduler                                                        #
    # ------------------------------------------------------------------ #
    lr_scheduler:  str   = "cosine"   # cosine | step | plateau | none
    warmup_epochs: int   = 5
    lr_step_size:  int   = 30         # para StepLR
    lr_gamma:      float = 0.5        # para StepLR / ReduceLROnPlateau
    lr_min:        float = 1e-6       # para CosineAnnealingLR

    # ------------------------------------------------------------------ #
    # Checkpointing                                                       #
    # ------------------------------------------------------------------ #
    checkpoint_dir:    str  = "checkpoints"
    checkpoint_metric: str  = "val_loss"
    checkpoint_mode:   str  = "min"       # "min" | "max"
    save_last:         bool = True
    save_top_k:        int  = 0

    # ------------------------------------------------------------------ #
    # Early stopping                                                      #
    # ------------------------------------------------------------------ #
    early_stopping:          bool  = True
    early_stopping_patience: int   = 20
    early_stopping_delta:    float = 1e-4
    early_stopping_metric:   Optional[str] = None  # None = checkpoint_metric

    # ------------------------------------------------------------------ #
    # Logging                                                             #
    # ------------------------------------------------------------------ #
    log_every:      int  = 20
    val_every:      int  = 1
    use_tensorboard: bool = False
    use_wandb:       bool = False
    wandb_project:   str  = "flowgenx"
    wandb_run_name:  Optional[str] = None
    extra_tags:      List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # EMA                                                                 #
    # ------------------------------------------------------------------ #
    use_ema:    bool  = False
    ema_decay:  float = 0.999
    ema_start:  int   = 10           # época desde la que acumular EMA

    # ------------------------------------------------------------------ #
    # Data splits                                                         #
    # ------------------------------------------------------------------ #
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15

    # ------------------------------------------------------------------ #
    # Metadatos libres                                                    #
    # ------------------------------------------------------------------ #
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Validación                                                          #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if self.checkpoint_mode not in ("min", "max"):
            raise ValueError(
                f"checkpoint_mode debe ser 'min' o 'max', got '{self.checkpoint_mode}'"
            )
        if self.lr_scheduler not in ("cosine", "step", "plateau", "none"):
            raise ValueError(
                f"lr_scheduler debe ser 'cosine'|'step'|'plateau'|'none', "
                f"got '{self.lr_scheduler}'"
            )
        if not (0 < self.train_ratio + self.val_ratio + self.test_ratio <= 1.0 + 1e-6):
            raise ValueError("Los ratios de split no suman a 1.0")

        # Normalizar ruta
        self.checkpoint_dir = str(Path(self.checkpoint_dir))

        # Si no se especifica la métrica de early stopping, usar la del checkpoint
        if self.early_stopping_metric is None:
            self.early_stopping_metric = self.checkpoint_metric

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Instancia TrainingConfig desde un dict (sección 'training' del YAML)."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered   = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
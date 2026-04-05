"""
generative_models/base.py
=========================
Clase base abstracta para todos los modelos generativos del framework.

Define el contrato que deben cumplir los modelos generativos como Transformer, DDPM y GAN:
  - train_step : un paso de optimización, devuelve dict de losses
  - generate   : genera n muestras sintéticas como Tensor
  - save/load  : persistencia del estado del modelo

La firma unificada de train_step() y generate() permite que
training/trainer.py sea completamente agnóstico al modelo concreto.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.logger_config import LOGGER


# ---------------------------------------------------------------------------
# Enumeraciones
# ---------------------------------------------------------------------------

class ModelType(Enum):
    AUTOREGRESSIVE = auto()   # Transformer GPT-style
    DIFFUSION      = auto()   # DDPM / score-based
    GAN            = auto()   # Adversarial
    VAE            = auto()   # Variational autoencoder


class InputDomain(Enum):
    """Dominio de entrada que acepta el modelo."""
    DISCRETE_SEQUENCE  = auto()   # tokens enteros (B, L)
    CONTINUOUS_IMAGE   = auto()   # imágenes float (B, C, H, W)
    CONTINUOUS_SEQUENCE = auto()  # series temporales float (B, L, F)


# ---------------------------------------------------------------------------
# Configuración base
# ---------------------------------------------------------------------------

@dataclass
class GenerativeModelConfig:
    """
    Parámetros comunes a todos los modelos generativos.
    Cada subclase extiende esta dataclass.
    """
    name: str = "base_model"

    # Dimensión de entrada (se valida contra la representación en runtime)
    input_shape: Optional[Tuple[int, ...]] = None

    # Dispositivo de entrenamiento
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Semilla de reproducibilidad
    seed: int = 42

    # Gradient clipping (None = desactivado)
    grad_clip: Optional[float] = 1.0

    # Metadatos libres
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clase base abstracta
# ---------------------------------------------------------------------------

class GenerativeModel(ABC):
    """
    Contrato base para todos los modelos generativos del framework.

    Ciclo de vida
    -------------
    1. Instanciar con config.
    2. Llamar a build() para inicializar capas y moverlas al device.
    3. Iterar: optimizer.zero_grad() → losses = train_step(batch)
               → losses["loss"].backward() → optimizer.step()
    4. Generar: samples = model.generate(n=100)
    5. Persistir: model.save(path) / GenerativeModel.load(path)

    Nota sobre train_step
    ---------------------
    Devuelve un dict con al menos la clave "loss" (el escalar a backpropagar).
    Pueden incluirse losses adicionales para logging:
      {"loss": tensor, "recon_loss": tensor, "kl_loss": tensor, ...}
    El Trainer solo hace .backward() sobre "loss"; el resto se loguea.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._built = False

        # Las subclases deben asignar self._networks en build()
        # para que save/load funcionen automáticamente
        self._networks: Dict[str, nn.Module] = {}

    # ------------------------------------------------------------------
    # Propiedades abstractas
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Tipo de modelo generativo."""

    @property
    @abstractmethod
    def input_domain(self) -> InputDomain:
        """Dominio de entrada que acepta el modelo."""

    # ------------------------------------------------------------------
    # Métodos abstractos
    # ------------------------------------------------------------------

    @abstractmethod
    def build(self) -> "GenerativeModel":
        """
        Inicializa todas las capas nn.Module y las mueve al device.
        Debe poblar self._networks con todos los módulos a persistir.

        Returns self para encadenamiento: model.build().to(device)
        """

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, Tensor]:
        """
        Ejecuta un paso de entrenamiento FORWARD ONLY.
        El llamador es responsable de .backward() y optimizer.step().

        Parameters
        ----------
        batch : Tensor o tupla (Tensor, labels) según el modelo.

        Returns
        -------
        dict con al menos {"loss": scalar_tensor}.
        Losses adicionales para TensorBoard/WandB son bienvenidos.
        """

    @abstractmethod
    def generate(
        self,
        n_samples:  int,
        **kwargs,
    ) -> Tensor:
        """
        Genera n_samples muestras sintéticas.

        Returns
        -------
        Tensor de forma (n_samples, *input_shape).
        Para modelos autoregresivos, (n_samples, seq_len).
        Para modelos visuales, (n_samples, C, H, W).
        """

    # ------------------------------------------------------------------
    # Métodos opcionales con implementación por defecto
    # ------------------------------------------------------------------

    def configure_optimizers(
        self,
        lr: float = 1e-4,
    ) -> Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]:
        """
        Devuelve el/los optimizadores para este modelo.

        Los modelos GAN sobreescriben esto para devolver
        {"generator": opt_g, "discriminator": opt_d}.
        """
        params = []
        for net in self._networks.values():
            params.extend(net.parameters())
        return torch.optim.AdamW(params, lr=lr)

    def get_num_parameters(self) -> Dict[str, int]:
        """Cuenta parámetros por módulo y total."""
        counts = {}
        total = 0
        for name, net in self._networks.items():
            n = sum(p.numel() for p in net.parameters() if p.requires_grad)
            counts[name] = n
            total += n
        counts["total"] = total
        return counts

    def to(self, device: Union[str, torch.device]) -> "GenerativeModel":
        """Mueve todos los módulos al device especificado."""
        self.device = torch.device(device)
        for net in self._networks.values():
            net.to(self.device)
        return self

    def train_mode(self) -> "GenerativeModel":
        for net in self._networks.values():
            net.train()
        return self

    def eval_mode(self) -> "GenerativeModel":
        for net in self._networks.values():
            net.eval()
        return self

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Guarda config + state_dicts de todos los módulos."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config":    self.config,
            "class":     self.__class__.__name__,
            "networks":  {
                name: net.state_dict()
                for name, net in self._networks.items()
            },
            "extra": self._extra_checkpoint_state(),
        }
        torch.save(checkpoint, path)
        LOGGER.info("Modelo '%s' guardado en %s", self.config.name, path)

    def load_weights(self, path: Union[str, Path]) -> None:
        """Carga solo los pesos (el modelo debe estar construido)."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for name, state_dict in checkpoint["networks"].items():
            if name in self._networks:
                self._networks[name].load_state_dict(state_dict)
        self._load_extra_checkpoint_state(checkpoint.get("extra", {}))
        LOGGER.info("Pesos cargados desde %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GenerativeModel":
        """Reconstruye el modelo completo desde un checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(checkpoint["config"])
        instance.build()
        instance.load_weights(path)
        return instance

    def _extra_checkpoint_state(self) -> Dict[str, Any]:
        """
        Estado adicional a guardar en el checkpoint.
        Las subclases pueden sobreescribir (e.g., DDPM guarda los betas).
        """
        return {}

    def _load_extra_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restaura el estado adicional del checkpoint."""
        pass

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _check_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                f"El modelo '{self.config.name}' no ha sido construido. "
                "Llama a build() antes de train_step() o generate()."
            )

    def __repr__(self) -> str:
        params = self.get_num_parameters() if self._built else {}
        return (
            f"{self.__class__.__name__}("
            f"name={self.config.name!r}, "
            f"type={self.model_type.name}, "
            f"domain={self.input_domain.name}, "
            f"params={params.get('total', '?'):,}, "
            f"device={self.device})"
        )
"""
generative_models/diffusion/ddpm.py
=====================================
DDPM completo: scheduler de ruido + entrenamiento + muestreo.

Proceso forward  (fijado, no aprendido):
  q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)
  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)

Proceso reverse (aprendido por el UNet):
  p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
  El UNet predice ε_θ(x_t, t) y se recalcula μ_θ.

Objetivo de entrenamiento (Ho et al., 2020):
  L_simple = E[‖ε - ε_θ(x_t, t)‖²]

Muestreo:
  DDPM estándar (T pasos) y DDIM acelerado (n_steps << T).

Compatibilidad
--------------
  - NprintRepresentation  → condicionado por clase (num_classes > 0)
  - GASFRepresentation    → no condicionado (num_classes = 0)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import (
    GenerativeModel,
    InputDomain,
    ModelType,
)
from ..diffusion.unet import UNet2D
from .config import DiffusionConfig

from ...utils.logger_config import LOGGER

# ---------------------------------------------------------------------------
# Scheduler de ruido
# ---------------------------------------------------------------------------

class NoiseScheduler:
    """
    Precalcula y almacena todos los coeficientes del proceso de difusión.

    Soporta dos schedules:
      "linear" : betas uniformemente espaciados (Ho et al., 2020)
      "cosine" : schedule de coseno (Nichol & Dhariwal, 2021)
                 → más suave, evita degradar la imagen demasiado al inicio
    """

    def __init__(
        self,
        timesteps:     int,
        beta_start:    float = 1e-4,
        beta_end:      float = 0.02,
        schedule:      str   = "cosine",
        device:        Union[str, torch.device] = "cpu",
    ) -> None:
        self.T      = timesteps
        self.device = torch.device(device)

        betas = self._make_betas(timesteps, beta_start, beta_end, schedule)
        self._register(betas)

    def _make_betas(
        self, T: int, b0: float, bT: float, schedule: str
    ) -> Tensor:
        if schedule == "linear":
            return torch.linspace(b0, bT, T)

        elif schedule == "cosine":
            # Nichol & Dhariwal (2021) — sección 3.2
            s = 0.008
            steps = T + 1
            x = torch.linspace(0, T, steps)
            alphas_cumprod = (
                torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return torch.clamp(betas, 0.0, 0.999)

        else:
            raise ValueError(f"Schedule desconocido: {schedule!r}")

    def _register(self, betas: Tensor) -> None:
        """Precalcula todos los coeficientes necesarios."""
        self.betas   = betas.to(self.device)
        alphas       = 1.0 - betas
        alphas_bar   = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        # Coeficientes para q(x_t | x_0)
        self.sqrt_alphas_bar      = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_ab    = torch.sqrt(1.0 - alphas_bar)

        # Coeficientes para la media del proceso reverse
        self.sqrt_recip_alphas    = torch.sqrt(1.0 / alphas)
        self.posterior_var        = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

        # Para la reconstrucción de x_0 desde ε
        self.sqrt_recip_alphas_bar     = torch.sqrt(1.0 / alphas_bar)
        self.sqrt_recip_m1_alphas_bar  = torch.sqrt(1.0 / alphas_bar - 1)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Proceso forward: añade ruido gaussiano a x0 en el paso t.

        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

        Returns
        -------
        (x_t, noise) — imagen ruidosa y ruido añadido
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alphas_bar.to(t.device)[t][:, None, None, None]
        sqrt_mab = self.sqrt_one_minus_ab.to(t.device)[t][:, None, None, None]

        x_t = sqrt_ab * x0 + sqrt_mab * noise
        return x_t, noise

    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Reconstruye x_0 dada la predicción del ruido ε."""
        sqrt_rec  = self.sqrt_recip_alphas_bar.to(t.device)[t][:, None, None, None]
        sqrt_recm = self.sqrt_recip_m1_alphas_bar.to(t.device)[t][:, None, None, None]
        return sqrt_rec * x_t - sqrt_recm * eps

    def p_mean_variance(
        self, eps_pred: Tensor, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Calcula la media y varianza del paso reverse p(x_{t-1}|x_t).
        """
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)

        # Media posterior
        sqrt_ra = self.sqrt_recip_alphas[t][:, None, None, None]
        betas_t = self.betas[t][:, None, None, None]
        sqrt_mab_t = self.sqrt_one_minus_ab[t][:, None, None, None]

        mean = sqrt_ra * (x_t - betas_t / sqrt_mab_t * eps_pred)
        var  = self.posterior_var[t][:, None, None, None]

        return mean, var

    def to(self, device: Union[str, torch.device]) -> "NoiseScheduler":
        self.device = torch.device(device)
        for attr in ["betas", "sqrt_alphas_bar", "sqrt_one_minus_ab",
                     "sqrt_recip_alphas", "posterior_var",
                     "sqrt_recip_alphas_bar", "sqrt_recip_m1_alphas_bar"]:
            setattr(self, attr, getattr(self, attr).to(self.device))
        return self


# ---------------------------------------------------------------------------
# Modelo DDPM completo
# ---------------------------------------------------------------------------

class TrafficDDPM(GenerativeModel):
    """
    Modelo de difusión DDPM para imágenes de tráfico de red.

    Soporta:
    - Entrenamiento estándar DDPM (L_simple MSE sobre ruido)
    - Condicionamiento por clase/protocolo
    - Muestreo DDPM completo (T pasos)
    - Muestreo DDIM acelerado (n_steps << T, mucho más rápido)

    Compatibilidad representaciones:
    - NprintRepresentation  (invertible) → genera tráfico funcional reconstruible
    - GASFRepresentation    (no invertible) → genera patrones visuales de tráfico

    Example
    -------
    >>> cfg = DiffusionConfig(
    ...     in_channels=1, image_height=20, image_width=193,
    ...     num_classes=5, timesteps=1000
    ... )
    >>> model = TrafficDDPM(cfg).build()
    >>> batch = torch.randn(4, 1, 20, 193)
    >>> labels = torch.randint(0, 5, (4,))
    >>> losses = model.train_step((batch, labels))
    >>> samples = model.generate(n_samples=4, labels=torch.zeros(4, dtype=torch.long))
    """

    def __init__(self, config: Optional[DiffusionConfig] = None) -> None:
        if config is None:
            config = DiffusionConfig()
        super().__init__(config)
        self.cfg = config

    # ------------------------------------------------------------------
    # Propiedades abstractas
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> ModelType:
        return ModelType.DIFFUSION

    @property
    def input_domain(self) -> InputDomain:
        return InputDomain.CONTINUOUS_IMAGE

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self) -> "TrafficDDPM":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)

        # UNet
        self.unet = UNet2D(
            in_channels      = cfg.in_channels,
            base_ch          = cfg.base_ch,
            channel_mults    = cfg.channel_mults,
            n_res_per_level  = cfg.n_res_per_level,
            attention_levels = cfg.attention_levels,
            n_heads          = cfg.n_heads,
            dropout          = cfg.dropout,
            num_classes      = cfg.num_classes,
        ).to(self.device)

        # Scheduler
        self.scheduler = NoiseScheduler(
            timesteps  = cfg.timesteps,
            beta_start = cfg.beta_start,
            beta_end   = cfg.beta_end,
            schedule   = cfg.beta_schedule,
            device     = cfg.device,
        )

        self._networks = {"unet": self.unet}
        self._built    = True
        LOGGER.info("TrafficDDPM construido: %s", self)
        return self

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: Any) -> Dict[str, Tensor]:
        """
        Un paso de entrenamiento DDPM.

        L_simple = E_{t,x_0,ε}[‖ε - ε_θ(x_t, t, [labels])‖²]

        Parameters
        ----------
        batch : Tensor (B, C, H, W)  — imágenes reales normalizadas en [-1, 1]
                o tupla (images, labels) para condicionamiento

        Returns
        -------
        {"loss": mse_loss, "t_mean": timestep medio del batch}
        """
        self._check_built()

        if isinstance(batch, (tuple, list)):
            x0     = batch[0].to(self.device)
            labels = batch[1].to(self.device) if len(batch) > 1 else None
        else:
            x0     = batch.to(self.device)
            labels = None

        B = x0.shape[0]

        # Muestrear timesteps aleatorios uniformemente
        t = torch.randint(0, self.cfg.timesteps, (B,), device=self.device)

        # Proceso forward: añadir ruido
        noise = torch.randn_like(x0)
        x_t, _ = self.scheduler.q_sample(x0, t, noise)

        # Predecir el ruido con el UNet
        eps_pred = self.unet(x_t, t, labels)

        # Loss: MSE entre ruido real y predicho
        loss = F.mse_loss(eps_pred, noise)

        return {
            "loss":   loss,
            "t_mean": t.float().mean(),
        }

    # ------------------------------------------------------------------
    # generate — DDPM estándar
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        labels:    Optional[Tensor] = None,
        use_ddim:  bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Genera n_samples imágenes de tráfico sintético.

        Parameters
        ----------
        n_samples : número de muestras a generar
        labels    : (n_samples,) etiquetas de clase (None si sin condicionar)
        use_ddim  : si True usa muestreo DDIM (más rápido), si False DDPM

        Returns
        -------
        Tensor (n_samples, C, H, W) de imágenes generadas en [-1, 1]
        """
        self._check_built()
        self.eval_mode()

        if labels is not None:
            labels = labels.to(self.device)

        C = self.cfg.in_channels
        H = self.cfg.image_height
        W = self.cfg.image_width

        if use_ddim:
            return self._ddim_sample(n_samples, C, H, W, labels)
        else:
            return self._ddpm_sample(n_samples, C, H, W, labels)

    # ------------------------------------------------------------------
    # Muestreo DDPM (T pasos)
    # ------------------------------------------------------------------

    def _ddpm_sample(
        self, n: int, C: int, H: int, W: int,
        labels: Optional[Tensor],
    ) -> Tensor:
        """Muestreo DDPM estándar. Lento pero fiel a la distribución."""
        x = torch.randn(n, C, H, W, device=self.device)

        for t_val in reversed(range(self.cfg.timesteps)):
            t = torch.full((n,), t_val, device=self.device, dtype=torch.long)

            eps_pred = self.unet(x, t, labels)
            mean, var = self.scheduler.p_mean_variance(eps_pred, x, t)

            # Ruido = 0 en el último paso
            noise = torch.randn_like(x) if t_val > 0 else torch.zeros_like(x)
            x = mean + torch.sqrt(var) * noise

            if self.cfg.clip_denoised:
                x = torch.clamp(x, -1.0, 1.0)

        return x

    # ------------------------------------------------------------------
    # Muestreo DDIM (n_ddim_steps << T)
    # ------------------------------------------------------------------

    def _ddim_sample(
        self, n: int, C: int, H: int, W: int,
        labels: Optional[Tensor],
    ) -> Tensor:
        """
        DDIM (Song et al., 2020) — muestreo determinístico acelerado.

        Usa solo ddim_steps evaluaciones del UNet en lugar de T.
        Con eta=0 es completamente determinístico.
        """
        T     = self.cfg.timesteps
        steps = self.cfg.ddim_steps
        eta   = self.cfg.ddim_eta

        # Seleccionar timesteps uniformemente espaciados
        ddim_timesteps = np.linspace(0, T - 1, steps, dtype=int)[::-1]

        x = torch.randn(n, C, H, W, device=self.device)

        for i, t_val in enumerate(ddim_timesteps):
            t = torch.full((n,), t_val, device=self.device, dtype=torch.long)

            eps_pred = self.unet(x, t, labels)

            # Coeficientes del paso actual y anterior
            ab_t  = self.scheduler.sqrt_alphas_bar[t_val] ** 2
            ab_tm = (
                self.scheduler.sqrt_alphas_bar[ddim_timesteps[i + 1]] ** 2
                if i + 1 < len(ddim_timesteps) else torch.tensor(1.0)
            )

            # Reconstrucción de x_0 predicha
            x0_pred = (x - math.sqrt(1 - ab_t) * eps_pred) / math.sqrt(ab_t)
            if self.cfg.clip_denoised:
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # Dirección apuntando a x_t
            dir_xt = math.sqrt(1 - ab_tm - eta ** 2 * (1 - ab_tm) / (1 - ab_t) * (1 - ab_t / ab_tm)) * eps_pred

            # Ruido estocástico (eta=0 → determinístico)
            noise = eta * math.sqrt((1 - ab_tm) / (1 - ab_t)) * math.sqrt(1 - ab_t / ab_tm) * torch.randn_like(x)

            x = math.sqrt(ab_tm) * x0_pred + dir_xt + noise

        return x

    # ------------------------------------------------------------------
    # Persistencia extra (scheduler no tiene parámetros aprendidos
    # pero sí coeficientes que conviene guardar junto al checkpoint)
    # ------------------------------------------------------------------

    def _extra_checkpoint_state(self) -> Dict[str, Any]:
        return {
            "scheduler_T":        self.scheduler.T,
            "scheduler_betas":    self.scheduler.betas.cpu(),
            "scheduler_schedule": self.cfg.beta_schedule,
        }
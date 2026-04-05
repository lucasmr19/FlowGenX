"""
generative_models/gan/model.py
================================
GAN secuencial para generación de tráfico discreto/continuo.

Arquitectura
------------
  Generator    : ruido latente z → secuencia de tokens (LSTM + proyección)
  Discriminator: secuencia → real/fake  (Transformer encoder)

Entrenamiento
-------------
  WGAN-GP (Gulrajani et al., 2017):
    - Discriminador con gradient penalty en lugar de clipping
    - Más estable que WGAN estándar y DCGAN
    - n_critic pasos del discriminador por cada paso del generador

Compatibilidad de representación
----------------------------------
  - FlatTokenizer / ProtocolAwareTokenizer → tokens discretos
    (se trabaja con one-hot o embeddings en continuo)
  - Compatible con secuencias de longitud variable (hasta max_seq_len)

Nota sobre discrete vs. continuous GANs
----------------------------------------
Las GANs nativas no son diferenciables sobre tokens discretos.
Este módulo opera en el espacio de embeddings (continuo) durante
el entrenamiento y aplica argmax para obtener tokens discretos
al generar. Es el enfoque estándar para SeqGANs sin reinforcement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import (
    GenerativeModel,
    InputDomain,
    ModelType,
)

from .config import GANConfig
from ...utils.logger_config import LOGGER


torch.set_float32_matmul_precision('high')

# ---------------------------------------------------------------------------
# Generador LSTM
# ---------------------------------------------------------------------------

class LSTMGenerator(nn.Module):
    """
    Genera secuencias de embeddings a partir de un vector de ruido z.

    Arquitectura:
      z (latent_dim) → Linear → estado inicial (h_0, c_0) del LSTM
      LSTM(seq_len pasos) → Linear → embeddings de tokens
      Durante inferencia: argmax para tokens discretos.

    La entrada a cada paso del LSTM es el embedding del token anterior
    (teacher forcing durante entrenamiento, autoregresivo al generar).
    """

    def __init__(self, config: GANConfig) -> None:
        super().__init__()
        cfg = config

        # Proyección de z al estado inicial del LSTM
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.gen_hidden * cfg.gen_layers * 2)

        # Embedding de input (token anterior → continuo)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.gen_hidden, padding_idx=cfg.pad_token_id)

        # LSTM
        self.lstm = nn.LSTM(
            input_size  = cfg.gen_hidden,
            hidden_size = cfg.gen_hidden,
            num_layers  = cfg.gen_layers,
            batch_first = True,
            dropout     = cfg.gen_dropout if cfg.gen_layers > 1 else 0.0,
        )

        # Proyección a espacio de vocabulario
        self.out_proj = nn.Linear(cfg.gen_hidden, cfg.vocab_size)
        self.dropout  = nn.Dropout(cfg.gen_dropout)

        self.cfg = cfg

    def forward(
        self,
        z:           Tensor,
        temperature: float = 1.0,
        return_logits: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        z            : (B, latent_dim) — ruido latente
        temperature  : temperatura de muestreo
        return_logits: si True devuelve logits además de tokens

        Returns
        -------
        tokens : (B, seq_len) — IDs de tokens generados
        logits : (B, seq_len, vocab_size) — solo si return_logits=True
        """
        B   = z.shape[0]
        L   = self.cfg.seq_len
        dev = z.device

        # Estado inicial del LSTM desde z
        h0_c0 = self.z_proj(z)                        # (B, hidden*layers*2)
        hidden_size = self.cfg.gen_hidden
        n_layers    = self.cfg.gen_layers

        h0 = h0_c0[:, :hidden_size * n_layers]
        c0 = h0_c0[:, hidden_size * n_layers:]
        h0 = h0.view(B, n_layers, hidden_size).permute(1, 0, 2).contiguous()
        c0 = c0.view(B, n_layers, hidden_size).permute(1, 0, 2).contiguous()

        # BOS token como primer input
        bos_id  = torch.full((B,), 2, dtype=torch.long, device=dev)
        x       = self.token_emb(bos_id).unsqueeze(1)  # (B, 1, hidden)
        hc      = (h0, c0)

        all_logits: List[Tensor] = []
        all_tokens: List[Tensor] = []

        for _ in range(L):
            out, hc = self.lstm(x, hc)     # (B, 1, hidden)
            logit   = self.out_proj(self.dropout(out.squeeze(1)))  # (B, V)

            # Muestreo gumbel-softmax (diferenciable) durante train
            if self.training:
                token = F.gumbel_softmax(logit / temperature, tau=1.0, hard=True).argmax(-1)
            else:
                token = (logit / temperature).argmax(-1)

            all_logits.append(logit)
            all_tokens.append(token)

            # Siguiente input: embedding del token generado
            x = self.token_emb(token).unsqueeze(1)

        logits = torch.stack(all_logits, dim=1)   # (B, L, V)
        tokens = torch.stack(all_tokens, dim=1)   # (B, L)

        if return_logits:
            return tokens, logits
        return tokens, logits

    def generate(self, z: Tensor, temperature: float = 1.0) -> Tensor:
        """Alias para uso externo."""
        tokens, _ = self.forward(z, temperature=temperature)
        return tokens


# ---------------------------------------------------------------------------
# Discriminador Transformer
# ---------------------------------------------------------------------------

class TransformerDiscriminator(nn.Module):
    """
    Discriminador basado en Transformer encoder.

    Clasifica una secuencia de tokens como real (alta puntuación)
    o generada (baja puntuación).

    Arquitectura:
      Tokens → Embedding → TransformerEncoder → [CLS] token → Linear → score
    """

    def __init__(self, config: GANConfig) -> None:
        super().__init__()
        cfg = config

        # Embedding de tokens (sin weight tying con el generador)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.disc_d_model, padding_idx=cfg.pad_token_id)

        # CLS token aprendible
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.disc_d_model))

        # Positional encoding (aprendible para el discriminador)
        self.pos_emb = nn.Embedding(cfg.seq_len + 1, cfg.disc_d_model)  # +1 por CLS

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.disc_d_model,
            nhead           = cfg.disc_n_heads,
            dim_feedforward = cfg.disc_d_ff,
            dropout         = cfg.disc_dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.disc_n_layers)
        self.transformer.use_nested_tensor = False

        # Cabeza de clasificación (sin sigmoid — WGAN usa score real)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.disc_d_model),
            nn.Linear(cfg.disc_d_model, 1),
        )

        self.cfg = cfg

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Parameters
        ----------
        tokens : (B, L) — IDs enteros

        Returns
        -------
        score : (B,) — puntuación wasserstein (sin clamping)
        """
        B, L = tokens.shape
        dev  = tokens.device

        # Embeddings de tokens
        x = self.token_emb(tokens)    # (B, L, d)

        # Prepend CLS token
        cls = self.cls_token.expand(B, 1, -1)   # (B, 1, d)
        x   = torch.cat([cls, x], dim=1)         # (B, L+1, d)

        # Positional encoding
        pos  = torch.arange(L + 1, device=dev).unsqueeze(0)  # (1, L+1)
        x    = x + self.pos_emb(pos)

        # Transformer
        x = self.transformer(x)    # (B, L+1, d)

        # Extraer CLS y proyectar
        cls_out = x[:, 0, :]       # (B, d)
        score   = self.head(cls_out).squeeze(-1)  # (B,)
        return score


# ---------------------------------------------------------------------------
# Modelo GAN completo (WGAN-GP)
# ---------------------------------------------------------------------------

class TrafficGAN(GenerativeModel):
    """
    GAN secuencial con WGAN-GP para generación de tráfico de red.

    Entrenamiento
    -------------
    Alterna entre:
      1. n_critic pasos del discriminador (maximizar Wasserstein distance)
      2. 1 paso del generador (minimizar - D(G(z)))

    Gradient penalty:
      GP = λ · E[(‖∇_x̂ D(x̂)‖₂ - 1)²]
      donde x̂ = ε·x_real + (1-ε)·x_fake  (interpolaciones)

    Generación
    ----------
    z ~ N(0, I) → Generator.generate(z) → tokens discretos (B, seq_len)

    Example
    -------
    >>> cfg   = GANConfig(vocab_size=5000, seq_len=128)
    >>> model = TrafficGAN(cfg).build()
    >>> opts  = model.configure_optimizers(lr=1e-4)
    >>>
    >>> # Entrenamiento manual (ver training/trainer.py para el bucle completo)
    >>> batch = torch.randint(0, 5000, (32, 128))
    >>> # Paso discriminador:
    >>> losses_d = model.train_step_discriminator(batch, opts["discriminator"])
    >>> # Paso generador (cada n_critic pasos):
    >>> losses_g = model.train_step_generator(opts["generator"])
    """

    def __init__(self, config: Optional[GANConfig] = None) -> None:
        if config is None:
            config = GANConfig()
        super().__init__(config)
        self.cfg = config
        self._train_step_count = 0

    # ------------------------------------------------------------------
    # Propiedades abstractas
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> ModelType:
        return ModelType.GAN

    @property
    def input_domain(self) -> InputDomain:
        return InputDomain.DISCRETE_SEQUENCE

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self) -> "TrafficGAN":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)

        self.generator     = LSTMGenerator(cfg).to(self.device)
        self.discriminator = TransformerDiscriminator(cfg).to(self.device)

        self._networks = {
            "generator":     self.generator,
            "discriminator": self.discriminator,
        }
        self._built = True
        LOGGER.info("TrafficGAN construido: %s", self)
        return self

    # ------------------------------------------------------------------
    # Optimizadores separados
    # ------------------------------------------------------------------

    def configure_optimizers(
        self, lr: float = 1e-4
    ) -> Dict[str, torch.optim.Optimizer]:
        """
        Devuelve optimizadores independientes para G y D.
        WGAN-GP recomienda Adam con betas=(0.0, 0.9).
        """
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr, betas=(0.0, 0.9),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr, betas=(0.0, 0.9),
        )
        return {"generator": opt_g, "discriminator": opt_d}

    # ------------------------------------------------------------------
    # train_step  (interfaz unificada del framework)
    # ------------------------------------------------------------------

    def train_step(self, batch: Any) -> Dict[str, Tensor]:
        """
        Interfaz unificada. El Trainer llama a este método.

        Internamente alterna entre pasos del discriminador y del generador
        según n_critic. Los optimizadores se esperan en self._optimizers
        (asignados por el Trainer antes de llamar a train_step).

        Returns
        -------
        Dict con "loss" (pérdida del generador o del discriminador según el paso).
        """
        self._check_built()
        self._train_step_count += 1

        # Obtener optimizadores (asignados por Trainer)
        opt_d = getattr(self, "_opt_discriminator", None)
        opt_g = getattr(self, "_opt_generator",     None)

        if opt_d is None or opt_g is None:
            raise RuntimeError(
                "TrafficGAN.train_step() requiere que el Trainer asigne "
                "self._opt_discriminator y self._opt_generator antes de llamar."
            )

        # Paso del discriminador
        losses_d = self.train_step_discriminator(batch, opt_d)

        # Paso del generador cada n_critic iteraciones
        losses_g: Dict[str, Tensor] = {}
        if self._train_step_count % self.cfg.n_critic == 0:
            losses_g = self.train_step_generator(opt_g)

        return {**losses_d, **losses_g, "loss": losses_d["loss_d"]}

    # ------------------------------------------------------------------
    # Paso del discriminador
    # ------------------------------------------------------------------

    def train_step_discriminator(
        self,
        real_tokens: Tensor,
        optimizer:   torch.optim.Optimizer,
    ) -> Dict[str, Tensor]:
        """
        Maximiza: E[D(real)] - E[D(fake)] - λ·GP

        Parameters
        ----------
        real_tokens : (B, L) — tokens reales del dataset
        optimizer   : optimizador del discriminador
        """
        B = real_tokens.shape[0]
        real_tokens = real_tokens.to(self.device)

        self.discriminator.train()
        optimizer.zero_grad()

        # Generar secuencias falsas
        z    = torch.randn(B, self.cfg.latent_dim, device=self.device)
        with torch.no_grad():
            fake_tokens, _ = self.generator(z)

        # Puntuaciones
        d_real = self.discriminator(real_tokens)
        d_fake = self.discriminator(fake_tokens.detach())

        # WGAN loss
        loss_d = -(d_real.mean() - d_fake.mean())

        # Gradient penalty
        if self.cfg.use_gradient_penalty:
            gp     = self._gradient_penalty(real_tokens, fake_tokens)
            loss_d = loss_d + self.cfg.lambda_gp * gp
        else:
            gp = torch.tensor(0.0, device=self.device)

        loss_d.backward()
        optimizer.step()

        # Weight clipping (alternativa al GP, menos recomendado)
        if not self.cfg.use_gradient_penalty:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.cfg.clip_value, self.cfg.clip_value)

        return {
            "loss_d":   loss_d.detach(),
            "d_real":   d_real.mean().detach(),
            "d_fake":   d_fake.mean().detach(),
            "gradient_penalty": gp.detach() if isinstance(gp, Tensor) else gp,
            "wasserstein_dist": (d_real.mean() - d_fake.mean()).detach(),
        }

    # ------------------------------------------------------------------
    # Paso del generador
    # ------------------------------------------------------------------

    def train_step_generator(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Minimiza: -E[D(G(z))]
        """
        B = batch_size or 32
        self.generator.train()
        optimizer.zero_grad()

        z    = torch.randn(B, self.cfg.latent_dim, device=self.device)
        fake_tokens, logits = self.generator(z, return_logits=True)

        d_fake  = self.discriminator(fake_tokens)
        loss_g  = -d_fake.mean()

        loss_g.backward()
        optimizer.step()

        return {
            "loss_g": loss_g.detach(),
            "d_fake_g": d_fake.mean().detach(),
        }

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        n_samples:   int,
        temperature: float = 1.0,
        **kwargs,
    ) -> Tensor:
        """
        Genera n_samples secuencias de tokens.

        Returns
        -------
        Tensor (n_samples, seq_len) de IDs enteros
        """
        self._check_built()
        self.eval_mode()

        z = torch.randn(n_samples, self.cfg.latent_dim, device=self.device)
        tokens = self.generator.generate(z, temperature=temperature)
        return tokens

    # ------------------------------------------------------------------
    # Gradient penalty (WGAN-GP)
    # ------------------------------------------------------------------

    def _gradient_penalty(
        self,
        real_tokens: Tensor,
        fake_tokens: Tensor,
    ) -> Tensor:
        """
        GP = E[(‖∇_x̂ D(x̂)‖₂ − 1)²]

        Nota: como las secuencias son discretas, el gradient penalty
        se calcula sobre los embeddings interpolados (espacio continuo).
        """
        B   = real_tokens.shape[0]
        eps = torch.rand(B, 1, device=self.device)

        # Obtener embeddings del discriminador para interpolación
        real_emb = self.discriminator.token_emb(real_tokens)  # (B, L, d)
        fake_emb = self.discriminator.token_emb(fake_tokens)  # (B, L, d)

        eps_3d   = eps.unsqueeze(-1).expand_as(real_emb)
        interp   = (eps_3d * real_emb + (1 - eps_3d) * fake_emb).requires_grad_(True)

        # Score sobre interpolación (inyectando embeddings directamente)
        # Para esto necesitamos un forward parcial del discriminador
        B, L, d = interp.shape
        cls  = self.discriminator.cls_token.expand(B, 1, -1)
        x    = torch.cat([cls, interp], dim=1)
        pos  = torch.arange(L + 1, device=self.device).unsqueeze(0)
        x    = x + self.discriminator.pos_emb(pos)
        x    = self.discriminator.transformer(x)
        score = self.discriminator.head(x[:, 0]).squeeze(-1)

        # Gradiente respecto a la interpolación
        grads = torch.autograd.grad(
            outputs    = score,
            inputs     = interp,
            grad_outputs = torch.ones_like(score),
            create_graph = True,
            retain_graph = True,
            only_inputs  = True,
        )[0]

        # ‖∇‖₂ promediado sobre la dimensión de features
        grad_norm = grads.norm(2, dim=(1, 2))
        gp = ((grad_norm - 1) ** 2).mean()
        return gp
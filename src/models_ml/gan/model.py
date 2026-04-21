# [FIX #1] Unified loss: single coherent scalar regardless of which
# networks were updated this step.
#
# • Generator updated  → use loss_g  = -E[D(fake)]
# • Generator NOT updated (n_critic cooldown) → use
#   -wasserstein_dist = -(E[D(real)] - E[D(fake)])
#   Both quantities share the same sign convention: lower is better.
# [FIX #2] Use hard=True (straight-through estimator) instead of
# hard=False. This produces one-hot gradients on the forward pass
# while keeping a differentiable path on the backward pass,
# reducing the mismatch between training embeddings and the
# real token embeddings seen at inference time. The discriminator
# receives inputs much closer to its real-data distribution,
# which stabilises the adversarial training signal.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm

from ..base import GenerativeModel, InputDomain, ModelType
from .config import GANConfig
from ...utils.logger_config import LOGGER

torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------
class LSTMGenerator(nn.Module):
    """
    Autoregressive LSTM generator for discrete sequences.

    The generator supports optional class conditioning. When conditional
    mode is enabled, a label embedding is:
      1) concatenated to the latent vector z to initialize the hidden state,
      2) concatenated to the token embedding at every decoding step.

    Public API:
      - forward(z, labels=None, temperature=1.0, sample_hard=False)
      - generate(z, labels=None, temperature=1.0)

    Notes:
      - During training, labels should be passed explicitly when the model
        is conditional.
      - If labels are omitted in conditional mode, labels are sampled
        uniformly. This keeps backward compatibility.

    Changes from original:
      [FIX #2] gumbel_softmax now uses hard=True (straight-through estimator)
               to reduce train/inference mismatch and stabilize discriminator.
    """

    def __init__(self, config: GANConfig) -> None:
        super().__init__()
        cfg = config
        self.cfg = cfg

        self.num_classes = cfg.num_classes
        self.cond_dim = cfg.cond_dim
        self.conditional = self.num_classes is not None and self.num_classes > 0

        if self.conditional:
            self.label_emb = nn.Embedding(self.num_classes, self.cond_dim)
            z_in_dim = cfg.latent_dim + self.cond_dim
            lstm_in_dim = cfg.gen_hidden + self.cond_dim
        else:
            self.label_emb = None
            z_in_dim = cfg.latent_dim
            lstm_in_dim = cfg.gen_hidden

        self.z_proj = nn.Linear(z_in_dim, cfg.gen_hidden * cfg.gen_layers * 2)

        self.token_emb = nn.Embedding(
            cfg.vocab_size,
            cfg.gen_hidden,
            padding_idx=cfg.pad_token_id,
        )

        self.lstm = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=cfg.gen_hidden,
            num_layers=cfg.gen_layers,
            batch_first=True,
            dropout=cfg.gen_dropout if cfg.gen_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(cfg.gen_hidden)
        self.out_proj = nn.utils.weight_norm(nn.Linear(cfg.gen_hidden, cfg.vocab_size))
        self.dropout = nn.Dropout(cfg.gen_dropout)

    def _sample_labels(self, batch_size: int, device: torch.device) -> Optional[Tensor]:
        if not self.conditional:
            return None
        return torch.randint(0, self.num_classes, (batch_size,), device=device)

    def _validate_labels(
        self,
        labels: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        allow_sampling: bool = True,
    ) -> Optional[Tensor]:
        """
        Normalize labels to a 1D LongTensor on the target device.

        Parameters
        ----------
        labels:
            Optional label tensor.
        batch_size:
            Expected batch size.
        device:
            Target device.
        allow_sampling:
            If True, sample labels when labels is None in conditional mode.
        """
        if not self.conditional:
            return None

        if labels is None:
            if not allow_sampling:
                raise ValueError("labels are required in conditional mode.")
            return self._sample_labels(batch_size, device)

        labels = labels.to(device).long()
        if labels.shape[0] != batch_size:
            raise ValueError(f"labels batch size {labels.shape[0]} != batch size {batch_size}")
        return labels

    def forward(
        self,
        z: Tensor,
        labels: Optional[Tensor] = None,
        temperature: float = 1.0,
        sample_hard: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Generate a batch of discrete sequences from latent vectors.

        Parameters
        ----------
        z:
            Latent tensor of shape (B, latent_dim).
        labels:
            Optional class labels of shape (B,).
        temperature:
            Gumbel-softmax temperature used when sample_hard=False.
        sample_hard:
            If True, the output tokens are obtained by argmax on logits.
            If False, a differentiable straight-through hard sample is used
            for embeddings (reduces train/inference mismatch).

        Returns
        -------
        dict with:
            - tokens:     (B, L)
            - logits:     (B, L, V)
            - embeddings: (B, L, H)
        """
        B = z.shape[0]
        L = self.cfg.seq_len
        dev = z.device

        if self.conditional:
            labels = self._validate_labels(labels, B, dev, allow_sampling=True)
            y_emb = self.label_emb(labels)
            z = torch.cat([z, y_emb], dim=-1)
        else:
            y_emb = None

        h0_c0 = self.z_proj(z)
        hidden_size = self.cfg.gen_hidden
        n_layers = self.cfg.gen_layers

        h0 = h0_c0[:, : hidden_size * n_layers]
        c0 = h0_c0[:, hidden_size * n_layers :]

        h0 = h0.view(B, n_layers, hidden_size).permute(1, 0, 2).contiguous()
        c0 = c0.view(B, n_layers, hidden_size).permute(1, 0, 2).contiguous()

        bos_id = getattr(self.cfg, "bos_token_id", 2)
        bos = torch.full((B,), bos_id, dtype=torch.long, device=dev)
        x = self.token_emb(bos).unsqueeze(1)

        hc = (h0, c0)

        all_logits: List[Tensor] = []
        all_tokens: List[Tensor] = []
        all_embs: List[Tensor] = []

        for _ in range(L):
            lstm_in = x
            if self.conditional:
                lstm_in = torch.cat([x, y_emb.unsqueeze(1)], dim=-1)

            out, hc = self.lstm(lstm_in, hc)
            hidden = self.norm(out.squeeze(1))
            logit = self.out_proj(self.dropout(hidden))  # (B, V)

            if sample_hard:
                token = torch.argmax(logit, dim=-1)
                emb = self.token_emb(token)
            else:
                probs = F.gumbel_softmax(logit, tau=temperature, hard=True, dim=-1)
                token = torch.argmax(probs, dim=-1)
                emb = probs @ self.token_emb.weight

            all_logits.append(logit)
            all_tokens.append(token)
            all_embs.append(emb)

            x = emb.unsqueeze(1)

        return {
            "tokens": torch.stack(all_tokens, dim=1),
            "logits": torch.stack(all_logits, dim=1),
            "embeddings": torch.stack(all_embs, dim=1),
        }

    @torch.no_grad()
    def generate(
        self,
        z: Tensor,
        labels: Optional[Tensor] = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Convenience wrapper around forward() that returns only tokens.
        """
        out = self.forward(z, labels=labels, temperature=temperature, sample_hard=True)
        return out["tokens"]


# ---------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------
class TransformerDiscriminator(nn.Module):
    """
    Transformer discriminator with projection conditioning.

    The sequence is encoded with a CLS token. Conditioning is applied via
    a projection term instead of prepending the label as an input token:

        score(x, y) = f(h_cls(x)) + <h_cls(x), e(y)>

    where:
      - h_cls(x) is the CLS representation produced by the transformer,
      - f is an unconditional scalar head,
      - e(y) is a label embedding.

    Public API:
      - forward(tokens=None, embeddings=None, labels=None, padding_mask=None)
    """

    def __init__(self, config: GANConfig) -> None:
        super().__init__()
        cfg = config
        self.cfg = cfg

        self.num_classes = cfg.num_classes
        self.conditional = self.num_classes is not None and self.num_classes > 0

        self.token_emb = nn.Embedding(
            cfg.vocab_size,
            cfg.disc_d_model,
            padding_idx=cfg.pad_token_id,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.disc_d_model))
        self.pos_emb = nn.Embedding(cfg.seq_len + 1, cfg.disc_d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.disc_d_model,
            nhead=cfg.disc_n_heads,
            dim_feedforward=cfg.disc_d_ff,
            dropout=cfg.disc_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.disc_n_layers)

        self.input_proj = spectral_norm(nn.Linear(cfg.gen_hidden, cfg.disc_d_model))
        self.uncond_head = spectral_norm(nn.Linear(cfg.disc_d_model, 1))
        self.norm = nn.LayerNorm(cfg.disc_d_model)

        if self.conditional:
            self.label_emb = nn.Embedding(self.num_classes, cfg.disc_d_model)
        else:
            self.label_emb = None

    def _sample_labels(self, batch_size: int, device: torch.device) -> Optional[Tensor]:
        if not self.conditional:
            return None
        return torch.randint(0, self.num_classes, (batch_size,), device=device)

    def _validate_labels(
        self,
        labels: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        allow_sampling: bool = True,
    ) -> Optional[Tensor]:
        """
        Normalize labels to a 1D LongTensor on the target device.
        """
        if not self.conditional:
            return None

        if labels is None:
            if not allow_sampling:
                raise ValueError("labels are required in conditional mode.")
            return self._sample_labels(batch_size, device)

        labels = labels.to(device).long()
        if labels.shape[0] != batch_size:
            raise ValueError(f"labels batch size {labels.shape[0]} != batch size {batch_size}")
        return labels

    def forward(
        self,
        tokens: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Score a batch of real or generated sequences.

        Parameters
        ----------
        tokens:
            Integer token ids of shape (B, L). Used when raw tokens are provided.
        embeddings:
            Continuous token embeddings of shape (B, L, H). Used for generator outputs.
        labels:
            Optional class labels of shape (B,).
        padding_mask:
            Optional boolean mask for padded positions, shape (B, L).

        Returns
        -------
        Tensor of shape (B,) with one score per sample.
        """
        if embeddings is None:
            if tokens is None:
                raise ValueError("Provide either tokens or embeddings.")
            embeddings = self.token_emb(tokens)
            padding_mask = tokens.eq(self.cfg.pad_token_id)
        else:
            if embeddings.shape[-1] == self.cfg.gen_hidden:
                embeddings = self.input_proj(embeddings)
            elif embeddings.shape[-1] != self.cfg.disc_d_model:
                raise ValueError(
                    f"Unexpected embedding dim {embeddings.shape[-1]}; "
                    f"expected {self.cfg.gen_hidden} or {self.cfg.disc_d_model}."
                )

        return self._forward_embeddings(embeddings, labels=labels, padding_mask=padding_mask)

    def _forward_embeddings(
        self,
        embeddings: Tensor,
        labels: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, _, _ = embeddings.shape
        dev = embeddings.device

        if self.conditional:
            labels = self._validate_labels(labels, B, dev, allow_sampling=True)
            y_emb = self.label_emb(labels)  # (B, D)
        else:
            y_emb = None

        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, embeddings], dim=1)  # (B, L+1, D)

        pos = torch.arange(x.shape[1], device=dev).unsqueeze(0)
        x = x + self.pos_emb(pos)

        if padding_mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=dev)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        h = self.norm(x[:, 0, :])  # CLS representation

        out = self.uncond_head(h).squeeze(-1)

        if self.conditional:
            out = out + torch.sum(h * y_emb, dim=1)

        return out


# ---------------------------------------------------------------------
# Full GAN
# ---------------------------------------------------------------------
class TrafficGAN(GenerativeModel):
    """
    Conditional or unconditional GAN for discrete traffic sequences.

    This class preserves the original public API:
      - build()
      - configure_optimizers()
      - train_step()
      - train_step_discriminator()
      - train_step_generator()
      - generate()
      - get_metrics_schema()       [NEW] canonical metric schema for trainers/loggers

    Conditioning behavior:
      - If cfg.num_classes > 0, labels are used consistently in both
        generator and discriminator.
      - If cfg.num_classes is None or <= 0, the model behaves unconditionally.
    """

    def __init__(self, config: Optional[GANConfig] = None) -> None:
        if config is None:
            config = GANConfig()
        super().__init__(config)
        self.cfg = config
        self._train_step_count = 0
        self._ema_generator: Optional[nn.Module] = None

    @property
    def model_type(self) -> ModelType:
        return ModelType.GAN

    @property
    def input_domain(self) -> InputDomain:
        return InputDomain.DISCRETE_SEQUENCE

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> "TrafficGAN":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)

        self.generator = LSTMGenerator(cfg).to(self.device)
        self.discriminator = TransformerDiscriminator(cfg).to(self.device)

        # [NEW #6] Exponential Moving Average copy of the generator.
        # Used only at inference time; never receives gradients.
        ema_decay = getattr(cfg, "ema_decay", 0.999)
        if ema_decay > 0.0:
            self._ema_generator = LSTMGenerator(cfg).to(self.device)
            self._ema_generator.load_state_dict(self.generator.state_dict())
            for p in self._ema_generator.parameters():
                p.requires_grad_(False)
            self._ema_decay = ema_decay
        else:
            self._ema_generator = None

        self._networks = {
            "generator": self.generator,
            "discriminator": self.discriminator,
        }
        self._built = True
        LOGGER.info("TrafficGAN built: %s", self)
        return self

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    def _update_ema(self) -> None:
        """Update the EMA generator weights after each generator step."""
        if self._ema_generator is None:
            return
        decay = self._ema_decay
        with torch.no_grad():
            for ema_p, gen_p in zip(
                self._ema_generator.parameters(), self.generator.parameters()
            ):
                ema_p.data.mul_(decay).add_(gen_p.data, alpha=1.0 - decay)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(
        self, lr: float = 1e-4
    ) -> Dict[str, torch.optim.Optimizer]:
        """
        Create independent optimizers for generator and discriminator.
        """
        lr_g = float(getattr(self.cfg, "lr_opt_g", lr))
        lr_d = float(getattr(self.cfg, "lr_opt_d", lr))

        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(0.0, 0.9),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(0.0, 0.9),
        )
        
        return {"generator": opt_g, "discriminator": opt_d}

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _extract_batch(self, batch: Any) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Extract tokens and optional labels from a batch.

        Supported formats:
          - Tensor: tokens only
          - tuple/list: (tokens, labels?)
          - dict: {"tokens": ..., "y": ...} or compatible aliases
        """
        if isinstance(batch, torch.Tensor):
            return batch, None

        if isinstance(batch, (tuple, list)):
            if len(batch) == 0:
                raise ValueError("Empty batch.")
            real_tokens = batch[0]
            labels = batch[1] if len(batch) > 1 else None
            return real_tokens, labels

        if isinstance(batch, dict):
            real_tokens = batch.get("tokens", batch.get("x"))
            if real_tokens is None:
                raise ValueError("Could not infer token batch from dict.")
            labels = batch.get("y", batch.get("label", batch.get("labels")))
            return real_tokens, labels

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _sample_labels(self, batch_size: int) -> Optional[Tensor]:
        if getattr(self.cfg, "num_classes", 0) <= 0:
            return None
        return torch.randint(0, self.cfg.num_classes, (batch_size,), device=self.device)

    def _normalize_labels(
        self,
        labels: Optional[Union[int, Tensor, List[int], Tuple[int, ...]]],
        n_samples: int,
    ) -> Optional[Tensor]:
        """
        Normalize labels for generation to a tensor on the correct device.
        """
        if labels is None:
            return self._sample_labels(n_samples)

        if isinstance(labels, int):
            y = torch.full((n_samples,), labels, device=self.device, dtype=torch.long)
        elif isinstance(labels, (list, tuple)):
            y = torch.tensor(labels, device=self.device, dtype=torch.long)
        elif torch.is_tensor(labels):
            y = labels.to(self.device).long()
        else:
            raise TypeError(f"Unsupported labels type: {type(labels)}")

        if y.shape[0] != n_samples:
            raise ValueError(f"labels has {y.shape[0]} labels but n_samples={n_samples}")

        return y

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def train_step(self, batch: Any) -> Dict[str, Tensor]:
        """
        Run one joint training step.

        The discriminator is updated first, then the generator every
        n_critic steps. For conditional training, the same labels tensor
        is reused consistently within the batch.

        Returns
        -------
        Dict containing all individual losses plus:
          - ``loss``:         alias for ``unified_loss`` (used by trainer/callbacks).
          - ``unified_loss``: coherent scalar for logging, checkpointing, and
                              early stopping across D-only and G+D steps.

        Changes:
          [FIX #1] ``unified_loss`` is defined explicitly instead of mixing
                   loss_g and loss_d scales.  When the generator is updated
                   in this step, unified_loss = loss_g (already on a
                   consistent scale: -E[D(fake)]).  When only the discriminator
                   runs (n_critic cooldown), unified_loss = -wasserstein_dist,
                   which shares the same scale and sign convention as loss_g.
        """
        self._check_built()
        self._train_step_count += 1

        opt_d = getattr(self, "_opt_discriminator", None)
        opt_g = getattr(self, "_opt_generator", None)

        if opt_d is None or opt_g is None:
            raise RuntimeError(
                "TrafficGAN.train_step() requires _opt_discriminator and _opt_generator "
                "to be assigned by the trainer."
            )

        real_tokens, labels = self._extract_batch(batch)
        real_tokens = real_tokens.to(self.device)

        if getattr(self.cfg, "num_classes", 0) > 0:
            if labels is None:
                labels = self._sample_labels(real_tokens.shape[0])
            else:
                labels = labels.to(self.device).long()
                if labels.shape[0] != real_tokens.shape[0]:
                    raise ValueError(
                        f"labels batch size {labels.shape[0]} != real batch size {real_tokens.shape[0]}"
                    )
        else:
            labels = None

        losses_d = self.train_step_discriminator(real_tokens, opt_d, labels=labels)

        losses_g: Dict[str, Tensor] = {}
        if self._train_step_count % self.cfg.n_critic == 0:
            losses_g = self.train_step_generator(
                opt_g,
                batch_size=real_tokens.shape[0],
                labels=labels,
            )
            # [NEW #6] Keep EMA weights in sync after every generator update.
            self._update_ema()


        if "loss_g" in losses_g:
            unified_loss: Tensor = losses_g["loss_g"]
        else:
            unified_loss = -losses_d["wasserstein_dist"]

        return {
            **losses_d,
            **losses_g,
            "loss": unified_loss,
            "unified_loss": unified_loss,
        }

    def train_step_discriminator(
        self,
        real_tokens: Tensor,
        optimizer: torch.optim.Optimizer,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Update the discriminator once.

        For conditional training, the same labels are used for:
          - real samples: D(x_real, y)
          - fake samples: D(G(z, y), y)

        Changes:
          [FIX #3, #4] _gradient_penalty now receives real_tokens so it can
                       project real embeddings to disc_d_model before
                       interpolating, and propagate the correct padding_mask.
        """
        B = real_tokens.shape[0]
        real_tokens = real_tokens.to(self.device)

        if getattr(self.cfg, "num_classes", 0) > 0:
            if labels is None:
                labels = self._sample_labels(B)
            else:
                labels = labels.to(self.device).long()
                if labels.shape[0] != B:
                    raise ValueError(f"labels batch size {labels.shape[0]} != real batch size {B}")
        else:
            labels = None

        self.generator.train()
        self.discriminator.train()
        optimizer.zero_grad()

        z = torch.randn(B, self.cfg.latent_dim, device=self.device)
        with torch.no_grad():
            fake_out = self.generator(
                z,
                labels=labels,
                temperature=getattr(self.cfg, "gen_temperature", 1.0),
                sample_hard=False,
            )
            fake_emb = fake_out["embeddings"]

        d_real = self.discriminator(tokens=real_tokens, labels=labels)
        d_fake = self.discriminator(embeddings=fake_emb, labels=labels)

        loss_d = d_fake.mean() - d_real.mean()

        if self.cfg.use_gradient_penalty:
            gp = self._gradient_penalty(real_tokens, fake_emb, labels=labels)
            loss_d = loss_d + self.cfg.lambda_gp * gp
        else:
            gp = torch.tensor(0.0, device=self.device)

        loss_d.backward()
        optimizer.step()

        if not self.cfg.use_gradient_penalty:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.cfg.clip_value, self.cfg.clip_value)

        wasserstein = d_real.mean() - d_fake.mean()

        return {
            "loss_d": loss_d.detach(),
            "d_real": d_real.mean().detach(),
            "d_fake": d_fake.mean().detach(),
            "gradient_penalty": gp.detach(),
            "wasserstein_dist": wasserstein.detach(),
        }

    def train_step_generator(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: Optional[int] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Update the generator once.

        For conditional training, the generator must receive the same labels
        tensor that will be used by the discriminator.
        """
        B = batch_size or getattr(self.cfg, "batch_size", 32)

        if getattr(self.cfg, "num_classes", 0) > 0:
            if labels is None:
                labels = self._sample_labels(B)
            else:
                labels = labels.to(self.device).long()
                if labels.shape[0] != B:
                    raise ValueError(f"labels batch size {labels.shape[0]} != batch size {B}")
        else:
            labels = None

        self.generator.train()
        self.discriminator.train()
        optimizer.zero_grad()

        z = torch.randn(B, self.cfg.latent_dim, device=self.device)
        fake_out = self.generator(
            z,
            labels=labels,
            temperature=getattr(self.cfg, "gen_temperature", 1.0),
            sample_hard=False,
        )
        fake_emb = fake_out["embeddings"]

        d_fake = self.discriminator(embeddings=fake_emb, labels=labels)
        loss_g = -d_fake.mean()

        loss_g.backward()
        optimizer.step()

        return {
            "loss_g": loss_g.detach(),
            "d_fake_g": d_fake.mean().detach(),
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        temperature: float = 1.0,
        labels: Optional[Union[int, Tensor, List[int], Tuple[int, ...]]] = None,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        """
        Generate discrete sequences from the GAN.

        Uses the EMA generator when available for more stable outputs.

        Parameters
        ----------
        n_samples:
            Number of samples to generate.
        temperature:
            Sampling temperature passed to the generator.
        labels:
            Optional class conditioning. Accepts: int, list[int],
            tuple[int, ...], or Tensor.  Ignored for unconditional models.
        **kwargs:
            Preserved for backward compatibility.
              - return_labels: bool

        Returns
        -------
        Tensor
            Token tensor of shape (B, L) by default.
        Tuple[Tensor, Optional[Tensor]]
            If return_labels=True, returns (tokens, labels).
        """
        self._check_built()
        self.eval_mode()

        return_labels = bool(kwargs.pop("return_labels", False))

        y_tensor = self._normalize_labels(labels, n_samples)

        z = torch.randn(n_samples, self.cfg.latent_dim, device=self.device)

        # [NEW #6] Prefer the EMA generator at inference time.
        gen = self._ema_generator if self._ema_generator is not None else self.generator
        out = gen(z, labels=y_tensor, temperature=temperature, sample_hard=True)
        tokens = out["tokens"]

        if return_labels:
            return tokens, y_tensor
        return tokens

    # ------------------------------------------------------------------
    # Gradient penalty
    # ------------------------------------------------------------------

    def _gradient_penalty(
        self,
        real_tokens: Tensor,
        fake_embeddings: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        WGAN-GP penalty computed in embedding space.

        Changes:
          [FIX #3] Real token embeddings are projected to disc_d_model before
                   interpolation so that both endpoints of the linear
                   interpolation live in the *same* space. Previously, real
                   embeddings were in gen_hidden space while fake embeddings
                   had already been projected, which broke the Lipschitz
                   constraint the penalty is meant to enforce.

          [FIX #4] The padding mask derived from real_tokens is forwarded to
                   _forward_embeddings so that padding positions are correctly
                   excluded from the gradient computation. Ignoring the mask
                   causes the gradient norm to be inflated by padded tokens,
                   making the 1-Lipschitz target harder to satisfy.
        """
        B = real_tokens.shape[0]

        real_embeddings = self.discriminator.token_emb(real_tokens)
        if real_embeddings.shape[-1] != self.cfg.disc_d_model:
            real_embeddings = self.discriminator.input_proj(real_embeddings)

        # Fake embeddings may still be in gen_hidden space — normalise them too.
        if fake_embeddings.shape[-1] != self.cfg.disc_d_model:
            fake_embeddings = self.discriminator.input_proj(fake_embeddings)

        eps = torch.rand(B, 1, 1, device=self.device)
        interp = (eps * real_embeddings + (1.0 - eps) * fake_embeddings).requires_grad_(True)

        padding_mask = real_tokens.eq(self.cfg.pad_token_id)

        score = self.discriminator._forward_embeddings(
            interp,
            labels=labels,
            padding_mask=padding_mask,
        )

        grads = torch.autograd.grad(
            outputs=score,
            inputs=interp,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_norm = grads.flatten(1).norm(2, dim=1)
        gp = ((grad_norm - 1.0) ** 2).mean()
        return gp

    # ------------------------------------------------------------------
    # Metrics schema
    # ------------------------------------------------------------------

    @staticmethod
    def get_metrics_schema() -> Dict[str, Optional[str]]:
        """
        Return a canonical description of every metric produced by train_step.

        Trainers and loggers should use this schema instead of hardcoding key
        names, so that future metric additions are automatically picked up.

        Returns
        -------
        Dict mapping metric key → human-readable description (or None).

        Usage example in a trainer
        --------------------------
        >>> schema = TrafficGAN.get_metrics_schema()
        >>> metrics = {k: out.get(k) for k in schema}
        """
        return {
            # Core losses
            "loss":             "Alias for unified_loss; used by trainer callbacks.",
            "unified_loss":     "Coherent scalar for logging/checkpointing/early-stopping.",
            "loss_d":           "WGAN discriminator loss = E[D(fake)] - E[D(real)].",
            "loss_g":           "Generator loss = -E[D(fake)].  None on D-only steps.",
            # Discriminator internals
            "d_real":           "Mean discriminator score on real samples.",
            "d_fake":           "Mean discriminator score on fake samples (D step).",
            "d_fake_g":         "Mean discriminator score on fake samples (G step).",
            "wasserstein_dist": "Wasserstein estimate = E[D(real)] - E[D(fake)].",
            "gradient_penalty": "WGAN-GP gradient penalty term (0 if disabled).",
        }
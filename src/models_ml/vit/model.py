"""
generative_models/vit/model.py
================================
Vision Transformer generativo (MAE-style) para representaciones
de tráfico basadas en imagen.

Arquitectura
------------

  Imagen (B, C, H, W)
        ↓
  PatchEmbedding   →  (B, N, D_enc)
        ↓
  PositionalEncoding 2D
        ↓
  [MASK] enmascaramiento aleatorio / por bloques / grid
        ↓
  N x ViTEncoderBlock  (self-attention + MLP)
        ↓
  LatentProjection  →  (B, N, D_dec)        ← paso al decoder
        ↓
  M x ViTDecoderBlock  (self-attention + MLP)
  con tokens [MASK] aprendibles en las posiciones ocultas
        ↓
  PixelHead  →  reconstrucción de parches enmascarados (B, N_masked, P²·C)

Generación iterativa (MAGE-inspired)
--------------------------------------
  1. Empezar con todos los parches enmascarados.
  2. En cada paso, predecir todos los parches y revelar los de
     mayor confianza (score = -entropía del softmax predicho).
  3. Repetir durante ``generation_steps`` iteraciones.

Compatibilidad
--------------
  - Hereda de GenerativeModel: misma API que TrafficTransformer y TrafficGAN.
  - Métodos públicos: build(), configure_optimizers(), train_step(), generate(),
    encode(), get_metrics_schema().

Inspirado en:
  - MAE (He et al., 2022)   https://arxiv.org/abs/2111.06377
  - MAGE (Li et al., 2023)  https://arxiv.org/abs/2211.09117
  - ViT (Dosovitskiy, 2020) https://arxiv.org/abs/2010.11929
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import GenerativeModel, InputDomain, ModelType
from .config import ViTConfig
from ...utils.logger_config import LOGGER


# ---------------------------------------------------------------------------
# Bloques auxiliares
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """
    Stochastic depth (Huang et al., 2016).
    Desactiva aleatoriamente bloques enteros durante el entrenamiento,
    equivalente a un ensemble de redes de profundidad variable.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, device=x.device).floor_().div_(keep)
        return x * noise

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class PatchEmbedding(nn.Module):
    """
    Proyección lineal de parches.

    Divide la imagen en parches no solapados de tamaño (P, P) y los proyecta
    a d_model dimensiones con una convolución stride=P (equivalente a la
    proyección lineal del paper original de ViT).

    Input:  (B, C, H, W)
    Output: (B, N, d_model)  donde N = (H/P)·(W/P)
    """

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)          # (B, d_model, H/P, W/P)
        x = x.flatten(2)          # (B, d_model, N)
        x = x.transpose(1, 2)     # (B, N, d_model)
        return self.norm(x)


class PositionalEncoding2D(nn.Module):
    """
    Codificación posicional sinusoidal 2D factorizada por fila y columna.

    Para cada posición (row, col) del grid de parches:
      PE[row, col, 2i]     = sin(row / 10000^(2i / d_model))  para la mitad eje-y
      PE[row, col, 2i+1]   = cos(row / 10000^(2i / d_model))
      PE[row, col, d+2i]   = sin(col / 10000^(2i / d_model))  para la mitad eje-x
      PE[row, col, d+2i+1] = cos(col / 10000^(2i / d_model))

    Ventaja frente a PE 1D: preserva la estructura espacial 2D.
    No tiene parámetros aprendibles → generaliza a resoluciones distintas.
    """

    def __init__(self, d_model: int, grid_size: int) -> None:
        super().__init__()
        assert d_model % 4 == 0, "d_model debe ser divisible por 4 para PE 2D."

        half = d_model // 2
        div = torch.exp(
            torch.arange(0, half, 2, dtype=torch.float) * (-math.log(10000.0) / half)
        )  # (half/2,)

        pe = torch.zeros(grid_size, grid_size, d_model)
        rows = torch.arange(grid_size, dtype=torch.float).unsqueeze(1)  # (G, 1)
        cols = torch.arange(grid_size, dtype=torch.float).unsqueeze(1)  # (G, 1)

        pe[:, :, 0:half:2]   = torch.sin(rows * div).unsqueeze(1).expand(-1, grid_size, -1)
        pe[:, :, 1:half:2]   = torch.cos(rows * div).unsqueeze(1).expand(-1, grid_size, -1)
        pe[:, :, half::2]    = torch.sin(cols * div).unsqueeze(0).expand(grid_size, -1, -1)
        pe[:, :, half + 1::2]= torch.cos(cols * div).unsqueeze(0).expand(grid_size, -1, -1)

        # (1, N, d_model)
        self.register_buffer("pe", pe.view(1, grid_size * grid_size, d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, d_model) — añadir PE a todos los parches visibles/ocultos
        return x + self.pe[:, : x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """
    Embeddings posicionales aprendibles (uno por posición de parche).
    Estándar en ViT y BERT.
    """

    def __init__(self, num_patches: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

    def forward(self, x: Tensor, ids: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x:
            (B, N', d_model) — puede ser un subconjunto de posiciones.
        ids:
            (B, N') — índices originales de cada parche en x.
            Si None, se asume que x contiene todos los parches en orden.
        """
        if ids is None:
            return x + self.pe[:, : x.size(1)]
        # Seleccionar PE para las posiciones indicadas
        return x + self.pe[:, ids]


class MLP(nn.Module):
    """
    Feed-Forward MLP de 2 capas con activación GELU.
    Utilizado dentro de cada bloque transformer.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ViTBlock(nn.Module):
    """
    Bloque transformer Pre-LN estándar:
      x = x + DropPath(Attn(LN(x)))
      x = x + DropPath(MLP(LN(x)))

    Pre-LayerNorm (norm_first=True) mejora la estabilidad del gradiente
    con respecto a la formulación original Post-LN.
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        d_ff:      int,
        dropout:   float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2    = nn.LayerNorm(d_model)
        self.mlp      = MLP(d_model, d_ff, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # Self-attention con Pre-LN
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop_path(h)
        # MLP con Pre-LN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Estrategias de enmascaramiento
# ---------------------------------------------------------------------------

class MaskingStrategy:
    """
    Factoria de estrategias de enmascaramiento de parches.

    Todas devuelven:
      ids_keep   : (B, N_visible) — índices de parches visibles
      ids_masked : (B, N_masked)  — índices de parches enmascarados
      ids_restore: (B, N)         — para restaurar el orden original
    """

    @staticmethod
    def random(
        batch_size: int, num_patches: int, mask_ratio: float, device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Enmascaramiento aleatorio uniforme (MAE original)."""
        N = num_patches
        N_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(batch_size, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)

        ids_keep   = ids_shuffle[:, :N_keep]
        ids_masked = ids_shuffle[:, N_keep:]

        return ids_keep, ids_masked, ids_restore

    @staticmethod
    def block(
        batch_size: int,
        num_patches: int,
        mask_ratio: float,
        grid_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Enmascaramiento por bloques contiguos.

        Selecciona aleatoriamente un punto de inicio en el grid y enmascara
        un bloque rectangular de aproximadamente mask_ratio parches.
        Más difícil que el aleatorio → representaciones más ricas.
        """
        N = num_patches
        N_masked_target = int(N * mask_ratio)
        block_side = max(1, int(math.sqrt(N_masked_target)))

        all_keep:   List[Tensor] = []
        all_masked: List[Tensor] = []
        all_restore: List[Tensor] = []

        for _ in range(batch_size):
            # Generar máscara 2D
            mask_2d = torch.zeros(grid_size, grid_size, dtype=torch.bool, device=device)
            r0 = torch.randint(0, max(1, grid_size - block_side + 1), (1,)).item()
            c0 = torch.randint(0, max(1, grid_size - block_side + 1), (1,)).item()
            r1 = min(grid_size, r0 + block_side)
            c1 = min(grid_size, c0 + block_side)
            mask_2d[r0:r1, c0:c1] = True

            flat   = mask_2d.view(-1)              # (N,)
            masked = flat.nonzero(as_tuple=False).squeeze(1)
            keep   = (~flat).nonzero(as_tuple=False).squeeze(1)
            full   = torch.cat([keep, masked])
            restore = torch.argsort(full)

            all_keep.append(keep)
            all_masked.append(masked)
            all_restore.append(restore)

        # Pad al mismo tamaño (el bloque puede variar un parche entre muestras)
        N_keep   = min(t.shape[0] for t in all_keep)
        N_masked = min(t.shape[0] for t in all_masked)

        ids_keep    = torch.stack([t[:N_keep]   for t in all_keep])
        ids_masked  = torch.stack([t[:N_masked] for t in all_masked])
        ids_restore = torch.stack(all_restore)

        return ids_keep, ids_masked, ids_restore

    @staticmethod
    def grid(
        batch_size: int, num_patches: int, grid_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Enmascaramiento en cuadrícula: enmascara 1 de cada 4 parches
        de forma fija (útil para tráfico con periodicidad espacial).
        """
        mask_2d = torch.zeros(grid_size, grid_size, dtype=torch.bool, device=device)
        mask_2d[::2, ::2] = True   # cada dos posiciones en ambos ejes → ~25% enmascarado

        flat    = mask_2d.view(-1)
        masked  = flat.nonzero(as_tuple=False).squeeze(1)
        keep    = (~flat).nonzero(as_tuple=False).squeeze(1)
        full    = torch.cat([keep, masked])
        restore = torch.argsort(full)

        ids_keep    = keep.unsqueeze(0).expand(batch_size, -1)
        ids_masked  = masked.unsqueeze(0).expand(batch_size, -1)
        ids_restore = restore.unsqueeze(0).expand(batch_size, -1)

        return ids_keep, ids_masked, ids_restore


# ---------------------------------------------------------------------------
# Modelo principal
# ---------------------------------------------------------------------------

class TrafficViT(GenerativeModel):
    """
    Vision Transformer generativo (MAE-style) para tráfico de red.

    Entrenamiento  : reconstrucción MAE de parches enmascarados.
    Representación : encoder ViT produce embeddings latentes densos (encode()).
    Generación     : desenmascaramiento iterativo tipo MAGE (generate()).

    Métodos públicos
    ----------------
    build()                → construye arquitectura y mueve al device.
    configure_optimizers() → devuelve un optimizer (único, a diferencia del GAN).
    train_step(batch)      → paso MAE; devuelve loss + métricas.
    encode(images)         → embeddings latentes del encoder (sin máscara).
    generate(n_samples)    → imágenes sintéticas mediante desenmascaramiento iterativo.
    get_metrics_schema()   → esquema canónico de métricas producidas.

    Compatibilidad con el pipeline secuencial
    -----------------------------------------
    - Misma herencia de GenerativeModel.
    - train_step devuelve siempre "loss" como clave principal.
    - generate() acepta n_samples, temperature, labels.
    """

    def __init__(self, config: Optional[ViTConfig] = None) -> None:
        if config is None:
            config = ViTConfig()
        super().__init__(config)
        self.cfg = config
        self._ema_encoder: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Propiedades abstractas
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> ModelType:
        return ModelType.AUTOREGRESSIVE   # reutilizamos el enum disponible

    @property
    def input_domain(self) -> InputDomain:
        return InputDomain.DISCRETE_SEQUENCE   # override si existe IMAGE en el enum

    # ------------------------------------------------------------------
    # build()
    # ------------------------------------------------------------------

    def build(self) -> "TrafficViT":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)

        # --- Patch embedding ---
        self.patch_emb = PatchEmbedding(cfg)

        # --- Positional encoding (encoder) ---
        if cfg.pos_encoding == "learnable":
            self.pos_enc_enc = LearnablePositionalEncoding(cfg.num_patches, cfg.d_model)
        else:
            self.pos_enc_enc = PositionalEncoding2D(cfg.d_model, cfg.grid_size)

        # --- CLS token (representación global del encoder) ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # --- Condicionamiento por clase ---
        if cfg.num_classes > 0:
            self.label_emb  = nn.Embedding(cfg.num_classes, cfg.cond_dim)
            self.label_proj = nn.Linear(cfg.cond_dim, cfg.d_model)
        else:
            self.label_emb  = None
            self.label_proj = None

        # --- Bloques encoder ---
        # Stochastic depth: incrementar drop_path linealmente capa a capa
        dpr = [
            x.item()
            for x in torch.linspace(0, cfg.drop_path_rate, cfg.n_layers)
        ]
        self.encoder_blocks = nn.ModuleList([
            ViTBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, dpr[i])
            for i in range(cfg.n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(cfg.d_model)

        # --- Proyección encoder→decoder (MAE) ---
        self.enc_to_dec = nn.Linear(cfg.d_model, cfg.decoder_d_model, bias=False)

        # --- Token [MASK] aprendible del decoder ---
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.decoder_d_model) * 0.02)

        # --- Positional encoding (decoder) —siempre sobre N parches completos ---
        self.pos_enc_dec = nn.Embedding(cfg.num_patches, cfg.decoder_d_model)

        # --- Bloques decoder ---
        self.decoder_blocks = nn.ModuleList([
            ViTBlock(
                cfg.decoder_d_model,
                cfg.decoder_n_heads,
                cfg.decoder_d_ff,
                cfg.dropout,
                drop_path=0.0,  # sin drop_path en el decoder (más ligero)
            )
            for _ in range(cfg.decoder_n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(cfg.decoder_d_model)

        # --- Cabeza de reconstrucción de píxeles ---
        # Salida: (B, N_masked, patch_dim) — parche aplanado
        self.pixel_head = nn.Linear(cfg.decoder_d_model, cfg.patch_dim, bias=True)

        # --- EMA del encoder (para encode() en inferencia) ---
        if cfg.ema_decay > 0.0:
            self._build_ema_encoder()

        # Registrar todos los módulos
        self._networks = {
            "patch_emb":      self.patch_emb,
            "pos_enc_enc":    self.pos_enc_enc,
            "encoder_blocks": self.encoder_blocks,
            "encoder_norm":   self.encoder_norm,
            "enc_to_dec":     self.enc_to_dec,
            "decoder_blocks": self.decoder_blocks,
            "decoder_norm":   self.decoder_norm,
            "pixel_head":     self.pixel_head,
        }

        self._init_weights()

        for net in self._networks.values():
            net.to(self.device)

        self._built = True
        LOGGER.info("TrafficViT construido: %s", self)
        return self

    # ------------------------------------------------------------------
    # Inicialización de pesos
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Inicialización estándar ViT (Dosovitskiy et al., 2020)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    def _build_ema_encoder(self) -> None:
        """Crea una copia del encoder para EMA (sin gradientes)."""
        cfg = self.cfg
        ema = nn.ModuleDict({
            "patch_emb":      PatchEmbedding(cfg),
            "pos_enc_enc":    (
                LearnablePositionalEncoding(cfg.num_patches, cfg.d_model)
                if cfg.pos_encoding == "learnable"
                else PositionalEncoding2D(cfg.d_model, cfg.grid_size)
            ),
            "encoder_blocks": nn.ModuleList([
                ViTBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, 0.0)
                for _ in range(cfg.n_layers)
            ]),
            "encoder_norm": nn.LayerNorm(cfg.d_model),
        })
        ema.load_state_dict(
            {k: v for k, v in self.state_dict().items() if k.split(".")[0] in ema},
            strict=False,
        )
        for p in ema.parameters():
            p.requires_grad_(False)
        self._ema_encoder = ema.to(self.device)

    def _update_ema(self) -> None:
        """Actualiza pesos EMA tras cada paso de optimización."""
        if self._ema_encoder is None:
            return
        decay = self.cfg.ema_decay
        src_map = {
            "patch_emb":      self.patch_emb,
            "encoder_blocks": self.encoder_blocks,
            "encoder_norm":   self.encoder_norm,
            "pos_enc_enc":    self.pos_enc_enc,
        }
        with torch.no_grad():
            for key, src_module in src_map.items():
                ema_module = self._ema_encoder[key]
                for ema_p, src_p in zip(ema_module.parameters(), src_module.parameters()):
                    ema_p.data.mul_(decay).add_(src_p.data, alpha=1.0 - decay)

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _apply_mask(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Delega a la estrategia configurada en cfg.mask_strategy."""
        cfg = self.cfg
        if cfg.mask_strategy == "random":
            return MaskingStrategy.random(batch_size, cfg.num_patches, cfg.mask_ratio, device)
        elif cfg.mask_strategy == "block":
            return MaskingStrategy.block(
                batch_size, cfg.num_patches, cfg.mask_ratio, cfg.grid_size, device
            )
        elif cfg.mask_strategy == "grid":
            return MaskingStrategy.grid(batch_size, cfg.num_patches, cfg.grid_size, device)
        else:
            raise ValueError(f"mask_strategy desconocida: {cfg.mask_strategy!r}")

    # ------------------------------------------------------------------
    # Condicionamiento
    # ------------------------------------------------------------------

    def _resolve_labels(
        self,
        n_samples: int,
        labels: Optional[Union[int, Tensor, List[int], Tuple[int, ...]]],
    ) -> Optional[Tensor]:
        if self.cfg.num_classes == 0:
            return None
        if labels is None:
            return torch.randint(0, self.cfg.num_classes, (n_samples,), device=self.device)
        if isinstance(labels, int):
            return torch.full((n_samples,), labels, dtype=torch.long, device=self.device)
        if isinstance(labels, (list, tuple)):
            y = torch.tensor(labels, dtype=torch.long, device=self.device)
        elif torch.is_tensor(labels):
            y = labels.to(self.device).long()
        else:
            raise TypeError(f"Tipo de labels no soportado: {type(labels)}")
        if y.shape[0] != n_samples:
            raise ValueError(f"labels tiene {y.shape[0]} elementos pero n_samples={n_samples}")
        return y

    def _build_cond_token(
        self, labels: Optional[Tensor], batch_size: int, d_model: int
    ) -> Optional[Tensor]:
        """Devuelve un token de clase (B, 1, d_model) o None."""
        if labels is None or self.label_emb is None:
            return None
        return self.label_proj(self.label_emb(labels)).unsqueeze(1)  # (B, 1, D)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _encode(
        self,
        images: Tensor,
        ids_keep: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_ema: bool = False,
    ) -> Tensor:
        """
        Codifica imágenes con el encoder ViT.

        Parameters
        ----------
        images:
            (B, C, H, W) en rango normalizado.
        ids_keep:
            (B, N_visible) — índices de parches a mantener.
            Si None, se pasan todos los parches (modo encode puro).
        labels:
            (B,) etiquetas de clase para condicionamiento.
        use_ema:
            Si True y EMA disponible, usa el encoder EMA.

        Returns
        -------
        (B, N_visible [+ 1_cls [+ 1_cond]], d_model)
        """
        patch_emb      = self.patch_emb
        pos_enc        = self.pos_enc_enc
        encoder_blocks = self.encoder_blocks
        encoder_norm   = self.encoder_norm

        if use_ema and self._ema_encoder is not None:
            patch_emb      = self._ema_encoder["patch_emb"]
            pos_enc        = self._ema_encoder["pos_enc_enc"]
            encoder_blocks = self._ema_encoder["encoder_blocks"]
            encoder_norm   = self._ema_encoder["encoder_norm"]

        B = images.shape[0]
        x = patch_emb(images)   # (B, N, D)

        # Añadir PE a todos los parches antes de filtrar
        if isinstance(pos_enc, LearnablePositionalEncoding):
            x = pos_enc(x)          # (B, N, D)
        else:
            x = pos_enc(x)

        # Filtrar parches visibles
        if ids_keep is not None:
            idx = ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = torch.gather(x, 1, idx)             # (B, N_vis, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)           # (B, N_vis+1, D)

        # Condicionamiento (mode: "prefix")
        cond_token = self._build_cond_token(labels, B, self.cfg.d_model)
        if cond_token is not None and self.cfg.cond_mode == "prefix":
            x = torch.cat([cond_token, x], dim=1)  # (B, N_vis+2, D)

        # Condicionamiento (mode: "add")
        if cond_token is not None and self.cfg.cond_mode == "add":
            x = x + cond_token   # broadcast sobre la longitud

        # Pasar por los bloques transformer
        for block in encoder_blocks:
            x = block(x)
        x = encoder_norm(x)

        return x   # (B, N_vis + [1|2], D)

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _decode(
        self,
        latent: Tensor,
        ids_masked: Tensor,
        ids_restore: Tensor,
        n_prefix: int,
    ) -> Tensor:
        """
        Decoder MAE: reconstruye los parches enmascarados.

        Parameters
        ----------
        latent:
            (B, N_vis + n_prefix, D_enc) — salida del encoder.
        ids_masked:
            (B, N_masked) — posiciones enmascaradas.
        ids_restore:
            (B, N) — permutación inversa para restaurar orden original.
        n_prefix:
            Número de tokens de prefijo (CLS + opcional clase) que
            deben excluirse antes de pasar al decoder.

        Returns
        -------
        (B, N_masked, patch_dim) — reconstrucción de los parches ocultos.
        """
        B = latent.shape[0]
        N = ids_restore.shape[1]

        # Proyectar al espacio del decoder
        tokens  = latent[:, n_prefix:]              # (B, N_vis, D_enc)
        tokens  = self.enc_to_dec(tokens)           # (B, N_vis, D_dec)

        # Crear tokens [MASK] para posiciones ocultas
        N_masked = ids_masked.shape[1]
        mask_tokens = self.mask_token.expand(B, N_masked, -1)  # (B, N_masked, D_dec)

        # Restaurar el orden original combinando parches visibles y [MASK]
        full = torch.cat([tokens, mask_tokens], dim=1)          # (B, N, D_dec)
        idx  = ids_restore.unsqueeze(-1).expand(-1, -1, full.size(-1))
        full = torch.gather(full, 1, idx)                        # (B, N, D_dec)

        # Añadir PE posicional
        pos_ids = torch.arange(N, device=latent.device).unsqueeze(0).expand(B, -1)
        full    = full + self.pos_enc_dec(pos_ids)               # (B, N, D_dec)

        # Pasar por los bloques del decoder
        for block in self.decoder_blocks:
            full = block(full)
        full = self.decoder_norm(full)                           # (B, N, D_dec)

        # Proyectar sólo las posiciones enmascaradas
        idx_m = ids_masked.unsqueeze(-1).expand(-1, -1, full.size(-1))
        masked_out = torch.gather(full, 1, idx_m)               # (B, N_masked, D_dec)

        return self.pixel_head(masked_out)                       # (B, N_masked, patch_dim)

    # ------------------------------------------------------------------
    # Pérdida de reconstrucción
    # ------------------------------------------------------------------

    def _recon_loss(
        self,
        images: Tensor,
        pred: Tensor,
        ids_masked: Tensor,
    ) -> Tensor:
        """
        Pérdida pixel-level sobre los parches enmascarados.

        Parameters
        ----------
        images:
            (B, C, H, W) — imagen original.
        pred:
            (B, N_masked, patch_dim) — parches reconstruidos.
        ids_masked:
            (B, N_masked) — posiciones enmascaradas.

        Returns
        -------
        Escalar — pérdida media sobre parches y píxeles enmascarados.
        """
        B, C, H, W = images.shape
        P = self.cfg.patch_size
        G = self.cfg.grid_size

        # Extraer parches target desde la imagen original: (B, N, patch_dim)
        # Usamos unfold sobre la imagen: más eficiente que reshape manual
        target = images.unfold(2, P, P).unfold(3, P, P)         # (B,C,G,G,P,P)
        target = target.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B,G,G,C,P,P)
        target = target.view(B, G * G, C * P * P)                # (B,N,patch_dim)

        # Seleccionar sólo los parches enmascarados
        idx_m  = ids_masked.unsqueeze(-1).expand(-1, -1, target.size(-1))
        target = torch.gather(target, 1, idx_m)                  # (B,N_masked,patch_dim)

        # Normalizar target por parche (práctica del MAE original)
        if self.cfg.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1e-6).sqrt()

        # Seleccionar función de pérdida
        if self.cfg.recon_loss == "mse":
            loss = F.mse_loss(pred, target)
        elif self.cfg.recon_loss == "mae":
            loss = F.l1_loss(pred, target)
        elif self.cfg.recon_loss == "smooth_l1":
            loss = F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"recon_loss desconocida: {self.cfg.recon_loss!r}")

        return loss

    # ------------------------------------------------------------------
    # Optimizador
    # ------------------------------------------------------------------

    def configure_optimizers(
        self, lr: float = 1.5e-4
    ) -> Dict[str, torch.optim.Optimizer]:
        """
        AdamW con weight decay, siguiendo las recomendaciones del paper MAE.

        A diferencia del GAN (dos optimizadores separados), el ViT usa uno solo.
        """
        # Separar parámetros con y sin weight decay
        # No aplicar WD a biases, LayerNorm y embeddings posicionales
        decay_params     = []
        no_decay_params  = []
        no_decay_names   = {"bias", "cls_token", "mask_token", "pe"}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay_names) or isinstance(
                self.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self,
                nn.LayerNorm,
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": 0.05},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        )
        return {"vit": optimizer}

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: Any) -> Dict[str, Tensor]:
        """
        Paso de entrenamiento MAE.

        Parameters
        ----------
        batch:
            - Tensor (B, C, H, W)
            - tupla  (images, labels)
            - dict   {"images": ..., "labels": ...}

        Returns
        -------
        Dict con:
          - ``loss``           : pérdida de reconstrucción (principal).
          - ``recon_loss``     : alias de loss.
          - ``mask_ratio_eff`` : fracción de parches realmente enmascarados.
        """
        self._check_built()

        # --- Extraer imágenes y etiquetas del batch ---
        images, labels = self._extract_batch(batch)
        images = images.to(self.device)
        if labels is not None:
            labels = labels.to(self.device).long()

        B = images.shape[0]

        # --- Enmascaramiento ---
        ids_keep, ids_masked, ids_restore = self._apply_mask(B, self.device)

        # --- Contar prefijos (CLS + posible clase) ---
        n_prefix = 1  # siempre CLS
        if self.cfg.num_classes > 0 and self.cfg.cond_mode == "prefix":
            n_prefix += 1

        # --- Forward encoder ---
        latent = self._encode(images, ids_keep=ids_keep, labels=labels)

        # --- Forward decoder ---
        pred = self._decode(latent, ids_masked, ids_restore, n_prefix)

        # --- Pérdida ---
        loss = self._recon_loss(images, pred, ids_masked) * self.cfg.recon_weight

        mask_ratio_eff = ids_masked.shape[1] / self.cfg.num_patches

        # --- Actualizar EMA ---
        self._update_ema()

        return {
            "loss":           loss,
            "recon_loss":     loss.detach(),
            "mask_ratio_eff": torch.tensor(mask_ratio_eff, device=self.device),
        }

    # ------------------------------------------------------------------
    # encode()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        images: Tensor,
        labels: Optional[Union[int, Tensor, List[int], Tuple[int, ...]]] = None,
        use_ema: bool = True,
        return_cls: bool = True,
    ) -> Tensor:
        """
        Extrae representaciones latentes sin enmascaramiento.

        Parameters
        ----------
        images:
            (B, C, H, W) imágenes normalizadas.
        labels:
            Etiquetas de clase opcionales.
        use_ema:
            Usar encoder EMA para features más estables (recomendado en inferencia).
        return_cls:
            Si True, devuelve solo el token CLS (B, D).
            Si False, devuelve todos los tokens del encoder (B, N+[1|2], D).

        Returns
        -------
        Tensor — representación latente.
        """
        self._check_built()
        self.eval_mode()

        images = images.to(self.device)
        y = self._resolve_labels(images.shape[0], labels)

        latent = self._encode(images, ids_keep=None, labels=y, use_ema=use_ema)

        if return_cls:
            # Token CLS es el segundo si hay token de clase, el primero en otro caso
            cls_idx = 1 if (self.cfg.num_classes > 0 and self.cfg.cond_mode == "prefix") else 0
            return latent[:, cls_idx, :]  # (B, D)

        return latent  # (B, N+1[+1], D)

    # ------------------------------------------------------------------
    # generate()  — Desenmascaramiento iterativo (MAGE-inspired)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        temperature: Optional[float] = None,
        labels: Optional[Union[int, Tensor, List[int], Tuple[int, ...]]] = None,
        generation_steps: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """
        Genera imágenes sintéticas mediante desenmascaramiento iterativo.

        Algoritmo (MAGE-inspired)
        -------------------------
        1. Inicializar todos los parches como [MASK] (sin parches visibles).
        2. En cada paso t de T:
           a. Pasar por encoder (sólo con prefijos CLS/clase, 0 parches visibles).
           b. Reconstruir todos los parches con el decoder.
           c. Convertir la predicción a espacio pixel (desnormalizar si aplica).
           d. Calcular score de confianza por parche = -entropía del softmax.
           e. Revelar el top-(t/T)·N parches más confiados.
           f. Los parches revelados se embeben y se pasan como visibles en t+1.
        3. Devolver imágenes reconstruidas.

        Parameters
        ----------
        n_samples:
            Número de imágenes a generar.
        temperature:
            Temperatura para el muestreo de parches (None = cfg.temperature).
        labels:
            Condicionamiento de clase (None = muestreo uniforme si condicional).
        generation_steps:
            Pasos de refinamiento iterativo (None = cfg.generation_steps).
        top_k, top_p:
            Filtros de muestreo sobre la distribución de parches.

        Returns
        -------
        Tensor (n_samples, C, H, W) — imágenes generadas en rango normalizado.
        """
        self._check_built()
        self.eval_mode()

        cfg   = self.cfg
        T     = generation_steps if generation_steps is not None else cfg.generation_steps
        temp  = temperature      if temperature      is not None else cfg.temperature
        k     = top_k            if top_k            is not None else cfg.top_k
        p     = top_p            if top_p            is not None else cfg.top_p

        B     = n_samples
        N     = cfg.num_patches
        P     = cfg.patch_size
        G     = cfg.grid_size
        C     = cfg.in_channels
        dev   = self.device

        y = self._resolve_labels(B, labels)

        # Número de prefijos del encoder
        n_prefix = 1
        if cfg.num_classes > 0 and cfg.cond_mode == "prefix":
            n_prefix += 1

        # Buffer de parches reconstruidos: (B, N, patch_dim)
        patch_buffer = torch.zeros(B, N, cfg.patch_dim, device=dev)

        # Máscara de estado: True = enmascarado (aún no revelado)
        still_masked = torch.ones(B, N, dtype=torch.bool, device=dev)

        for step in range(T):
            # Cuántos parches revelar en este paso (programación coseno)
            ratio_revealed = math.cos(math.pi / 2 * (T - step - 1) / T)
            n_reveal_total = max(1, int(ratio_revealed * N))
            n_visible      = N - still_masked[0].sum().item()

            if n_visible >= N:
                break   # todos los parches revelados

            # --- Construir entrada del encoder con los parches ya revelados ---
            ids_keep_list = []
            for b in range(B):
                visible_idx = (~still_masked[b]).nonzero(as_tuple=False).squeeze(1)
                ids_keep_list.append(visible_idx)

            if all(len(ids) == 0 for ids in ids_keep_list):
                # Primer paso: sin parches visibles → sólo CLS + clase
                # Codificar con N=0 parches visibles usando un placeholder
                # (sólo los prefijos irán al decoder)
                latent = self._encode_prefix_only(B, y, n_prefix, dev)
            else:
                # Pad ids_keep al mismo tamaño dentro del batch
                max_vis = max(len(ids) for ids in ids_keep_list)
                ids_keep_padded = torch.stack([
                    F.pad(ids, (0, max_vis - len(ids)), value=0)
                    for ids in ids_keep_list
                ])
                # Construir imágenes parciales desde patch_buffer
                partial_images = self._patches_to_images(patch_buffer, G, P, C, B, dev)
                latent = self._encode(partial_images, ids_keep=ids_keep_padded, labels=y)

            # --- Decoder: reconstruir todos los parches enmascarados ---
            ids_masked_all = still_masked.nonzero(as_tuple=False)
            if ids_masked_all.numel() == 0:
                break

            # Reconstruir en modo "todos enmascarados" para obtener scores
            ids_masked_batch = torch.stack([
                still_masked[b].nonzero(as_tuple=False).squeeze(1)
                for b in range(B)
            ])  # (B, N_masked)

            ids_restore = self._build_restore(still_masked, B, N, dev)

            pred = self._decode(latent, ids_masked_batch, ids_restore, n_prefix)
            # pred: (B, N_masked, patch_dim)

            # --- Calcular scores de confianza ---
            # Score proxy: varianza inversa de la predicción (parches más "seguros")
            pred_var = pred.var(dim=-1)                # (B, N_masked)
            conf_score = -pred_var                     # mayor confianza = menor varianza

            # Opcionalmente muestrear con temperatura
            if temp != 1.0:
                conf_score = conf_score / temp

            # --- Revelar los parches más confiados ---
            n_to_reveal = min(n_reveal_total - int(n_visible), ids_masked_batch.shape[1])
            if n_to_reveal <= 0:
                continue

            _, top_local = torch.topk(conf_score, n_to_reveal, dim=1)  # (B, n_to_reveal)

            for b in range(B):
                global_indices = ids_masked_batch[b][top_local[b]]
                patch_buffer[b, global_indices] = pred[b, top_local[b]]
                still_masked[b, global_indices] = False

        # Rellenar cualquier parche que quedara enmascarado con la última predicción
        # (cierre de bucle por seguridad)
        for b in range(B):
            remaining = still_masked[b].nonzero(as_tuple=False).squeeze(1)
            if remaining.numel() > 0:
                # Predicción final sin condición de revelado
                patch_buffer[b, remaining] = torch.randn(
                    remaining.numel(), cfg.patch_dim, device=dev
                ) * 0.1

        # Ensamblar imagen final desde los parches
        images_out = self._patches_to_images(patch_buffer, G, P, C, B, dev)
        return images_out  # (B, C, H, W)

    # ------------------------------------------------------------------
    # Helpers de generate()
    # ------------------------------------------------------------------

    def _encode_prefix_only(
        self, B: int, labels: Optional[Tensor], n_prefix: int, device: torch.device
    ) -> Tensor:
        """Encoder con 0 parches visibles: sólo CLS (y clase si aplica)."""
        # Crear un token dummy para dar contexto al decoder
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = cls
        if labels is not None and self.cfg.cond_mode == "prefix":
            cond = self._build_cond_token(labels, B, self.cfg.d_model)
            x = torch.cat([cond, x], dim=1)       # (B, 2, D)
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_norm(x)               # (B, n_prefix, D)

    @staticmethod
    def _build_restore(
        still_masked: Tensor, B: int, N: int, device: torch.device
    ) -> Tensor:
        """
        Construye ids_restore a partir del estado de máscara actual.
        ids_restore deshace la permutación [visible | masked].
        """
        restore_list = []
        for b in range(B):
            visible = (~still_masked[b]).nonzero(as_tuple=False).squeeze(1)
            masked  = still_masked[b].nonzero(as_tuple=False).squeeze(1)
            perm    = torch.cat([visible, masked])
            restore = torch.argsort(perm)
            restore_list.append(restore)
        return torch.stack(restore_list)   # (B, N)

    @staticmethod
    def _patches_to_images(
        patch_buffer: Tensor,
        G: int, P: int, C: int, B: int, device: torch.device,
    ) -> Tensor:
        """
        Reensambla parches en imágenes.

        patch_buffer: (B, N, C*P*P)
        Returns:      (B, C, G*P, G*P)
        """
        H = W = G * P
        x = patch_buffer.view(B, G, G, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5)        # (B, C, G, P, G, P)
        x = x.contiguous().view(B, C, H, W)
        return x

    # ------------------------------------------------------------------
    # Extracción de batch
    # ------------------------------------------------------------------

    def _extract_batch(
        self, batch: Any
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Misma interfaz que TrafficGAN._extract_batch."""
        if isinstance(batch, Tensor):
            return batch, None
        if isinstance(batch, (tuple, list)):
            images = batch[0]
            labels = batch[1] if len(batch) > 1 else None
            return images, labels
        if isinstance(batch, dict):
            images = batch.get("images", batch.get("x"))
            if images is None:
                raise ValueError("No se encontró clave 'images' o 'x' en el batch.")
            labels = batch.get("labels", batch.get("y", batch.get("label")))
            return images, labels
        raise TypeError(f"Tipo de batch no soportado: {type(batch)}")

    # ------------------------------------------------------------------
    # Esquema de métricas  [coherente con TrafficGAN.get_metrics_schema]
    # ------------------------------------------------------------------

    @staticmethod
    def get_metrics_schema() -> Dict[str, Optional[str]]:
        """
        Esquema canónico de métricas producidas por train_step.

        Permite a trainers y loggers iterar las métricas de forma
        declarativa sin hardcodear nombres de clave.

        Uso en un trainer
        -----------------
        >>> schema  = TrafficViT.get_metrics_schema()
        >>> metrics = {k: out.get(k) for k in schema}
        """
        return {
            "loss":           "Pérdida principal de reconstrucción MAE.",
            "recon_loss":     "Alias de loss; misma escala, facilita comparación entre runs.",
            "mask_ratio_eff": "Fracción real de parches enmascarados en este batch.",
        }
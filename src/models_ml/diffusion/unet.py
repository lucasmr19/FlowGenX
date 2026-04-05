"""
generative_models/diffusion/unet.py
=====================================
UNet2D para modelos de difusión sobre imágenes de tráfico de red.

Arquitectura
------------
  Input (B, C, H, W) + time embedding t
        ↓
  [DownBlock x n_levels]  — ResBlock + (opcional) SelfAttention + Downsample
        ↓
  [BottleneckBlock]       — ResBlock + SelfAttention + ResBlock
        ↓
  [UpBlock x n_levels]    — ResBlock + (opcional) SelfAttention + Upsample
        ↓
  Output (B, C, H, W)     — predice el ruido ε añadido en el paso t

Compatibilidad
--------------
  - NprintRepresentation  (H=max_packets, W=total_bits, C=1) ← INVERTIBLE
  - GASFRepresentation    (H=W=image_size, C=2)              ← NO invertible

Inspirado en: NetDiffusion (Jiang et al., 2024),
              NetDiffus (Sivaroopan et al., 2024),
              DDPM (Ho et al., 2020), Improved DDPM (Nichol & Dhariwal, 2021).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Embedding sinusoidal del timestep t → vector de dimensión dim.

    Convierte un escalar t ∈ {0, ..., T-1} en un vector denso
    que se inyecta en cada ResidualBlock del UNet.

    Formulación: igual que el positional encoding de Vaswani et al.
    pero aplicado al tiempo de difusión.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim debe ser par para sinusoidal embedding"
        self.dim = dim

        # Proyección MLP tras el embedding sinusoidal
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        t : (B,) — timesteps enteros en [0, T)

        Returns
        -------
        emb : (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        ).float()
        args = t[:, None].float() * freqs[None]         # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# Class / protocol conditioning embedding (para NetDiffusion-style)
# ---------------------------------------------------------------------------

class ConditioningEmbedding(nn.Module):
    """
    Embedding de etiqueta de clase (protocolo / tipo de tráfico).

    Permite generar tráfico condicionado a un protocolo específico
    (TCP, UDP, HTTP, DNS, etc.), similar al enfoque de NetDiffusion.

    Se suma al time embedding para un condicionamiento aditivo simple.
    """

    def __init__(self, num_classes: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, dim)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, labels: Tensor) -> Tensor:
        """labels: (B,) enteros en [0, num_classes)"""
        return self.proj(self.embedding(labels))


# ---------------------------------------------------------------------------
# ResidualBlock con time conditioning
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Bloque residual con:
      - GroupNorm + SiLU (activación más estable que ReLU en difusión)
      - Inyección del time embedding mediante suma (shift)
      - Projection shortcut si in_ch != out_ch
    """

    def __init__(
        self,
        in_ch:     int,
        out_ch:    int,
        time_dim:  int,
        num_groups: int = 8,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        # num_groups debe dividir in_ch y out_ch
        g_in  = min(num_groups, in_ch)
        g_out = min(num_groups, out_ch)

        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2   = nn.GroupNorm(g_out, out_ch)
        self.conv2   = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

        # Proyección del time embedding al número de canales
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        # Shortcut
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

        self.act = nn.SiLU()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        x     : (B, in_ch, H, W)
        t_emb : (B, time_dim)
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Inyectar time embedding (shift)
        h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


# ---------------------------------------------------------------------------
# Self-Attention Block (para resoluciones bajas)
# ---------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention 2D para el UNet.

    Se aplica solo a resoluciones bajas (bottleneck y algunos UpBlocks)
    para capturar dependencias globales sin coste cuadrático extremo.
    """

    def __init__(self, channels: int, n_heads: int = 4, num_groups: int = 8) -> None:
        super().__init__()
        assert channels % n_heads == 0
        g = min(num_groups, channels)
        self.norm    = nn.GroupNorm(g, channels)
        self.attn    = nn.MultiheadAttention(
            embed_dim   = channels,
            num_heads   = n_heads,
            batch_first = True,
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        h = self.norm(x)

        # Aplanar dimensiones espaciales → secuencia
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        h, _ = self.attn(h, h, h)                     # self-attention
        h = h.permute(0, 2, 1).reshape(B, C, H, W)   # (B, C, H, W)
        h = self.proj_out(h)

        return x + h   # residual


# ---------------------------------------------------------------------------
# Bloques Down / Up
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """
    Bloque de bajada:
      ResidualBlock (x n_res) → [SelfAttention] → AvgPool2d (stride 2)

    Guarda el skip connection ANTES del downsampling.
    """

    def __init__(
        self,
        in_ch:      int,
        out_ch:     int,
        time_dim:   int,
        n_res:      int  = 2,
        use_attn:   bool = False,
        n_heads:    int  = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_ch if i == 0 else out_ch,
                out_ch, time_dim, dropout=dropout
            )
            for i in range(n_res)
        ])
        self.attn = SelfAttentionBlock(out_ch, n_heads) if use_attn else None
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        skip     : (B, out_ch, H, W)   — antes del downsampling
        downsampled : (B, out_ch, H//2, W//2)
        """
        for res in self.res_blocks:
            x = res(x, t_emb)
        if self.attn is not None:
            x = self.attn(x)
        skip = x
        return skip, self.downsample(x)


class UpBlock(nn.Module):
    """
    Bloque de subida:
      Upsample (x2) → concatenar skip → ResidualBlock (x n_res) → [SelfAttention]
    """

    def __init__(
        self,
        in_ch:      int,
        skip_ch:    int,
        out_ch:     int,
        time_dim:   int,
        n_res:      int  = 2,
        use_attn:   bool = False,
        n_heads:    int  = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # Tras concatenar con skip: in_ch + skip_ch canales
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                (in_ch + skip_ch) if i == 0 else out_ch,
                out_ch, time_dim, dropout=dropout
            )
            for i in range(n_res)
        ])
        self.attn = SelfAttentionBlock(out_ch, n_heads) if use_attn else None

    def forward(self, x: Tensor, skip: Tensor, t_emb: Tensor) -> Tensor:
        x = self.upsample(x)

        # Recortar si hay diferencia de tamaño (padding impar)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)
        for res in self.res_blocks:
            x = res(x, t_emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


# ---------------------------------------------------------------------------
# UNet2D completo
# ---------------------------------------------------------------------------

class UNet2D(nn.Module):
    """
    UNet 2D para predecir el ruido ε en el proceso de difusión.

    Parametrización
    ---------------
    channel_mults  : multiplicadores de canales por nivel.
                     Ej: (1, 2, 4, 8) crea 4 resoluciones.
    attention_levels: índices de niveles donde se aplica self-attention.
                     En resoluciones bajas (últimos niveles).
    num_classes    : >0 activa el condicionamiento por clase (NetDiffusion-style).

    Ejemplo de configuración para nprint (20x193):
    >>> unet = UNet2D(in_channels=1, base_ch=64,
    ...               channel_mults=(1,2,4), attention_levels=(2,))

    Ejemplo para GASF (2x64x64):
    >>> unet = UNet2D(in_channels=2, base_ch=64,
    ...               channel_mults=(1,2,4,8), attention_levels=(2,3))
    """

    def __init__(
        self,
        in_channels:      int,
        base_ch:          int           = 64,
        channel_mults:    Tuple[int, ...] = (1, 2, 4, 8),
        n_res_per_level:  int           = 2,
        attention_levels: Tuple[int, ...] = (2, 3),
        n_heads:          int           = 4,
        dropout:          float         = 0.1,
        num_classes:      int           = 0,    # 0 = sin condicionamiento
    ) -> None:
        super().__init__()

        time_dim = base_ch * 4

        # --- Time embedding ---
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        # --- Class conditioning ---
        self.class_emb: Optional[ConditioningEmbedding] = None
        if num_classes > 0:
            self.class_emb = ConditioningEmbedding(num_classes, time_dim)

        # --- Convolución inicial ---
        self.input_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # --- Encoder (Down) ---
        self.down_blocks = nn.ModuleList()
        ch_in = base_ch
        skip_channels: List[int] = []

        for level, mult in enumerate(channel_mults):
            ch_out   = base_ch * mult
            use_attn = level in attention_levels
            self.down_blocks.append(DownBlock(
                ch_in, ch_out, time_dim,
                n_res=n_res_per_level,
                use_attn=use_attn,
                n_heads=n_heads,
                dropout=dropout,
            ))
            skip_channels.append(ch_out)
            ch_in = ch_out

        # --- Bottleneck ---
        self.bottleneck = nn.ModuleList([
            ResidualBlock(ch_in, ch_in, time_dim, dropout=dropout),
            SelfAttentionBlock(ch_in, n_heads),
            ResidualBlock(ch_in, ch_in, time_dim, dropout=dropout),
        ])

        # --- Decoder (Up) ---
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            ch_out   = base_ch * (channel_mults[level - 1] if level > 0 else 1)
            skip_ch  = skip_channels[level]
            use_attn = level in attention_levels
            self.up_blocks.append(UpBlock(
                ch_in, skip_ch, ch_out, time_dim,
                n_res=n_res_per_level,
                use_attn=use_attn,
                n_heads=n_heads,
                dropout=dropout,
            ))
            ch_in = ch_out

        # --- Salida ---
        self.output_norm = nn.GroupNorm(min(8, ch_in), ch_in)
        self.output_conv = nn.Conv2d(ch_in, in_channels, kernel_size=3, padding=1)
        self.output_act  = nn.SiLU()

    def forward(
        self,
        x:      Tensor,
        t:      Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x      : (B, C, H, W) — imagen ruidosa en el paso t
        t      : (B,)         — timesteps
        labels : (B,) optional — etiquetas de clase para condicionamiento

        Returns
        -------
        (B, C, H, W) — ruido predicho ε_θ(x_t, t)
        """
        # Time embedding
        t_emb = self.time_emb(t)

        # Class conditioning (suma al time embedding)
        if self.class_emb is not None and labels is not None:
            t_emb = t_emb + self.class_emb(labels)

        # Encoder
        x = self.input_conv(x)
        skips = []
        for down in self.down_blocks:
            skip, x = down(x, t_emb)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck[0](x, t_emb)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t_emb)

        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, t_emb)

        # Salida
        x = self.output_act(self.output_norm(x))
        return self.output_conv(x)
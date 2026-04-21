"""
representations/vision/nprint_image.py
=====================================

Compact visual representation of nPrint traffic features using patch-based encoding.

---------------------------------------------------------------------
1. Overview
---------------------------------------------------------------------
NprintImageRepresentation converts the native nPrint tensor:

    (max_packets, total_bits)  with values in {-1, 0, 1}

into a compact, image-like tensor:

    (C, H, W) = (3|4, max_packets, n_patches)

suitable for generative models such as DDPM, GANs, or Transformers,
without modifying the rest of the framework.

---------------------------------------------------------------------
2. Motivation
---------------------------------------------------------------------
The original nPrint representation (~1024 x 1609) is not suitable for
attention-based architectures due to:

    • Quadratic complexity O(N²)
    • Excessive VRAM usage

Previous work (NetDiffusion, Jiang et al., 2024) addresses this by
mapping the tensor into an image and using Stable Diffusion.

This implementation follows the same principle but introduces a more
structured and information-preserving encoding.

---------------------------------------------------------------------
3. Representation Definition
---------------------------------------------------------------------

The representation is based on patching contiguous bits:

    (N_packets, W_bits) → (N_packets, n_patches, patch_size)

Each patch is encoded into 3 or 4 channels:

    ch0 — Presence:
        Indicates whether bits in the patch are present (not -1).
        Mode "mean": fraction of present bits ∈ [0, 1]
        Mode "any":  1.0 if any bit is present, else 0.0   [DEFAULT]
        Mode "max":  same as "any" (max of bool mask)

    ch1 — Bit Value:
        Mean of bits equal to 1 among present bits.
        Range: [0, 1]  — interpretable as P(bit=1 | present)

    ch2 — Protocol Group:
        Normalized structural prior per patch.
        Static mode: mean of normalized group IDs ∈ [0, 1]
        Embedding mode (use_embedding_ch2=True):
            Learnable scalar nn.Embedding(n_protocols, 1), initialized
            with the same normalized values. Allows the model to adjust
            the structural prior during fine-tuning.

    ch3 — Intra-Patch Variance  [OPTIONAL, enabled by default]:
        Variance of bit values within the patch among present bits.
        Range: [0, 0.25]  — distinguishes [1,1,1,1] (var=0) vs
        [1,0,1,0] (var=0.25), which ch0+ch1 cannot separate.

Final tensor shape:

    use_ch3_variance=True  → Tensor(4, max_packets, n_patches)
    use_ch3_variance=False → Tensor(3, max_packets, n_patches)

---------------------------------------------------------------------
4. Encoding Pipeline
---------------------------------------------------------------------

PacketWindow
  → NprintRepresentation.encode()
        → (N_packets, total_bits) ∈ {-1, 0, 1}

  → Field exclusion (STRUCTURAL_EXCLUDE)
        → (N_packets, W_masked)

  → Trim to multiple of patch_size
        → (N_packets, W_trimmed)

  → Patchify
        → (3|4, N_packets, n_patches)

  → Pad to pad_to_height rows  [optional]
        → (3|4, pad_to_height, n_patches)

No interpolation or spatial distortion is applied.

---------------------------------------------------------------------
5. Decoding Pipeline (Approximate)
---------------------------------------------------------------------

Tensor(3|4, N_packets, n_patches)
  → Thresholding or Bernoulli sampling:
        ch0 ≥ thr_presence  or  Bernoulli(ch0) → bits present
        ch1 ≥ thr_bit_one   or  Bernoulli(ch1) → bit = 1

  → Patch expansion:
        repeat_interleave(patch_size)

  → Scatter to original bit positions

  → NprintRepresentation.decode()

Note:
    Reconstruction is approximate due to intra-patch aggregation.
    Bernoulli sampling is stochastic by design.

---------------------------------------------------------------------
6. Design Choices and Improvements
---------------------------------------------------------------------

ch0 Mode: "any" vs "mean"
--------------------------
"mean" smooths the presence signal across the patch; a single present
bit produces ch0 < 1.0.  "any"/"max" uses a binary signal: the patch
is either fully present or absent, which better matches how nPrint
populates protocol fields (all bits of a field are present or none).

Bernoulli Decode
----------------
Hard thresholding (>= 0.5) collapses the stochastic channel into a
deterministic function, breaking compatibility with diffusion models
that output continuous distributions.  Bernoulli sampling preserves
the generative nature of the model:

    bit ~ Bernoulli(ch1)     presence ~ Bernoulli(ch0)

This makes the decode step a proper probabilistic inverse of encode.

ch3 Intra-Patch Variance
-------------------------
ch0 and ch1 cannot distinguish [1,1,1,1] (uniform active) from
[1,0,1,0] (alternating).  ch3 = E[(b - ĉ1)²] over present bits
adds this discriminative signal at negligible cost.

Learnable ch2 (nn.Embedding)
-----------------------------
The static protocol-group encoding is a fixed positional prior.
Replacing it with a learnable nn.Embedding(n_proto, 1) allows
the representation to adjust the structural weight of each protocol
during end-to-end fine-tuning, at the cost of introducing trainable
parameters into the preprocessor.  Disabled by default.

Uniform Height Padding
-----------------------
When pad_to_height is set, PacketWindows with fewer packets are padded with
pad_value (typically -1) to reach a fixed height.  This guarantees
a constant (C, H, W) shape regardless of PacketWindow length, required by
batch loaders that stack tensors without masking.

---------------------------------------------------------------------
7. Limitations
---------------------------------------------------------------------

• Intra-patch information loss:
    Bit-level patterns within each patch are not fully recoverable
    (only mean and variance are preserved, not individual bits).

• Approximate decoding:
    Bernoulli sampling introduces stochasticity; multiple decodes of
    the same tensor may yield different packets.

• Protocol channel (ch2):
    Static mode provides a positional prior but no dynamic signal.
    Learnable mode requires gradient PacketWindow into the representation.

---------------------------------------------------------------------
8. Output Shape Example
---------------------------------------------------------------------

With default parameters (use_ch3_variance=True):

    total_bits (full nPrint) : 1609
    after exclusion          : ~997
    patch_size               : 4

    n_patches = 996 / 4 = 249

    output shape = (4, 128, 249)

---------------------------------------------------------------------
9. Integration
---------------------------------------------------------------------

The representation is fully compatible with the existing framework:

    NprintImageRepresentation
        └── NprintRepresentation
                └── TrafficRepresentation

All downstream modules (loaders, evaluation, generative models)
operate on Tensor(C, H, W) without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .nprint import NprintConfig, NprintRepresentation
from ...preprocessing import PacketWindow, ParsedPacket
from ...utils.logger_config import LOGGER


# ---------------------------------------------------------------------------
# Protocol group mapping for ch2  (normalised 0 → 1)
# ---------------------------------------------------------------------------
_PROTO_LABELS: List[str] = ["eth", "ipv4", "ipv6", "tcp", "udp", "icmp", "meta"]
_N_PROTOCOLS: int = len(_PROTO_LABELS)

_PROTO_GROUP: Dict[str, float] = {
    label: (i + 1) / _N_PROTOCOLS
    for i, label in enumerate(_PROTO_LABELS)
}

# Integer IDs used by nn.Embedding (0 = unknown, 1-7 = protocols)
_PROTO_INT: Dict[str, int] = {
    label: i + 1
    for i, label in enumerate(_PROTO_LABELS)
}
_EMBEDDING_VOCAB_SIZE: int = _N_PROTOCOLS + 1  # +1 for unknown (idx 0)


def _field_proto_group_float(field_name: str) -> float:
    """Return the normalised group float for a field name."""
    for prefix, gid in _PROTO_GROUP.items():
        if field_name.startswith(prefix):
            return gid
    return 0.0


def _field_proto_group_int(field_name: str) -> int:
    """Return the integer embedding index for a field name."""
    for prefix, idx in _PROTO_INT.items():
        if field_name.startswith(prefix):
            return idx
    return 0  # unknown


# ---------------------------------------------------------------------------
# Default exclusion set — follows NetDiffusion criteria
# ---------------------------------------------------------------------------
#
# Exclusion criteria:
#   · IPs / MACs      — host/network identifiers, not generalisable
#   · src + dst ports — session-specific; remove tcp_dprt/udp_dport from
#                       this set to preserve the service-port signal
#   · TCP seq / ack   — randomly initialised, no learnable pattern
#   · Checksums       — deterministic functions of the payload
#
STRUCTURAL_EXCLUDE: FrozenSet[str] = frozenset({
    # MACs
    "eth_dhost", "eth_shost",
    # IPs
    "ipv4_src", "ipv4_dst",
    "ipv6_src", "ipv6_dst",
    # Ports (src + dst, TCP + UDP)
    "tcp_sprt", "tcp_dprt",
    "udp_sport", "udp_dport",
    # TCP sequence / acknowledgement numbers
    "tcp_seq", "tcp_ackn",
    # Checksums
    "ipv4_cksum", "tcp_cksum", "udp_cksum", "icmp_cksum",
})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NprintImageConfig(NprintConfig):
    """
    Extends NprintConfig with visual-compression parameters.

    Specific parameters
    -------------------
    patch_size : int
        Number of consecutive bits compressed into a single «pixel».
        Default 4 → n_patches ≈ 249 → output (3|4, 128, 249).

    max_packets : int  (overrides NprintConfig default → 128)
        Number of row-packets in the image.

    excluded_fields : FrozenSet[str]
        Fields excluded before patchifying.
        Default: STRUCTURAL_EXCLUDE.  Pass frozenset() to use all fields.

    ch0_mode : Literal["mean", "any", "max"]
        How to compute the presence channel per patch.
        "mean"  → fraction of present bits  (original behaviour)
        "any"   → 1.0 if any bit is present [DEFAULT — matches nPrint semantics]
        "max"   → alias for "any"

    use_ch3_variance : bool
        If True, append a 4th channel with intra-patch bit variance.
        Distinguishes [1,1,1,1] (var=0) from [1,0,1,0] (var=0.25).
        Default: True.

    use_bernoulli_decode : bool
        If True, decode uses Bernoulli sampling from ch0 / ch1 instead
        of hard thresholds.  Preserves stochastic nature of generative
        models.  Default: True.

    use_embedding_ch2 : bool
        If True, ch2 is computed via a learnable nn.Embedding(vocab, 1)
        initialised with the static normalised group values.
        Requires gradient PacketWindow into the representation.
        Default: False (static normalised float).

    thr_presence : float
        Hard-threshold for ch0 when use_bernoulli_decode=False.

    thr_bit_one : float
        Hard-threshold for ch1 when use_bernoulli_decode=False.

    pad_to_height : Optional[int]
        If set, pad the packet dimension to this fixed height using
        pad_value.  Ensures constant (C, H, W) for batch loaders.
        Default: None (no padding beyond max_packets).
    """
    representation_type:str = "nprint_image"
    name: str = "nprint_image"

    # Patchify
    patch_size: int = 4

    # Reduced from 1024 → 128 packets for manageable shape
    max_packets: int = 128

    # Field-level exclusion mask
    excluded_fields: FrozenSet[str] = field(
        default_factory=lambda: STRUCTURAL_EXCLUDE
    )

    # ------------------------------------------------------------------ NEW
    # ch0 computation mode
    ch0_mode: Literal["mean", "any", "max"] = "any"

    # ch3: intra-patch variance channel
    use_ch3_variance: bool = True

    # Stochastic decode via Bernoulli sampling
    use_bernoulli_decode: bool = False

    # Learnable embedding for ch2
    use_embedding_ch2: bool = False

    # Uniform height padding
    pad_to_height: Optional[int] = None
    # ------------------------------------------------------------------ /NEW

    # Hard-threshold fallback (used only when use_bernoulli_decode=False)
    thr_presence: float = 0.35
    thr_bit_one:  float = 0.50


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class NprintImageRepresentation(NprintRepresentation):
    """
    Compressed nPrint → rectangular image representation.

    Encode PacketWindow
    -----------
    PacketWindow
      → NprintRepresentation.encode()     → Tensor(N_pkt, total_bits)
      → _apply_exclusion_mask()           → Tensor(N_pkt, W_masked)
      → _patchify()                       → Tensor(C, N_pkt, n_patches)
      → _pad_height()       [optional]    → Tensor(C, H, n_patches)

    Decode PacketWindow (approximate / stochastic)
    ---------------------------------------
    Tensor(C, N_pkt, n_patches)
      → _unpatchify()      → Tensor(N_pkt, W_trimmed)   {-1, 0, 1}
      → _scatter_mask()    → Tensor(N_pkt, total_bits)
      → super().decode()   → List[ParsedPacket]

    Channels
    --------
    ch0 : presence  (mode: any|max|mean)
    ch1 : bit value  P(bit=1 | present)
    ch2 : protocol group  (static float or learnable embedding)
    ch3 : intra-patch variance  [optional, enabled by default]
    """

    def __init__(self, config: Optional[NprintImageConfig] = None) -> None:
        if config is None:
            config = NprintImageConfig()
        super().__init__(config)
        self.img_cfg: NprintImageConfig = config

        # Boolean inclusion mask: (total_bits,)
        self._excl_mask: np.ndarray = self._build_exclusion_mask()
        self._W_masked: int = int(self._excl_mask.sum())

        # Trim to exact multiple of patch_size
        P = self.img_cfg.patch_size
        self._W_trimmed: int = (self._W_masked // P) * P
        self._n_patches: int = self._W_trimmed // P

        # Static protocol-group buffers (always built; used as fallback / init)
        self._proto_group_buf: Tensor = self._build_proto_group_tensor()
        self._proto_int_buf: Tensor = self._build_proto_int_tensor()

        # Learnable ch2 embedding  (vocab=8, dim=1)
        self._ch2_embedding: Optional[nn.Embedding] = None
        if self.img_cfg.use_embedding_ch2:
            self._ch2_embedding = self._build_ch2_embedding()

        LOGGER.info(
            "NprintImage init: total_bits=%d → masked=%d → trimmed=%d "
            "→ n_patches=%d  output_shape=%s  "
            "ch0_mode=%s  ch3=%s  bernoulli=%s  embedding=%s  pad_h=%s",
            self._total_bits, self._W_masked, self._W_trimmed,
            self._n_patches, self.output_shape,
            self.img_cfg.ch0_mode,
            self.img_cfg.use_ch3_variance,
            self.img_cfg.use_bernoulli_decode,
            self.img_cfg.use_embedding_ch2,
            self.img_cfg.pad_to_height,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        """Number of image channels: 3 (base) + 1 if ch3 variance enabled."""
        return 4 if self.img_cfg.use_ch3_variance else 3

    @property
    def output_shape(self) -> Tuple[int, ...]:
        H = self.img_cfg.pad_to_height or self.img_cfg.max_packets
        return (self.n_channels, H, self._n_patches)

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def encode(self, sample: PacketWindow) -> Tensor:
        """
        PacketWindow → Tensor(C, H, n_patches)  in [0, 1].

        C = 3 or 4 depending on use_ch3_variance.
        H = max_packets  or  pad_to_height if set.
        """
        self._check_fitted()

        # 1. Base nPrint tensor: (max_packets, total_bits)  ∈ {-1, 0, 1}
        raw: Tensor = super().encode(sample)

        # 2. Exclusion mask → (max_packets, W_masked)
        excl_t = torch.from_numpy(self._excl_mask)
        masked = raw[:, excl_t]

        # 3. Trim to multiple of patch_size → (N, W_trimmed)
        masked = masked[:, : self._W_trimmed]

        # 4. Patchify → (C, N, n_patches)
        img = self._patchify(masked)

        # 5. Optional: pad to uniform height
        if self.img_cfg.pad_to_height is not None:
            img = self._pad_height(img)

        LOGGER.debug(
            "encode: %d pkts → shape=%s  min=%.3f max=%.3f",
            len(sample.packets), tuple(img.shape), img.min().item(), img.max().item(),
        )
        return img

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, tensor: Tensor) -> List[ParsedPacket]:
        """
        Tensor(C, H, n_patches) → List[ParsedPacket].

        When use_bernoulli_decode=True, multiple calls on the same
        tensor may yield different results (stochastic inverse).
        """
        self._check_fitted()

        # Strip padding rows if pad_to_height was applied
        N = self.img_cfg.max_packets
        tensor = tensor[:, :N, :]  # (C, max_packets, n_patches)

        # 1. Unpatchify → (N, W_trimmed)  ∈ {-1, 0, 1}
        recovered = self._unpatchify(tensor)

        # 2. Re-pad to W_masked if trim dropped bits
        pad = self._W_masked - self._W_trimmed
        if pad > 0:
            padding = torch.full(
                (recovered.shape[0], pad),
                self.img_cfg.pad_value,
                dtype=recovered.dtype,
                device=recovered.device,
            )
            recovered = torch.cat([recovered, padding], dim=1)

        # 3. Scatter back to total_bits width
        device = recovered.device

        full = torch.full(
            (N, self._total_bits),
            self.img_cfg.pad_value,
            dtype=torch.float32,
            device=device,
        )
        excl_t = torch.from_numpy(self._excl_mask).to(device)
        full[:, excl_t] = recovered

        # 4. Delegate to parent decoder
        return self._decode_from_bit_matrix(full)
    
    def get_default_aggregator(self):
        from ...preprocessing import PacketWindowAggregator
        return PacketWindowAggregator
    
    def _fast_project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proyección aproximada al manifold de nprintimage.
        Recupera geométricamente la forma de imagen válida sin pasar por decode/encode,
        por lo que no recupera semanticamente.
        Espera forma (B, C, H, W).
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        assert x.ndim == 4, f"Expected (B,C,H,W), got {x.shape}"

        x = x.clone()

        # Clamp global
        x = torch.clamp(x, 0.0, 1.0)

        # --- Canal 0: presencia ---
        x[:, 0] = (x[:, 0] > 0.5).float()

        # --- Canal 1: probabilidad ---
        # (no tocar)

        # --- Canal 2: protocolo ---
        K = getattr(self.cfg, "num_protocol_bins", 8)
        x[:, 2] = torch.round(x[:, 2] * (K - 1)) / (K - 1)

        # --- Canal 3: varianza ---
        if x.shape[1] > 3:
            x[:, 3] = torch.clamp(x[:, 3], 0.0, 0.25)

        return x
    
    def _encode_packets(self, packets: List[ParsedPacket]) -> Tensor:
        flow = PacketWindow(
            packets=packets)
        return self.encode(flow)
    
    def project(self, x: torch.Tensor, exact: bool = False, **kwargs) -> torch.Tensor:
        """
        Proyecta muestras generadas al espacio válido de la representación.

        Estrategia:
            decode → encode  (proyección al manifold válido)
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if not exact:
            return self._fast_project(x)

        # proyección exacta
        projected = []
        for sample in x:
            decoded = self.decode(sample)
            reencoded = self.encode(decoded)
            projected.append(reencoded)

        return torch.stack(projected)

    # ------------------------------------------------------------------
    # _patchify
    # ------------------------------------------------------------------

    def _patchify(self, masked: Tensor) -> Tensor:
        """
        masked : (N, W_trimmed)  values in {-1, 0, 1}
        returns: (C, N, n_patches)  values in [0, 1]

        Channel layout
        --------------
        ch0  presence   — mode-dependent (any | max | mean)
        ch1  bit value  — P(bit=1 | present)
        ch2  protocol   — static float or learnable embedding scalar
        ch3  variance   — intra-patch bit variance  [optional]
        """
        N, W = masked.shape
        P = self.img_cfg.patch_size
        n_patches = W // P

        # (N, n_patches, P)
        patches = masked.view(N, n_patches, P)

        # ---- ch0: presence ---------------------------------------------------
        present_mask = (patches > -0.5)                          # bool (N, n, P)

        if self.img_cfg.ch0_mode == "mean":
            # Original behaviour: fraction of present bits
            ch0 = present_mask.float().mean(dim=-1)              # (N, n_patches)
        else:
            # "any" or "max": binary — 1.0 if at least one bit is present
            # This better matches nPrint field semantics (all-or-nothing)
            ch0 = present_mask.any(dim=-1).float()               # (N, n_patches)

        # ---- ch1: bit value --------------------------------------------------
        bit_one = (patches > 0.5).float()                        # (N, n, P)
        n_present = present_mask.float().sum(dim=-1).clamp(min=1.0)
        ch1 = (bit_one * present_mask.float()).sum(dim=-1) / n_present
        # Zero out patches where no bit is present (avoid div-by-zero artefacts)
        fully_absent = ~present_mask.any(dim=-1)
        ch1[fully_absent] = 0.0                                  # (N, n_patches)

        # ---- ch2: protocol group ---------------------------------------------
        if self.img_cfg.use_embedding_ch2 and self._ch2_embedding is not None:
            # Learnable path: embed integer protocol IDs, squeeze to scalar
            # proto_int_buf: (W_trimmed,) int64
            idx_patches = self._proto_int_buf.view(n_patches, P)   # (n, P)
            # Use the first bit's protocol ID as representative for the patch
            # (all bits in a protocol field share the same ID, so this is exact)
            idx_repr = idx_patches[:, 0]                           # (n_patches,)
            # Embedding output: (n_patches, 1) → squeeze → (n_patches,)
            ch2_row = self._ch2_embedding(idx_repr).squeeze(-1)    # (n_patches,)
            # Clamp to [0, 1] so it stays in image range
            ch2_row = ch2_row.clamp(0.0, 1.0)
        else:
            # Static path: mean of normalised group IDs (original behaviour)
            proto_patches = self._proto_group_buf.view(n_patches, P)
            ch2_row = proto_patches.mean(dim=-1)                   # (n_patches,)

        ch2 = ch2_row.unsqueeze(0).expand(N, -1)                   # (N, n_patches)

        # ---- Build channel stack ---------------------------------------------
        channels = [ch0, ch1, ch2]

        # ---- ch3: intra-patch variance [optional] ----------------------------
        if self.img_cfg.use_ch3_variance:
            # Variance of bit values among present bits only
            # ch1 already holds the mean of present bits → use it as μ
            # Var = E[(b - μ)²] over present bits
            mu = ch1.unsqueeze(-1)                               # (N, n, 1)
            diff_sq = ((bit_one - mu) ** 2) * present_mask.float()
            ch3 = diff_sq.sum(dim=-1) / n_present                # (N, n_patches)
            ch3[fully_absent] = 0.0
            channels.append(ch3)

        return torch.stack(channels, dim=0)                      # (C, N, n_patches)

    # ------------------------------------------------------------------
    # _unpatchify
    # ------------------------------------------------------------------

    def _unpatchify(self, img: Tensor) -> Tensor:
        """
        img : (C, N, n_patches)
        returns: (N, W_trimmed)  with values in {-1, 0, 1}

        Presence and bit-value channels are inverted via:
          • Bernoulli sampling  (use_bernoulli_decode=True)  [DEFAULT]
          • Hard thresholding   (use_bernoulli_decode=False)

        ch3 (variance) and ch2 (protocol) are ignored during decode.
        """
        ch0, ch1 = img[0], img[1]                                # (N, n_patches)
        P = self.img_cfg.patch_size

        if self.img_cfg.use_bernoulli_decode:
            # Stochastic inverse — preserves the generative distribution
            # Clamp to valid probability range before sampling
            p_present = ch0.clamp(0.0, 1.0)
            p_one     = ch1.clamp(0.0, 1.0)

            present  = torch.bernoulli(p_present).bool()         # (N, n_patches)
            bit_val  = torch.bernoulli(p_one)                    # (N, n_patches)
        else:
            # Deterministic inverse via hard thresholds (original behaviour)
            present  = (ch0 >= self.img_cfg.thr_presence)
            bit_val  = (ch1 >= self.img_cfg.thr_bit_one).float()

        bits_per_patch = torch.where(
            present,
            bit_val.float(),
            torch.full_like(bit_val, -1.0),
        )                                                         # (N, n_patches)

        # Expand each patch to P bit-columns
        recovered = bits_per_patch.repeat_interleave(P, dim=1)   # (N, W_trimmed)
        return recovered

    # ------------------------------------------------------------------
    # _pad_height
    # ------------------------------------------------------------------

    def _pad_height(self, img: Tensor) -> Tensor:
        """
        Pad the packet (height) dimension to pad_to_height rows.

        img : (C, N, W)
        returns: (C, pad_to_height, W)

        Padding is appended at the bottom using img_cfg.pad_value,
        normalised to [0, 1]:  -1 → 0.0  (absence of traffic).
        """
        target_H = self.img_cfg.pad_to_height
        C, N, W  = img.shape

        if N >= target_H:
            # Already tall enough — no-op (or trim if over)
            return img[:, :target_H, :]

        pad_rows = target_H - N
        # pad_value is typically -1; in image space absence = 0.0
        fill = 0.0
        pad_tensor = torch.full(
            (C, pad_rows, W), fill, dtype=img.dtype, device=img.device
        )
        padded = torch.cat([img, pad_tensor], dim=1)             # (C, target_H, W)

        LOGGER.debug("_pad_height: %d → %d rows", N, target_H)
        return padded

    # ------------------------------------------------------------------
    # _build_exclusion_mask
    # ------------------------------------------------------------------

    def _build_exclusion_mask(self) -> np.ndarray:
        """
        bool ndarray of shape (total_bits,):
            True  = column included
            False = column excluded
        """
        mask = np.ones(self._total_bits, dtype=bool)
        offset = 0
        for fname, n_bits in self.img_cfg.field_schema:
            if fname in self.img_cfg.excluded_fields:
                mask[offset: offset + n_bits] = False
            offset += n_bits

        n_incl = int(mask.sum())
        LOGGER.debug(
            "exclusion_mask: %d bits excluded → %d retained",
            self._total_bits - n_incl, n_incl,
        )
        if n_incl == 0:
            raise ValueError(
                "excluded_fields removes ALL bits. "
                "Check NprintImageConfig.excluded_fields."
            )
        return mask

    # ------------------------------------------------------------------
    # _build_proto_group_tensor  (static float, for ch2 static mode)
    # ------------------------------------------------------------------

    def _build_proto_group_tensor(self) -> Tensor:
        """
        float32 Tensor of shape (W_trimmed,) with normalised group ID
        per bit after exclusion.  Pre-computed once at init.
        """
        groups: List[float] = []
        for fname, n_bits in self.img_cfg.field_schema:
            if fname in self.img_cfg.excluded_fields:
                continue
            gid = _field_proto_group_float(fname)
            groups.extend([gid] * n_bits)

        groups = groups[: self._W_trimmed]
        return torch.tensor(groups, dtype=torch.float32)

    # ------------------------------------------------------------------
    # _build_proto_int_tensor  (integer IDs, for nn.Embedding mode)
    # ------------------------------------------------------------------

    def _build_proto_int_tensor(self) -> Tensor:
        """
        int64 Tensor of shape (W_trimmed,) with integer protocol IDs
        per bit after exclusion.  Used by the learnable embedding path.
        """
        ids: List[int] = []
        for fname, n_bits in self.img_cfg.field_schema:
            if fname in self.img_cfg.excluded_fields:
                continue
            gid = _field_proto_group_int(fname)
            ids.extend([gid] * n_bits)

        ids = ids[: self._W_trimmed]
        return torch.tensor(ids, dtype=torch.long)

    # ------------------------------------------------------------------
    # _build_ch2_embedding  (learnable, initialised from static values)
    # ------------------------------------------------------------------

    def _build_ch2_embedding(self) -> nn.Embedding:
        """
        Build nn.Embedding(vocab_size=8, embedding_dim=1) initialised
        with the normalised static group values so that the learnable
        path starts from the same prior as the static path.

        Embedding indices
        -----------------
        0  → unknown / no protocol  (0.0)
        1  → eth                    (1/7 ≈ 0.143)
        2  → ipv4                   (2/7 ≈ 0.286)
        3  → ipv6                   (3/7 ≈ 0.429)
        4  → tcp                    (4/7 ≈ 0.571)
        5  → udp                    (5/7 ≈ 0.714)
        6  → icmp                   (6/7 ≈ 0.857)
        7  → meta                   (7/7  = 1.000)
        """
        init_weights = torch.zeros(_EMBEDDING_VOCAB_SIZE, 1)
        init_weights[0, 0] = 0.0  # unknown
        for label, idx in _PROTO_INT.items():
            init_weights[idx, 0] = _PROTO_GROUP[label]

        emb = nn.Embedding(_EMBEDDING_VOCAB_SIZE, 1)
        with torch.no_grad():
            emb.weight.copy_(init_weights)

        LOGGER.debug(
            "ch2 embedding initialised: vocab=%d  weights=%s",
            _EMBEDDING_VOCAB_SIZE, init_weights.squeeze().tolist(),
        )
        return emb

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _get_state_dict(self) -> Dict:
        state = super()._get_state_dict()
        state["excl_mask"] = self._excl_mask.tolist()
        if self._ch2_embedding is not None:
            state["ch2_embedding_weights"] = (
                self._ch2_embedding.weight.detach().cpu().tolist()
            )
        return state

    def _set_state_dict(self, state: Dict) -> None:
        super()._set_state_dict(state)

        self._excl_mask = np.array(state["excl_mask"], dtype=bool)
        self._W_masked  = int(self._excl_mask.sum())

        P = self.img_cfg.patch_size
        self._W_trimmed = (self._W_masked // P) * P
        self._n_patches = self._W_trimmed // P

        self._proto_group_buf = self._build_proto_group_tensor()
        self._proto_int_buf   = self._build_proto_int_tensor()

        if self.img_cfg.use_embedding_ch2:
            self._ch2_embedding = self._build_ch2_embedding()
            if "ch2_embedding_weights" in state:
                weights = torch.tensor(
                    state["ch2_embedding_weights"], dtype=torch.float32
                )
                with torch.no_grad():
                    self._ch2_embedding.weight.copy_(weights)
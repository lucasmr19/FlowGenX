"""
generative_models/transformer/model.py
=======================================
Transformer autoregresivo GPT-style para generación de tráfico secuencial.

Arquitectura
------------
  TokenEmbedding + SinusoidalPositionalEncoding
        ↓
  N x TransformerDecoderLayer  (causal mask, sin cross-attention)
        ↓
  LM Head (Linear → vocab_size)

Compatibilidad de representación
---------------------------------
  - FlatTokenizer           (tokens discretos, vocabulario campo:valor)
  - ProtocolAwareTokenizer  (tokens con separadores de capa)

Entrada:  (B, L)  — IDs enteros
Salida:   (B, L, V) — logits sobre vocabulario

Generación: autoregresiva token a token con top-k / top-p / greedy.

Inspirado en: NetGPT (Meng et al., 2023), TrafficGPT (Qu et al., 2024).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import (
    GenerativeModel,
    InputDomain,
    ModelType,
)

from .config import TransformerConfig

from ...utils.logger_config import LOGGER

# ---------------------------------------------------------------------------
# Bloques de la arquitectura
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    """Embedding de tokens con escalado √d_model (estilo Vaswani et al. 2017)."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model   = d_model
        self.scale     = math.sqrt(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L) → (B, L, d_model)
        return self.embedding(x) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal (no aprendida).
    Añade información de posición absoluta a los embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Precompute PE matrix shape: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)   # (L, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Modelo principal
# ---------------------------------------------------------------------------

class TrafficTransformer(GenerativeModel):
    """
    Transformer autoregresivo GPT-style para tráfico de red.

    Reutiliza nn.TransformerDecoder de PyTorch (sin cross-attention:
    memory = tgt, equivalente a un decoder-only / GPT).

    Entrenamiento: NLL sobre el siguiente token (teacher forcing).
    Generación:    autoregresiva con top-k y/o nucleus sampling.

    Example
    -------
    >>> cfg   = TransformerConfig(vocab_size=5000, d_model=256, n_layers=4)
    >>> model = TrafficTransformer(cfg).build()
    >>> batch = torch.randint(0, 5000, (8, 128))   # (B, L)
    >>> losses = model.train_step(batch)
    >>> print(losses["loss"].item())
    >>> samples = model.generate(n_samples=4, max_new_tokens=128)
    """

    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        if config is None:
            config = TransformerConfig()
        super().__init__(config)
        self.cfg = config

    # ------------------------------------------------------------------
    # Propiedades abstractas
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> ModelType:
        return ModelType.AUTOREGRESSIVE

    @property
    def input_domain(self) -> InputDomain:
        return InputDomain.DISCRETE_SEQUENCE
    
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
            raise TypeError(f"Unsupported labels type: {type(labels)}")

        if y.shape[0] != n_samples:
            raise ValueError(f"labels has {y.shape[0]} elements but n_samples={n_samples}")

        return y

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self) -> "TrafficTransformer":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)

        # --- Capas de embedding ---
        self.token_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc   = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_seq_len, cfg.dropout
        )

        # --- Decoder Transformer ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = cfg.d_model,
            nhead           = cfg.n_heads,
            dim_feedforward = cfg.d_ff,
            dropout         = cfg.dropout,
            batch_first     = True,    # (B, L, d_model) — más intuitivo
            norm_first      = True,    # Pre-LN: más estable que Post-LN
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers    = cfg.n_layers,
            norm          = nn.LayerNorm(cfg.d_model),
        )

        # --- Cabeza de lenguaje ---
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: el lm_head comparte pesos con token_emb
        # (reduce parámetros y mejora generalización)
        self.lm_head.weight = self.token_emb.embedding.weight

        # Registrar módulos para save/load
        self._networks = {
            "token_emb":  self.token_emb,
            "pos_enc":    self.pos_enc,
            "transformer": self.transformer,
            "lm_head":    self.lm_head,
        }

        # Inicialización de pesos
        self._init_weights()

        # Mover al device
        for net in self._networks.values():
            net.to(self.device)

        self._built = True
        LOGGER.info("TrafficTransformer construido: %s", self)
        return self

    def _init_weights(self) -> None:
        """Inicialización estándar tipo GPT."""
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:      Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns logits aligned with input_ids length.
        If conditional, the conditioning prefix is prepended in embedding space,
        but the returned logits exclude the prefix position.
        
        Parameters
        ----------
        input_ids      : (B, L) — IDs de tokens
        attention_mask : (B, L) — 1 = token real, 0 = padding (opcional)

        Returns
        -------
        logits : (B, L, vocab_size)
        """
        B, L = input_ids.shape

        cond_emb = None
        if self.cfg.num_classes > 0:
            if labels is None:
                labels = torch.randint(0, self.cfg.num_classes, (B,), device=input_ids.device)
            else:
                labels = labels.to(input_ids.device).long()
                if labels.shape[0] != B:
                    raise ValueError(f"labels batch size {labels.shape[0]} != input batch size {B}")
            cond_emb = self.label_proj(self.label_emb(labels)).unsqueeze(1)  # (B,1,D)

        x = self.token_emb(input_ids)  # (B,L,D)

        if cond_emb is not None:
            x = torch.cat([cond_emb, x], dim=1)  # (B,L+1,D)

        x = self.pos_enc(x)

        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)
            key_padding_mask = (attention_mask == 0)
            if cond_emb is not None:
                cond_mask = torch.ones((B, 1), dtype=torch.bool, device=input_ids.device)
                key_padding_mask = torch.cat([cond_mask, key_padding_mask], dim=1)
        else:
            key_padding_mask = None

        causal_mask = self._make_causal_mask(x.size(1), input_ids.device)

        out = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
        )

        logits = self.lm_head(out)  # (B, L+1, V) if conditional, else (B, L, V)
        if self.cfg.num_classes > 0:
            logits = logits[:, 1:, :]  # remove conditioning position
        return logits

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: Any) -> Dict[str, Tensor]:
        """
        Un paso forward de NLL (next-token prediction).

        Parameters
        ----------
        batch : Tensor (B, L) de IDs enteros,
                o tupla (input_ids, attention_mask).

        Returns
        -------
        {"loss": nll_loss, "perplexity": ppl}
        """
        self._check_built()

        if isinstance(batch, (tuple, list)):
            input_ids = batch[0].to(self.device)
            labels = batch[1].to(self.device) if len(batch) > 1 else None
        else:
            input_ids = batch.to(self.device)
            labels = None

        attention_mask = (input_ids != self.cfg.pad_token_id).long()

        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        mask = attention_mask[:, 1:]

        logits = self.forward(inputs, attention_mask=mask, labels=labels)

        loss = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            ignore_index=self.cfg.pad_token_id,
        )

        with torch.no_grad():
            ppl = torch.exp(loss.detach())

        return {"loss": loss, "perplexity": ppl}

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        n_samples:      int,
        max_new_tokens: int   = 256,
        bos_token_id:   int   = 2,
        eos_token_id:   int   = 3,
        temperature:    Optional[float] = None,
        top_k:          Optional[int]   = None,
        top_p:          Optional[float] = None,
        labels: Optional[Union[int, Tensor, List[int], Tuple[int, ...]]] = None,
    ) -> Tensor:
        """
        Genera n_samples secuencias de forma autoregresiva.

        Parameters
        ----------
        n_samples      : número de secuencias a generar
        max_new_tokens : longitud máxima de generación
        bos_token_id   : token de inicio (default: índice de <BOS>)
        eos_token_id   : token de fin (default: índice de <EOS>)
        temperature    : temperatura de muestreo (None = usa config)
        top_k          : filtro top-k (None = usa config)
        top_p          : filtro nucleus (None = usa config)

        Returns
        -------
        Tensor (n_samples, max_new_tokens) de IDs enteros
        """
        self._check_built()
        self.eval_mode()
        
        y_gen = self._resolve_labels(n_samples, labels)

        temp  = temperature if temperature is not None else self.cfg.temperature
        k     = top_k       if top_k       is not None else self.cfg.top_k
        p     = top_p       if top_p       is not None else self.cfg.top_p

        # Inicializar con BOS
        generated = torch.full(
            (n_samples, 1), bos_token_id,
            dtype=torch.long, device=self.device
        )
        finished = torch.zeros(n_samples, dtype=torch.bool, device=self.device)

        for _ in range(max_new_tokens - 1):
            logits = self.forward(generated, labels=y_gen)      # (B, cur_len, V)
            next_logits = logits[:, -1, :]        # (B, V) — último token

            # Temperatura
            if temp != 1.0:
                next_logits = next_logits / temp

            # Top-k
            if k > 0:
                next_logits = self._top_k_filter(next_logits, k)

            # Nucleus (top-p)
            if p < 1.0:
                next_logits = self._top_p_filter(next_logits, p)

            probs      = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Marcar secuencias que ya acabaron
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        return generated

    # ------------------------------------------------------------------
    # Helpers de muestreo
    # ------------------------------------------------------------------

    @staticmethod
    def _make_causal_mask(size: int, device: torch.device) -> Tensor:
        """
        Máscara causal triangular superior.
        nn.TransformerDecoder espera True donde NO atender.
        """
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1
        )
        return mask

    @staticmethod
    def _top_k_filter(logits: Tensor, k: int) -> Tensor:
        """Pone -inf a todos los tokens fuera del top-k."""
        threshold = logits.topk(k, dim=-1).values[:, -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _top_p_filter(logits: Tensor, p: float) -> Tensor:
        """Nucleus sampling: mantiene los tokens que suman probabilidad p."""
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Eliminar tokens que superen el umbral (excepto el primero)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) > p
        sorted_logits[remove] = float("-inf")

        # Restaurar orden original
        return sorted_logits.scatter(1, sorted_idx, sorted_logits)
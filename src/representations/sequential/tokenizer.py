"""
representations/sequential/tokenizer.py
========================================
Representaciones secuenciales del tráfico de red.

Implementa dos variantes según el diseño del TFG:
  1. FlatTokenizer           — tokenización byte-level o field-level plana
  2. ProtocolAwareTokenizer  — tokenización jerárquica consciente del protocolo
                               (inspirada en NetGPT)

Ambas heredan de TrafficRepresentation y devuelven tensores de tokens
con forma (max_length,) listos para un Transformer autoregresivo.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ...data_utils.preprocessing import Flow, ParsedPacket
from ..base import (
    Invertibility,
    RepresentationConfig,
    RepresentationType,
    TrafficRepresentation,
)

from ...utils.logger_config import LOGGER


# ---------------------------------------------------------------------------
# Tokens especiales
# ---------------------------------------------------------------------------

PAD_TOKEN   = "<PAD>"    # relleno hasta max_length
UNK_TOKEN   = "<UNK>"    # token desconocido (fuera de vocabulario)
BOS_TOKEN   = "<BOS>"    # inicio de secuencia
EOS_TOKEN   = "<EOS>"    # fin de secuencia
SEP_TOKEN   = "<SEP>"    # separador entre paquetes
MASK_TOKEN  = "<MASK>"   # para masked language modeling (opcional)
DIR_FWD     = "<FWD>"    # dirección forward
DIR_BWD     = "<BWD>"    # dirección backward

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
                  SEP_TOKEN, MASK_TOKEN, DIR_FWD, DIR_BWD]


# ---------------------------------------------------------------------------
# Vocabulario
# ---------------------------------------------------------------------------

class TokenVocabulary:
    """
    Mapa bidireccional token (str) <-> ID (int).
    """

    def __init__(self, max_vocab_size: int = 10_000) -> None:
        self.max_vocab_size = max(max_vocab_size, len(SPECIAL_TOKENS))
        self._token2id: Dict[str, int] = {}
        self._id2token: Dict[int, str] = {}
        self._built = False

        for tok in SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self._token2id:
            idx = len(self._token2id)
            self._token2id[token] = idx
            self._id2token[idx] = token
        return self._token2id[token]

    def add_token(self, token: str) -> int:
        if token in self._token2id:
            return self._token2id[token]
        if len(self._token2id) >= self.max_vocab_size:
            return self._token2id[UNK_TOKEN]
        return self._add(token)

    def build_from_corpus(self, token_sequences: List[List[str]]) -> "TokenVocabulary":
        counter: Counter = Counter()

        for seq in token_sequences:
            counter.update(seq)

        for tok in SPECIAL_TOKENS:
            counter.pop(tok, None)

        most_common = counter.most_common(self.max_vocab_size - len(SPECIAL_TOKENS))

        for token, _ in most_common:
            if len(self._token2id) >= self.max_vocab_size:
                break
            self.add_token(token)

        self._built = True
        LOGGER.info("Vocabulario construido: %d tokens", len(self))
        return self

    def encode_sequence(self, tokens: List[str]) -> List[int]:
        return [
            self._token2id.get(tok, self.unk_id)
            for tok in tokens
        ]

    def decode_sequence(self, ids: List[int]) -> List[str]:
        return [
            self._id2token.get(i, UNK_TOKEN)
            for i in ids
        ]

    def __len__(self) -> int:
        return len(self._token2id)

    # -----------------------------
    # Properties
    # -----------------------------
    @property
    def pad_id(self) -> int:
        return self._token2id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._token2id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._token2id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._token2id[EOS_TOKEN]

    @property
    def sep_id(self) -> int:
        return self._token2id[SEP_TOKEN]

    @property
    def mask_id(self) -> int:
        return self._token2id[MASK_TOKEN]

    @property
    def fwd_id(self) -> int:
        return self._token2id[DIR_FWD]

    @property
    def bwd_id(self) -> int:
        return self._token2id[DIR_BWD]

    @property
    def vocab_size(self) -> int:
        return len(self._token2id)
# ---------------------------------------------------------------------------
# Configuración secuencial
# ---------------------------------------------------------------------------

@dataclass
class SequentialConfig(RepresentationConfig):
    """Parámetros para representaciones secuenciales."""
    name: str = "sequential"

    # Vocabulario
    max_vocab_size: int = 10_000

    # Granularidad de tokenización
    # "field"  → un token por campo de protocolo (e.g. "dport:443")
    # "byte"   → un token por byte del payload
    # "field+byte" → campos + bytes concatenados
    granularity: str = "field"

    # Campos de protocolo a incluir en la tokenización
    # None = todos los disponibles
    include_fields: Optional[List[str]] = field(default_factory=lambda: [
        "ip_proto", "sport", "dport", "ip_len",
        "tcp_flags", "tcp_win", "direction",
    ])

    # Discretización de campos continuos
    # Número de bins para IAT y tamaño de paquete
    n_bins_iat:  int = 32
    n_bins_size: int = 32

    # Incluir token de dirección por paquete
    include_direction: bool = True

    # Incluir bytes del payload
    include_payload: bool = False
    max_payload_tokens: int = 10


# ---------------------------------------------------------------------------
# 1. FlatTokenizer
# ---------------------------------------------------------------------------

class FlatTokenizer(TrafficRepresentation):
    """
    Tokenización plana: cada campo de protocolo → un token string.

    Cada paquete se convierte en una lista de tokens ("campo:valor") y
    los paquetes de un flujo se concatenan con SEP entre ellos:

      BOS field1:val1 field2:val2 ... SEP field1:val1 ... EOS [PAD PAD ...]

    La representación NO es estrictamente invertible porque los valores
    continuos se discretizan en bins.

    Parameters
    ----------
    config : SequentialConfig
    """

    def __init__(self, config: Optional[SequentialConfig] = None) -> None:
        if config is None:
            config = SequentialConfig()
        super().__init__(config)
        self.cfg = config  # alias tipado
        self.vocab = TokenVocabulary(max_vocab_size=config.max_vocab_size)

        # Bins para discretización (se calculan en fit)
        self._iat_bins:  Optional[np.ndarray] = None
        self._size_bins: Optional[np.ndarray] = None

    # --- Propiedades abstractas ---

    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.SEQUENTIAL

    @property
    def invertibility(self) -> Invertibility:
        return Invertibility.APPROXIMATE

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self.cfg.max_length,)

    # --- fit ---

    def fit(self, samples: List[Flow]) -> "FlatTokenizer":
        """
        1. Calcula bins de discretización sobre todos los paquetes de train.
        2. Construye el vocabulario.
        """
        all_iats:  List[float] = []
        all_sizes: List[float] = []
        all_token_seqs: List[List[str]] = []

        for flow in samples:
            for pkt in flow.packets:
                all_iats.append(pkt.iat)
                all_sizes.append(float(pkt.ip_len))

        # Bins percentile (robustez a outliers)
        iats_arr  = np.array(all_iats,  dtype=np.float32)
        sizes_arr = np.array(all_sizes, dtype=np.float32)

        self._iat_bins  = np.percentile(
            iats_arr[iats_arr > 0],
            np.linspace(0, 100, self.cfg.n_bins_iat + 1),
        ) if (iats_arr > 0).sum() > 0 else np.linspace(0, 1, self.cfg.n_bins_iat + 1)

        self._size_bins = np.percentile(
            sizes_arr,
            np.linspace(0, 100, self.cfg.n_bins_size + 1),
        )

        # Construir corpus de tokens
        for flow in samples:
            seq = self._flow_to_tokens(flow)
            all_token_seqs.append(seq)

        self.vocab.build_from_corpus(all_token_seqs)
        self._is_fitted = True
        return self

    # --- encode / decode ---

    def encode(self, sample: Flow) -> Tensor:
        """Flow → Tensor(max_length,) de IDs enteros."""
        self._check_fitted()
        tokens = self._flow_to_tokens(sample)
        ids    = self._pad_or_truncate(
            [self.vocab.bos_id]
            + self.vocab.encode_sequence(tokens)
            + [self.vocab.eos_id]
        )
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, tensor: Tensor) -> List[str]:
        """
        Tensor → lista de tokens string.
        La reconstrucción a paquetes reales no es exacta (discretización).
        """
        self._check_fitted()
        ids = tensor.tolist()
        return self.vocab.decode_sequence(ids)

    # --- Helpers ---

    def _flow_to_tokens(self, flow: Flow) -> List[str]:
        tokens: List[str] = []
        for pkt in flow.packets:
            tokens.extend(self._pkt_to_tokens(pkt))
            tokens.append(SEP_TOKEN)
        # Quitar último SEP
        if tokens and tokens[-1] == SEP_TOKEN:
            tokens.pop()
        return tokens

    def _pkt_to_tokens(self, pkt: ParsedPacket) -> List[str]:
        tokens: List[str] = []
        fields = self.cfg.include_fields or []

        # Dirección
        if self.cfg.include_direction:
            tokens.append(DIR_FWD if pkt.direction == 0 else DIR_BWD)

        for f in fields:
            val = getattr(pkt, f, None)
            if val is None:
                continue
            # Discretizar campos continuos
            if f == "iat":
                val = self._discretize(float(val), self._iat_bins, "iat")
            elif f == "ip_len":
                val = self._discretize(float(val), self._size_bins, "size")
            else:
                val = str(val)
            tokens.append(f"{f}:{val}")

        # Payload como bytes
        if self.cfg.include_payload and pkt.payload_bytes:
            for b in pkt.payload_bytes[: self.cfg.max_payload_tokens]:
                tokens.append(f"byte:{b:02x}")

        return tokens

    def _discretize(
        self, value: float, bins: np.ndarray, prefix: str
    ) -> str:
        bin_idx = int(np.digitize(value, bins)) - 1
        bin_idx = max(0, min(bin_idx, len(bins) - 2))
        return f"{prefix}_bin{bin_idx}"

    def _pad_or_truncate(self, ids: List[int]) -> List[int]:
        L = self.cfg.max_length
        if len(ids) >= L:
            return ids[:L - 1] + [self.vocab.eos_id]
        return ids + [self.vocab.pad_id] * (L - len(ids))

    # --- Persistencia ---

    def _get_state_dict(self) -> Dict:
        return {
            "vocab_token2id": self.vocab._token2id,
            "vocab_id2token": self.vocab._id2token,
            "iat_bins":       self._iat_bins,
            "size_bins":      self._size_bins,
        }

    def _set_state_dict(self, state: Dict) -> None:
        self.vocab._token2id = state["vocab_token2id"]
        self.vocab._id2token = state["vocab_id2token"]
        self.vocab._built    = True
        self._iat_bins       = state["iat_bins"]
        self._size_bins      = state["size_bins"]


# ---------------------------------------------------------------------------
# 2. ProtocolAwareTokenizer  (inspirado en NetGPT)
# ---------------------------------------------------------------------------

@dataclass
class ProtocolAwareConfig(SequentialConfig):
    """
    Extiende SequentialConfig con parámetros para tokenización jerárquica.
    """
    name: str = "protocol_aware"

    # Codificar estado TCP como token explícito
    encode_tcp_state: bool = True

    # Separadores de capa
    layer3_sep: str = "<L3>"
    layer4_sep: str = "<L4>"
    payload_sep: str = "<PAY>"

    # Vocabulario de flags TCP como tokens atómicos
    tcp_flag_tokens: bool = True


class ProtocolAwareTokenizer(FlatTokenizer):
    """
    Tokenización estructural consciente del protocolo.

    Mejoras sobre FlatTokenizer:
    - Separa campos por capa (L3 / L4 / Payload) con tokens de separación.
    - Codifica el estado TCP como token atómico (SYN, SYN-ACK, DATA, FIN…).
    - Permite al modelo aprender coherencia protocolaria entre capas.

    Inspirado en: NetGPT (Meng et al., 2023).

    La estructura de cada paquete es:
      DIR L3_SEP ip_proto:X ip_len:Y ... L4_SEP tcp_flags:SYN dport:443 ...
                                          PAY_SEP byte:xx ...
    """

    # Mapa de flags TCP → nombre de estado
    TCP_STATES = {
        0x002: "SYN",
        0x012: "SYN-ACK",
        0x010: "ACK",
        0x018: "PSH-ACK",
        0x001: "FIN",
        0x011: "FIN-ACK",
        0x004: "RST",
        0x014: "RST-ACK",
    }

    def __init__(self, config: Optional[ProtocolAwareConfig] = None) -> None:
        if config is None:
            config = ProtocolAwareConfig()
        super().__init__(config)
        self.pcfg = config  # alias tipado

        # Añadir tokens especiales de capa al vocabulario base
        for tok in [config.layer3_sep, config.layer4_sep, config.payload_sep]:
            self.vocab._add(tok)

        # Añadir tokens de estado TCP
        if config.tcp_flag_tokens:
            for state in self.TCP_STATES.values():
                self.vocab._add(f"tcp_state:{state}")

    def _pkt_to_tokens(self, pkt: ParsedPacket) -> List[str]:
        """
        Genera tokens respetando la jerarquía de capas del protocolo.
        """
        tokens: List[str] = []

        # --- Dirección ---
        if self.cfg.include_direction:
            tokens.append(DIR_FWD if pkt.direction == 0 else DIR_BWD)

        # --- Capa de Red (L3) ---
        tokens.append(self.pcfg.layer3_sep)
        tokens.append(f"ip_version:{pkt.ip_version}")
        tokens.append(f"ip_proto:{pkt.ip_proto}")
        tokens.append(self._discretize(float(pkt.ip_len), self._size_bins, "size"))
        if pkt.ip_ttl >= 0:
            # TTL agrupado en rangos estándar
            tokens.append(f"ip_ttl:{self._ttl_bucket(pkt.ip_ttl)}")

        # --- Capa de Transporte (L4) ---
        tokens.append(self.pcfg.layer4_sep)
        tokens.append(f"sport:{pkt.sport}")
        tokens.append(f"dport:{pkt.dport}")

        if pkt.ip_proto == 6:  # TCP
            if self.pcfg.tcp_flag_tokens:
                state = self._tcp_state_token(pkt.tcp_flags)
                tokens.append(state)
            else:
                tokens.append(f"tcp_flags:{pkt.tcp_flags}")
            if pkt.tcp_win > 0:
                tokens.append(f"tcp_win:{self._win_bucket(pkt.tcp_win)}")

        elif pkt.ip_proto == 17:  # UDP
            tokens.append(f"udp_len:{pkt.udp_len}")

        # --- IAT ---
        tokens.append(self._discretize(pkt.iat, self._iat_bins, "iat"))

        # --- Payload ---
        if self.pcfg.include_payload and pkt.payload_bytes:
            tokens.append(self.pcfg.payload_sep)
            for b in pkt.payload_bytes[: self.pcfg.max_payload_tokens]:
                tokens.append(f"byte:{b:02x}")

        return tokens

    # --- Helpers TCP / TTL ---

    def _tcp_state_token(self, flags: int) -> str:
        state = self.TCP_STATES.get(flags & 0x3F, "OTHER")
        return f"tcp_state:{state}"

    @staticmethod
    def _ttl_bucket(ttl: int) -> str:
        """Agrupa TTL en rangos típicos de SO."""
        if ttl <= 32:   return "low"
        if ttl <= 64:   return "mid64"
        if ttl <= 128:  return "mid128"
        return "high"

    @staticmethod
    def _win_bucket(win: int) -> str:
        """Agrupa TCP window size en rangos."""
        if win == 0:        return "zero"
        if win < 1024:      return "small"
        if win < 16384:     return "medium"
        if win < 65535:     return "large"
        return "max"

# ---------------------------------------------------------------------------
# Helpers de bajo nivel (byte-level, estilo NetGPT)
# ---------------------------------------------------------------------------

def _bytes_to_hex_tokens(
    data: bytes,
    max_tokens: int,
    include_position: bool = False,
) -> List[str]:
    """
    Convierte bytes crudos en tokens byte-level.
    Si include_position=True, añade un token pos:i antes de cada byte.
    """
    tokens: List[str] = []
    for i, b in enumerate(data[:max_tokens]):
        if include_position:
            tokens.append(f"pos:{i}")
        tokens.append(f"byte:{b:02x}")
    return tokens


def _bytes_to_bigram_tokens(
    data: bytes,
    max_tokens: int,
    include_position: bool = False,
) -> List[str]:
    """
    Bigramas deslizantes (overlapping), no disjuntos.
    Ejemplo: (0,1), (1,2), (2,3)...
    """
    tokens: List[str] = []
    limit = min(len(data) - 1, max_tokens)
    for i in range(limit):
        if include_position:
            tokens.append(f"pos:{i}")
        tokens.append(f"bg:{data[i]:02x}{data[i + 1]:02x}")
    return tokens


def _anonymize_raw_bytes(raw: bytes) -> bytes:
    """
    Anonimiza campos sensibles sobre el frame Ethernet crudo,
    siguiendo la estrategia de NetGPT (zeroing de MAC, IP, puerto).

    Offsets asumidos: Ethernet (14 B) + IPv4 cabecera fija (20 B).
    El llamador debe garantizar len(raw) >= 38 antes de invocar.

      Byte  0-5  : Ethernet dst MAC  → 0x00
      Byte  6-11 : Ethernet src MAC  → 0x00
      Byte 26-29 : IPv4 src addr     → 0.0.0.0
      Byte 30-33 : IPv4 dst addr     → 0.0.0.0
      Byte 34-35 : TCP/UDP sport     → 0x0000
      Byte 36-37 : TCP/UDP dport     → 0x0000
    """
    buf = bytearray(raw)
    buf[0:6]   = b'\x00' * 6   # dst MAC
    buf[6:12]  = b'\x00' * 6   # src MAC
    buf[26:30] = b'\x00' * 4   # IPv4 src
    buf[30:34] = b'\x00' * 4   # IPv4 dst
    buf[34:36] = b'\x00' * 2   # sport
    buf[36:38] = b'\x00' * 2   # dport
    return bytes(buf)


# ---------------------------------------------------------------------------
# 3. SemanticByteTokenizer  — semántico + byte-level
# ---------------------------------------------------------------------------

@dataclass
class SemanticByteConfig(ProtocolAwareConfig):
    """
    Configuración para el tokenizador híbrido semántico + byte-level.
    """
    name: str = "semantic_byte"

    # Sección semántica
    semantic_section_token: str = "<SEM>"

    # --- Bytes de cabecera (L3 + L4 crudos) ---
    include_header_bytes: bool = True
    max_header_tokens: int = 64
    anonymize_header_bytes: bool = True
    header_byte_sep: str = "<HDR_BYTES>"

    # --- Bytes de payload crudo ---
    max_payload_tokens: int = 64

    # --- Anti-duplicación de payload ---
    disable_semantic_payload_when_raw: bool = True

    # --- Modo de codificación byte ---
    use_byte_bigrams: bool = False

    # --- Posición intra-paquete ---
    include_byte_position: bool = True
    max_byte_position: int = 256

    # --- Delimitadores de sección byte-level ---
    byte_section_sep: str = "<BYTES>"
    byte_section_end: str = "</BYTES>"

    # --- Señal anti-sesgo por longitud de payload ---
    include_payload_length_token: bool = True
    n_bins_payload_len: int = 16


class SemanticByteTokenizer(ProtocolAwareTokenizer):
    """
    Tokenizador híbrido: tokenización semántica protocolaria + byte-level.

    Funcionalidades:
    - separa explícitamente los espacios semántico y byte-level;
    - evita mutaciones temporales de configuración;
    - usa bigramas deslizantes si se activan;
    - añade token de posición intra-paquete;
    - añade señal de longitud de payload para reducir sesgo por truncado.
    
    Estructura:
    <SEM> ...tokens protocolarios...
    <BYTES> <HDR_BYTES> ...bytes cabecera... <PAY> ...bytes payload... </BYTES>
    """

    def __init__(self, config: Optional[SemanticByteConfig] = None) -> None:
        if config is None:
            config = SemanticByteConfig()
        super().__init__(config)
        self.hcfg = config

        self._payload_len_bins: Optional[np.ndarray] = None

        # Registrar tokens específicos del híbrido.
        for tok in [
            config.semantic_section_token,
            config.header_byte_sep,
            config.byte_section_sep,
            config.byte_section_end,
        ]:
            self.vocab._add(tok)

        # Tokens de posición intra-paquete.
        if config.include_byte_position:
            for i in range(config.max_byte_position):
                self.vocab._add(f"pos:{i}")

        # Pre-registrar bytes si se usa modo byte-level clásico.
        # En bigramas no conviene pre-registrar todo el espacio porque explota el vocabulario.
        if not config.use_byte_bigrams:
            for b in range(256):
                self.vocab._add(f"byte:{b:02x}")

        # Mantener coherente el límite tras añadir tokens reservados.
        self.vocab.max_vocab_size = max(self.vocab.max_vocab_size, len(self.vocab))

        if config.use_byte_bigrams:
            LOGGER.warning(
                "SemanticByteTokenizer: use_byte_bigrams=True activa un espacio de vocabulario mucho mayor."
            )

    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.SEQUENTIAL

    @property
    def invertibility(self) -> Invertibility:
        return Invertibility.APPROXIMATE

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self.cfg.max_length,)

    def fit(self, samples: List[Flow]) -> "SemanticByteTokenizer":
        """
        Ajusta bins de discretización y construye vocabulario.
        """
        all_iats: List[float] = []
        all_sizes: List[float] = []
        all_payload_lens: List[float] = []
        all_token_seqs: List[List[str]] = []

        for flow in samples:
            for pkt in flow.packets:
                if getattr(pkt, "iat", None) is not None:
                    all_iats.append(float(pkt.iat))

                if getattr(pkt, "ip_len", None) is not None:
                    all_sizes.append(float(pkt.ip_len))

                payload = self._get_payload_bytes(pkt)
                all_payload_lens.append(float(len(payload) if payload is not None else 0))

        iats_arr = np.array(all_iats, dtype=np.float32) if all_iats else np.array([0.0], dtype=np.float32)
        sizes_arr = np.array(all_sizes, dtype=np.float32) if all_sizes else np.array([0.0], dtype=np.float32)
        payload_arr = np.array(all_payload_lens, dtype=np.float32) if all_payload_lens else np.array([0.0], dtype=np.float32)

        self._iat_bins = (
            np.percentile(
                iats_arr[iats_arr > 0],
                np.linspace(0, 100, self.cfg.n_bins_iat + 1),
            )
            if (iats_arr > 0).sum() > 0
            else np.linspace(0, 1, self.cfg.n_bins_iat + 1)
        )

        self._size_bins = (
            np.percentile(
                sizes_arr,
                np.linspace(0, 100, self.cfg.n_bins_size + 1),
            )
            if len(sizes_arr) > 0
            else np.linspace(0, 1, self.cfg.n_bins_size + 1)
        )

        self._payload_len_bins = (
            np.percentile(
                payload_arr,
                np.linspace(0, 100, self.hcfg.n_bins_payload_len + 1),
            )
            if len(payload_arr) > 0
            else np.linspace(0, 1, self.hcfg.n_bins_payload_len + 1)
        )

        for flow in samples:
            all_token_seqs.append(self._flow_to_tokens(flow))

        self.vocab.build_from_corpus(all_token_seqs)
        self._is_fitted = True
        return self

    def encode(self, sample: Flow) -> Tensor:
        self._check_fitted()
        tokens = self._flow_to_tokens(sample)
        ids = self._pad_or_truncate(
            [self.vocab.bos_id]
            + self.vocab.encode_sequence(tokens)
            + [self.vocab.eos_id]
        )
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, tensor: Tensor) -> List[str]:
        self._check_fitted()
        return self.vocab.decode_sequence(tensor.tolist())

    def _pkt_to_tokens(self, pkt: ParsedPacket) -> List[str]:
        """
        Combina:
          <SEM> ...tokens semánticos...
          <BYTES> ...cabecera/payload crudos...
        """
        semantic_tokens = self._pkt_to_semantic_tokens(pkt)
        byte_tokens = self._pkt_to_byte_tokens(pkt)
        return semantic_tokens + byte_tokens

    def _pkt_to_semantic_tokens(self, pkt: ParsedPacket) -> List[str]:
        """
        Tokenización semántica sin sección byte-level.
        """
        tokens: List[str] = [self.hcfg.semantic_section_token]

        if self.cfg.include_direction:
            tokens.append(DIR_FWD if getattr(pkt, "direction", 0) == 0 else DIR_BWD)

        tokens.append(self.pcfg.layer3_sep)
        tokens.append(f"ip_version:{getattr(pkt, 'ip_version', 'unk')}")
        tokens.append(f"ip_proto:{getattr(pkt, 'ip_proto', 'unk')}")

        ip_len = getattr(pkt, "ip_len", None)
        if ip_len is not None:
            tokens.append(self._discretize(float(ip_len), self._size_bins, "size"))

        ttl = getattr(pkt, "ip_ttl", None)
        if ttl is not None and ttl >= 0:
            tokens.append(f"ip_ttl:{self._ttl_bucket(int(ttl))}")

        if self.hcfg.include_payload_length_token:
            payload_len = self._get_payload_length(pkt)
            tokens.append(self._payload_len_token(payload_len))

        tokens.append(self.pcfg.layer4_sep)
        sport = getattr(pkt, "sport", None)
        dport = getattr(pkt, "dport", None)
        if sport is not None:
            tokens.append(f"sport:{sport}")
        if dport is not None:
            tokens.append(f"dport:{dport}")

        ip_proto = getattr(pkt, "ip_proto", None)
        if ip_proto == 6:
            if self.pcfg.tcp_flag_tokens:
                tokens.append(self._tcp_state_token(getattr(pkt, "tcp_flags", 0) or 0))
            else:
                tokens.append(f"tcp_flags:{getattr(pkt, 'tcp_flags', 0)}")

            tcp_win = getattr(pkt, "tcp_win", None)
            if tcp_win is not None and tcp_win > 0:
                tokens.append(f"tcp_win:{self._win_bucket(int(tcp_win))}")

        elif ip_proto == 17:
            udp_len = getattr(pkt, "udp_len", None)
            if udp_len is not None:
                tokens.append(f"udp_len:{udp_len}")

        # Payload semántico opcional.
        payload = self._get_payload_bytes(pkt)
        if self.hcfg.include_payload and payload:
            if self.hcfg.disable_semantic_payload_when_raw:
                # En híbrido suele ser mejor no duplicar el payload si ya hay byte-level.
                pass
            else:
                tokens.append(self.pcfg.payload_sep)
                for b in payload[: self.hcfg.max_payload_tokens]:
                    tokens.append(f"byte:{b:02x}")

        return tokens

    def _pkt_to_byte_tokens(self, pkt: ParsedPacket) -> List[str]:
        """
        Sección byte-level completa:
          <BYTES>
            <HDR_BYTES> ...
            <PAY> ...
          </BYTES>
        """
        inner: List[str] = []

        if self.hcfg.include_header_bytes:
            header_raw = self._get_header_bytes(pkt)
            if header_raw:
                inner.append(self.hcfg.header_byte_sep)
                inner.extend(
                    self._encode_raw_bytes(
                        header_raw,
                        self.hcfg.max_header_tokens,
                    )
                )

        payload_raw = self._get_payload_bytes(pkt)
        if self.hcfg.include_payload and payload_raw:
            inner.append(self.hcfg.payload_sep)
            inner.extend(
                self._encode_raw_bytes(
                    payload_raw,
                    self.hcfg.max_payload_tokens,
                )
            )

        if not inner:
            return []

        return [self.hcfg.byte_section_sep] + inner + [self.hcfg.byte_section_end]

    def _encode_raw_bytes(self, data: bytes, max_tokens: int) -> List[str]:
        if self.hcfg.use_byte_bigrams:
            return _bytes_to_bigram_tokens(
                data,
                max_tokens=max_tokens,
                include_position=self.hcfg.include_byte_position,
            )
        return _bytes_to_hex_tokens(
            data,
            max_tokens=max_tokens,
            include_position=self.hcfg.include_byte_position,
        )

    def _get_payload_bytes(self, pkt: ParsedPacket) -> Optional[bytes]:
        """
        Devuelve payload_bytes si existe. Si no, intenta recortarlo desde raw_bytes
        usando payload_offset si el objeto lo expone.
        """
        payload = getattr(pkt, "payload_bytes", None)
        if payload is not None:
            return payload

        raw = getattr(pkt, "raw_bytes", None)
        payload_offset = getattr(pkt, "payload_offset", None)
        if raw is not None and payload_offset is not None and 0 <= int(payload_offset) < len(raw):
            return raw[int(payload_offset):]

        return None

    def _get_payload_length(self, pkt: ParsedPacket) -> int:
        payload = self._get_payload_bytes(pkt)
        if payload is not None:
            return len(payload)
        payload_len = getattr(pkt, "payload_len", None)
        if payload_len is not None:
            return int(payload_len)
        return 0

    def _payload_len_token(self, payload_len: int) -> str:
        if self._payload_len_bins is None:
            return f"payload_len:{payload_len}"
        return self._discretize(float(payload_len), self._payload_len_bins, "payload_len")

    def _get_header_bytes(self, pkt: ParsedPacket) -> Optional[bytes]:
        """
        Extrae bytes de cabecera de forma segura.

        Estrategia:
          1) usar header_bytes si ya viene pre-extraído;
          2) si existe raw_bytes y payload_offset, recortar por offsets explícitos;
          3) si no hay offsets, intentar heurística IPv4/Ethernet solo cuando sea seguro;
          4) en caso contrario, devolver None antes que fabricar offsets erróneos.
        """
        header = getattr(pkt, "header_bytes", None)
        if header is not None:
            return header

        raw: Optional[bytes] = getattr(pkt, "raw_bytes", None)
        if raw is None:
            return None

        # Offset explícito preferido.
        payload_offset = getattr(pkt, "payload_offset", None)
        if payload_offset is not None:
            end = int(payload_offset)
            if 0 < end <= len(raw):
                header = raw[:end]
                return _anonymize_raw_bytes(header) if self.hcfg.anonymize_header_bytes else header

        # Heurística segura solo para Ethernet + IPv4.
        ip_version = getattr(pkt, "ip_version", None)
        ip_proto = getattr(pkt, "ip_proto", None)
        if ip_version != 4 or len(raw) < 34:
            return None

        eth_len = 14
        ihl = (raw[eth_len] & 0x0F) * 4
        if ihl < 20 or len(raw) < eth_len + ihl:
            return None

        l4_start = eth_len + ihl
        if ip_proto == 6:
            if len(raw) < l4_start + 20:
                return None
            tcp_data_offset = ((raw[l4_start + 12] >> 4) & 0x0F) * 4
            if tcp_data_offset < 20:
                tcp_data_offset = 20
            end = l4_start + tcp_data_offset
        elif ip_proto == 17:
            end = l4_start + 8
        else:
            end = l4_start

        if end > len(raw):
            return None

        header = raw[:end]
        if self.hcfg.anonymize_header_bytes:
            # Solo aplica correctamente a Ethernet/IPv4 con offsets compatibles.
            return _anonymize_raw_bytes(header)
        return header

    def _tcp_state_token(self, flags: int) -> str:
        state = self.TCP_STATES.get(flags & 0x3F, "OTHER")
        return f"tcp_state:{state}"

    @staticmethod
    def _ttl_bucket(ttl: int) -> str:
        if ttl <= 32:
            return "low"
        if ttl <= 64:
            return "mid64"
        if ttl <= 128:
            return "mid128"
        return "high"

    @staticmethod
    def _win_bucket(win: int) -> str:
        if win == 0:
            return "zero"
        if win < 1024:
            return "small"
        if win < 16384:
            return "medium"
        if win < 65535:
            return "large"
        return "max"

    def _get_state_dict(self) -> Dict:
        state = super()._get_state_dict()
        state["semantic_byte_config"] = {
            "semantic_section_token": self.hcfg.semantic_section_token,
            "include_header_bytes": self.hcfg.include_header_bytes,
            "max_header_tokens": self.hcfg.max_header_tokens,
            "anonymize_header_bytes": self.hcfg.anonymize_header_bytes,
            "max_payload_tokens": self.hcfg.max_payload_tokens,
            "disable_semantic_payload_when_raw": self.hcfg.disable_semantic_payload_when_raw,
            "use_byte_bigrams": self.hcfg.use_byte_bigrams,
            "include_byte_position": self.hcfg.include_byte_position,
            "max_byte_position": self.hcfg.max_byte_position,
            "include_payload_length_token": self.hcfg.include_payload_length_token,
            "n_bins_payload_len": self.hcfg.n_bins_payload_len,
        }
        state["payload_len_bins"] = self._payload_len_bins
        return state

    def _set_state_dict(self, state: Dict) -> None:
        super()._set_state_dict(state)

        for k, v in state.get("semantic_byte_config", {}).items():
            if hasattr(self.hcfg, k):
                setattr(self.hcfg, k, v)

        self._payload_len_bins = state.get("payload_len_bins", None)
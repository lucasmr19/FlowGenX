"""
representations/vision/nprint.py
========================
Representaciones visuales del tráfico de red.

Representación estructurada e invertible que codifica cada paquete como una fila de bits, con campos serializados
según un esquema configurable. Inspirada en NetDiffusion (Jiang et al., 2024) y fiel al esquema
de la librería nPrint 1.2.1 (https://github.com/nprint/nprint).
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from ...data_utils.preprocessing import PacketWindow, ParsedPacket
from ..base import (
    Invertibility,
    RepresentationConfig,
    RepresentationType,
    TrafficRepresentation,
)
from ...utils.logger_config import LOGGER


# ---------------------------------------------------------------------------
# Campos que se almacenan como secuencias de bytes (no como un único entero).
# Clave: nombre del campo → número de bytes esperados.
# ---------------------------------------------------------------------------
_BYTES_FIELDS: Dict[str, int] = {
    "eth_dhost":  6,   # MAC destino      (6 B)
    "eth_shost":  6,   # MAC origen       (6 B)
    "ipv4_src":   4,   # IPv4 src addr    (4 B)
    "ipv4_dst":   4,   # IPv4 dst addr    (4 B)
    "ipv4_opt":  40,   # IPv4 options     (40 B  = 40 * 8 = 320 bits)
    "ipv6_src":  16,   # IPv6 src addr    (16 B  = 16 * 8 = 128 bits)
    "ipv6_dst":  16,   # IPv6 dst addr    (16 B  = 16 * 8 = 128 bits)
    "tcp_opt":   40,   # TCP options      (40 B  = 40 * 8 = 320 bits)
    "payload":   10,   # Primeros 10 B    (80 bits)
}


@dataclass
class NprintConfig(RepresentationConfig):
    """Parámetros para la representación nPrint completa."""
    representation_type: str = "nprint"
    name: str = "nprint"

    # Número máximo de paquetes por flujo (filas de la imagen)
    max_packets: int = 1024  # valor por defecto alto para no truncar flujos cortos

    # -----------------------------------------------------------------------
    # Esquema completo de campos siguiendo la especificación de nPrint 1.2.1.
    # ver https://github.com/nprint/nprint/blob/master/src/nprint.cpp
    # Cada entrada: (nombre_campo, bits).
    # El ancho total determina la anchura de la imagen resultante.
    #
    # Protocolos incluidos: Ethernet · IPv4 · IPv6 · TCP · UDP · ICMP
    # Metafields:           direction · iat_q · payload
    # -----------------------------------------------------------------------
    field_schema: List[Tuple[str, int]] = field(default_factory=lambda: [
        # --- Ethernet (112 bits) -------------------------------------------
        ("eth_dhost",      48),   # MAC destino
        ("eth_shost",      48),   # MAC origen
        ("eth_ethertype",  16),   # EtherType

        # --- IPv4 (480 bits) ------------------------------------------------
        ("ipv4_ver",        4),   # Versión
        ("ipv4_hl",         4),   # Header length (IHL)
        ("ipv4_tos",        8),   # Type of service / DSCP+ECN
        ("ipv4_tl",        16),   # Total length
        ("ipv4_id",        16),   # Identification
        ("ipv4_rbit",       1),   # Reserved bit (siempre 0)
        ("ipv4_dfbit",      1),   # Don't Fragment
        ("ipv4_mfbit",      1),   # More Fragments
        ("ipv4_foff",      13),   # Fragment offset
        ("ipv4_ttl",        8),   # Time To Live
        ("ipv4_proto",      8),   # Protocolo encapsulado
        ("ipv4_cksum",     16),   # Checksum cabecera
        ("ipv4_src",       32),   # IP origen
        ("ipv4_dst",       32),   # IP destino
        ("ipv4_opt",      320),   # Opciones (hasta 40 B)

        # --- IPv6 (320 bits) ------------------------------------------------
        ("ipv6_ver",        4),   # Versión (siempre 6)
        ("ipv6_tc",         8),   # Traffic class
        ("ipv6_fl",        20),   # PacketWindow label
        ("ipv6_len",       16),   # Payload length
        ("ipv6_nh",         8),   # Next header
        ("ipv6_hl",         8),   # Hop limit
        ("ipv6_src",      128),   # IP origen
        ("ipv6_dst",      128),   # IP destino

        # --- TCP (480 bits) -------------------------------------------------
        ("tcp_sprt",       16),   # Puerto origen
        ("tcp_dprt",       16),   # Puerto destino
        ("tcp_seq",        32),   # Número de secuencia
        ("tcp_ackn",       32),   # Número de ACK
        ("tcp_doff",        4),   # Data offset
        ("tcp_res",         3),   # Reservado
        ("tcp_ns",          1),   # Nonce Sum (ECN)
        ("tcp_cwr",         1),   # Congestion Window Reduced
        ("tcp_ece",         1),   # ECN-Echo
        ("tcp_urg",         1),   # Urgent pointer valid
        ("tcp_ackf",        1),   # Acknowledgment field valid
        ("tcp_psh",         1),   # Push function
        ("tcp_rst",         1),   # Reset connection
        ("tcp_syn",         1),   # Synchronize seq numbers
        ("tcp_fin",         1),   # Fin — cierre de conexión (no en el help de
                                  # nPrint 1.2.1 pero sí en el código fuente)
        ("tcp_wsize",      16),   # Window size
        ("tcp_cksum",      16),   # Checksum
        ("tcp_urp",        16),   # Urgent pointer
        ("tcp_opt",       320),   # Opciones TCP (hasta 40 B)

        # --- UDP (64 bits) --------------------------------------------------
        ("udp_sport",      16),   # Puerto origen
        ("udp_dport",      16),   # Puerto destino
        ("udp_len",        16),   # Longitud total UDP
        ("udp_cksum",      16),   # Checksum

        # --- ICMP (64 bits) -------------------------------------------------
        ("icmp_type",       8),   # Tipo de mensaje
        ("icmp_code",       8),   # Código
        ("icmp_cksum",     16),   # Checksum
        ("icmp_roh",       32),   # Rest of Header (datos extra según tipo)

        # --- Meta / extra fields --------------------------------------------
        ("direction",       1),   # 0 = cliente→servidor, 1 = servidor→cliente
        ("iat_q",           8),   # IAT cuantizado a 8 bits (0-255)
        ("payload",        80),   # Primeros 10 bytes del payload (10 × 8)
    ])

    # Valor para rellenar posiciones sin datos (protocolo ausente)
    pad_value: float = -1.0


class NprintRepresentation(TrafficRepresentation):
    """
    Representación nPrint completa: imagen binaria/continua de dimensión
    (max_packets x total_bits), donde:

      - height = max_packets   → un paquete por fila
      - width  = Σ bits        → todos los campos de protocolo serializados

    Valores de cada celda:
        1.0  → bit activo
        0.0  → bit inactivo
       -1.0  → campo/protocolo inexistente en este paquete

    Protocolos soportados: Ethernet, IPv4, IPv6, TCP, UDP, ICMP.
    Metafields: dirección de flujo, IAT cuantizado, payload.

    La representación es INVERTIBLE: dada la imagen se pueden reconstruir
    los campos de cabecera con precisión exacta (salvo la cuantización del
    IAT a 8 bits).

    Referencia: nPrint 1.2.1 — https://github.com/nprint/nprint
    """

    # Campos que se manejan como arrays de bytes en lugar de como un único int
    _BYTES_FIELDS = _BYTES_FIELDS
    
    _FIELD_MAP = {
        # IPv4
        "ipv4_ver":   "ipv4_ver",
        "ipv4_hl":    "ipv4_hl",
        "ipv4_tos":   "ipv4_tos",
        "ipv4_tl":    "ipv4_tl",   # existe en ParsedPacket
        "ipv4_id":    "ipv4_id",
        "ipv4_rbit":  "ipv4_rbit",
        "ipv4_dfbit": "ipv4_dfbit",
        "ipv4_mfbit": "ipv4_mfbit",
        "ipv4_foff":  "ipv4_foff",
        "ipv4_ttl":   "ipv4_ttl",
        "ipv4_proto": "ipv4_proto",
        "ipv4_cksum": "ipv4_cksum",
        "ipv4_src":   "ipv4_src",
        "ipv4_dst":   "ipv4_dst",
        "ipv4_opt":   "ipv4_opt",

        # IPv6
        "ipv6_ver":   "ipv6_ver",
        "ipv6_tc":    "ipv6_tc",
        "ipv6_fl":    "ipv6_fl",
        "ipv6_len":   "ipv6_len",
        "ipv6_nh":    "ipv6_nh",
        "ipv6_hl":    "ipv6_hl",
        "ipv6_src":   "ipv6_src",
        "ipv6_dst":   "ipv6_dst",

        # TCP
        "tcp_sprt":   "tcp_sprt",
        "tcp_dprt":   "tcp_dprt",
        "tcp_seq":    "tcp_seq",
        "tcp_ackn":   "tcp_ackn",
        "tcp_doff":   "tcp_doff",
        "tcp_res":    "tcp_res",
        "tcp_ns":     "tcp_ns",
        "tcp_cwr":    "tcp_cwr",
        "tcp_ece":    "tcp_ece",
        "tcp_urg":    "tcp_urg",
        "tcp_ackf":   "tcp_ackf",
        "tcp_psh":    "tcp_psh",
        "tcp_rst":    "tcp_rst",
        "tcp_syn":    "tcp_syn",
        "tcp_fin":    "tcp_fin",
        "tcp_wsize":  "tcp_wsize",
        "tcp_cksum":  "tcp_cksum",
        "tcp_urp":    "tcp_urp",
        "tcp_opt":    "tcp_opt",

        # UDP
        "udp_sport":  "udp_sport",
        "udp_dport":  "udp_dport",
        "udp_len":    "udp_len",
        "udp_cksum":  "udp_cksum",

        # ICMP
        "icmp_type":  "icmp_type",
        "icmp_code":  "icmp_code",
        "icmp_cksum": "icmp_cksum",
        "icmp_roh":   "icmp_roh",

        # Payload
        "payload":    "payload_bytes",
        # Meta
        "direction":  "direction",
        "iat_q":      "iat",   # special case handled in getter
    }

    def __init__(self, config: Optional[NprintConfig] = None) -> None:
        if config is None:
            config = NprintConfig()
        super().__init__(config)
        self.cfg = config

        # Ancho total de la fila en bits
        self._total_bits: int = sum(bits for _, bits in config.field_schema)

        # Máximo de IAT para cuantización; se calibra en fit()
        self._iat_max: float = 1.0

    # -----------------------------------------------------------------------
    # Propiedades abstractas
    # -----------------------------------------------------------------------

    @property
    def representation_type(self) -> RepresentationType:
        return RepresentationType.VISUAL

    @property
    def invertibility(self) -> Invertibility:
        return Invertibility.INVERTIBLE

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self.cfg.max_packets, self._total_bits)

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def fit(self, samples: List[PacketWindow]) -> "NprintRepresentation":
        """Calcula el percentil 99 del IAT para cuantización a 8 bits."""
        all_iats = [
            pkt.iat
            for PacketWindow in samples
            for pkt in PacketWindow.packets
            if pkt.iat is not None and pkt.iat > 0
        ]
        self._iat_max = float(np.percentile(all_iats, 99)) if all_iats else 1.0
        self._is_fitted = True
        LOGGER.debug(
            "nPrint fit: iat_max=%.6f  bits_per_row=%d  shape=%s",
            self._iat_max, self._total_bits, self.output_shape,
        )
        return self

    # -----------------------------------------------------------------------
    # encode
    # -----------------------------------------------------------------------

    def encode(self, sample: PacketWindow) -> Tensor:
        """
        PacketWindow → Tensor(max_packets, total_bits) de float32 en {-1.0, 0.0, 1.0}.

        Los paquetes se truncan a max_packets; las filas sobrantes se rellenan
        con pad_value.
        """
        self._check_fitted()
        H, W = self.cfg.max_packets, self._total_bits
        image = np.full((H, W), self.cfg.pad_value, dtype=np.float32)

        for row, pkt in enumerate(sample.packets[:H]):
            image[row] = self._pkt_to_row(pkt)

        return torch.tensor(image, dtype=torch.float32)

    # -----------------------------------------------------------------------
    # decode
    # -----------------------------------------------------------------------

    def decode(self, tensor: Tensor) -> List[ParsedPacket]:
        self._check_fitted()
        tensor = tensor.detach().cpu()
        return self._decode_from_bit_matrix(tensor)


    def _decode_from_bit_matrix(self, tensor: Tensor) -> List[ParsedPacket]:
        tensor = tensor.detach().cpu()

        image = tensor.detach().cpu().numpy()
        packets: List[ParsedPacket] = []

        for row in image:
            if np.all(np.isclose(row, self.cfg.pad_value)):
                continue
            packets.append(self._row_to_pkt(row))

        return packets
    
    def get_default_aggregator(self):
        from ...data_utils.preprocessing import PacketWindowAggregator
        return PacketWindowAggregator


    # -----------------------------------------------------------------------
    # _pkt_to_row  (encode de un paquete individual)
    # -----------------------------------------------------------------------

    def _pkt_to_row(self, pkt: ParsedPacket) -> np.ndarray:
        """
        ParsedPacket → np.ndarray(total_bits,) de float32.

        Itera sobre el field_schema y serializa cada campo bit a bit,
        distinguiendo entre campos de bytes y campos escalares.
        """
        bits: List[float] = []

        for field_name, n_bits in self.cfg.field_schema:

            if field_name in self._BYTES_FIELDS:
                # --- Campos multi-byte (MAC, IP, opciones, payload) ---------
                n_bytes = self._BYTES_FIELDS[field_name]
                raw = self._get_bytes_field(pkt, field_name, n_bytes)
                if raw is None:
                    bits.extend([-1.0] * n_bits)
                else:
                    for byte_val in raw:
                        bits.extend(self._int_to_bits(byte_val, 8))
            else:
                # --- Campos escalares (enteros, flags, etc.) ----------------
                val = self._get_scalar_field(pkt, field_name)
                if val < 0:
                    bits.extend([-1.0] * n_bits)
                else:
                    bits.extend(self._int_to_bits(int(val), n_bits))

        # Seguridad: truncar o rellenar hasta total_bits exacto
        row = bits[:self._total_bits]
        if len(row) < self._total_bits:
            row.extend([self.cfg.pad_value] * (self._total_bits - len(row)))
        if len(bits) != self._total_bits:
            LOGGER.warning("Mismatch between expected and actual number of bits.")

        return np.array(row, dtype=np.float32)

    # -----------------------------------------------------------------------
    # _row_to_pkt  (decode de una fila individual)
    # -----------------------------------------------------------------------

    def _row_to_pkt(self, bits: np.ndarray) -> ParsedPacket:
        """
        np.ndarray(total_bits,) → ParsedPacket.

        Itera sobre el field_schema y deserializa campo a campo.
        Los campos con todos los bits a -1 se omiten (protocolo ausente).
        """
        pkt = ParsedPacket()
        offset = 0

        for field_name, n_bits in self.cfg.field_schema:
            field_bits = bits[offset: offset + n_bits]

            if np.all(field_bits < 0):
                # Protocolo/campo no presente en este paquete
                offset += n_bits
                continue

            if field_name in self._BYTES_FIELDS:
                # --- Campos multi-byte --------------------------------------
                n_bytes = self._BYTES_FIELDS[field_name]
                raw_bytes = bytes(
                    self._bits_to_int(field_bits[i * 8: (i + 1) * 8])
                    for i in range(n_bytes)
                )
                self._set_bytes_field(pkt, field_name, raw_bytes)
            else:
                # --- Campos escalares ---------------------------------------
                val = self._bits_to_int(field_bits)
                self._set_scalar_field(pkt, field_name, val)

            offset += n_bits

        return pkt

    # -----------------------------------------------------------------------
    # Getters de campo
    # -----------------------------------------------------------------------

    def _get_bytes_field(
        self, pkt: ParsedPacket, field_name: str, n_bytes: int
    ) -> Optional[bytes]:
        """
        Obtiene un campo multi-byte como ``bytes`` de longitud exacta n_bytes.
        Soporta atributos almacenados como bytes, bytearray, int o str (IP).
        Devuelve None si el campo no está presente.
        """
        attr = self._FIELD_MAP.get(field_name, field_name)
        raw = getattr(pkt, attr, None)

        if raw is None:
            return None

        # IPv4
        if field_name in ("ipv4_src", "ipv4_dst"):
            try:
                raw = socket.inet_aton(raw)
            except OSError:
                return None

        # IPv6
        elif field_name in ("ipv6_src", "ipv6_dst"):
            try:
                raw = socket.inet_pton(socket.AF_INET6, raw)
            except OSError:
                return None

        if isinstance(raw, bytes):
            return raw[:n_bytes].ljust(n_bytes, b"\x00")

        return None

    def _get_scalar_field(self, pkt: ParsedPacket, field_name: str) -> float:
        """
        Obtiene un campo escalar como float.
        Devuelve -1.0 si el campo no está presente o su protocolo es ajeno
        al paquete actual.
        """
        # special: iat quantized
        if field_name == "iat_q":
            if getattr(pkt, "iat", None) is None:
                return -1.0
            return float(min(255, int((pkt.iat / max(self._iat_max, 1e-9)) * 255)))

        if field_name == "direction":
            v = getattr(pkt, "direction", None)
            return float(v) if v is not None else -1.0

        # map to actual attribute name
        attr = self._FIELD_MAP.get(field_name, field_name)
        val = getattr(pkt, attr, None)

        if val is None:
            return -1.0

        if isinstance(val, bool):
            return 1.0 if val else 0.0

        if isinstance(val, (int, float)):
            return float(val)

        # fallback (strings/bytes not handled here)
        return -1.0

    # -----------------------------------------------------------------------
    # Setters de campo
    # -----------------------------------------------------------------------

    def _set_bytes_field(
        self, pkt: ParsedPacket, field_name: str, raw: bytes
    ) -> None:
        """
        Escribe un campo multi-byte en ParsedPacket.
        Las IPs se convierten de vuelta a cadena para compatibilidad.
        """
        if field_name == "payload":
            pkt.payload_bytes = raw
            pkt.payload_len = len(raw.rstrip(b"\x00"))
            return

        if field_name in ("ipv4_src", "ipv4_dst"):
            try:
                setattr(pkt, field_name, socket.inet_ntoa(raw))
            except OSError:
                setattr(pkt, field_name, raw)
            return

        if field_name in ("ipv6_src", "ipv6_dst"):
            try:
                setattr(pkt, field_name, socket.inet_ntop(socket.AF_INET6, raw))
            except OSError:
                setattr(pkt, field_name, raw)
            return

        # eth_dhost, eth_shost, ipv4_opt, tcp_opt → se guardan como bytes
        try:
            setattr(pkt, field_name, raw)
        except AttributeError:
            pass

    def _set_scalar_field(self, pkt: ParsedPacket, field_name: str, val: int) -> None:
        """Escribe un campo escalar en ParsedPacket."""
        if field_name == "iat_q":
            pkt.iat = (val / 255.0) * self._iat_max
            return
        try:
            setattr(pkt, field_name, val)
        except AttributeError:
            pass

    # -----------------------------------------------------------------------
    # Utilidades bit a bit
    # -----------------------------------------------------------------------

    @staticmethod
    def _int_to_bits(value: int, n_bits: int) -> List[float]:
        """Entero → lista de n_bits floats (MSB primero) en {0.0, 1.0}."""
        value = max(0, min(value, (1 << n_bits) - 1))
        return [float((value >> (n_bits - 1 - i)) & 1) for i in range(n_bits)]

    @staticmethod
    def _bits_to_int(bits: np.ndarray) -> int:
        """Lista de bits (umbral 0.5) → entero (MSB primero)."""
        result = 0
        for b in bits:
            result = (result << 1) | (1 if b > 0.5 else 0)
        return result

    # -----------------------------------------------------------------------
    # Persistencia del estado
    # -----------------------------------------------------------------------

    def _get_state_dict(self) -> Dict:
        return {"iat_max": self._iat_max}

    def _set_state_dict(self, state: Dict) -> None:
        self._iat_max = state["iat_max"]
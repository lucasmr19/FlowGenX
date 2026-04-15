"""
src/reconstruction/serialization.py
=====================================
Serialización de SyntheticSample a ficheros .pcap via Scapy.

Este módulo es completamente OPCIONAL. Si Scapy no está instalado,
las funciones no están disponibles pero el resto del módulo funciona.

Funciones principales
---------------------
samples_to_pcap(samples, output_path)
    Escribe una lista de SyntheticSample en un archivo .pcap.

flow_to_packets(flow)
    Convierte un SyntheticSample en una lista de paquetes Scapy.

pcap_to_samples(pcap_path)
    (Utilidad inversa) Lee un .pcap y devuelve una lista de SyntheticSample
    para comparación o evaluación.

Notas de diseño
---------------
- Los checksums IP/TCP/UDP se recalculan por Scapy automáticamente
  si se omiten en la construcción (del=["chksum"]).
- Se reintenta la construcción del paquete con fallback a Raw si el
  decode produce bytes inválidos para la cabecera TCP/IP.
- Si un paquete tiene ip_proto no soportado, se escribe como Raw.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

from src.reconstruction.base import ParsedPacket, SyntheticSample

logger = logging.getLogger(__name__)

# Guard de importación opcional
try:
    from scapy.all import (
        IP, TCP, UDP, Raw, Ether,
        wrpcap, rdpcap,
        PcapWriter,
    )
    _SCAPY_AVAILABLE = True
except ImportError:
    _SCAPY_AVAILABLE = False
    logger.warning(
        "[serialization] Scapy no está instalado. "
        "La serialización a .pcap no está disponible. "
        "Instala con: pip install scapy"
    )


# ---------------------------------------------------------------------------
# Conversión de un ParsedPacket → Scapy packet
# ---------------------------------------------------------------------------

def _parsed_packet_to_scapy(pkt: ParsedPacket):
    """
    Convierte un ParsedPacket en un paquete Scapy.

    - Se omiten los checksums (Scapy los recalcula al serializar).
    - Si el protocolo no es TCP/UDP se empaqueta como IP/Raw.
    - En caso de excepción se devuelve None (el paquete se omite).
    """
    if not _SCAPY_AVAILABLE:
        raise RuntimeError("Scapy no está disponible. Instala con: pip install scapy")

    try:
        ip_layer = IP(
            src=pkt.ip_src,
            dst=pkt.ip_dst,
            proto=pkt.ip_proto,
            ttl=pkt.ip_ttl,
        )

        if pkt.ip_proto == 6:  # TCP
            transport = TCP(
                sport=pkt.sport,
                dport=pkt.dport,
                seq=pkt.tcp_seq,
                ack=pkt.tcp_ack,
                flags=pkt.tcp_flags,
                window=pkt.tcp_window,
            )
        elif pkt.ip_proto == 17:  # UDP
            transport = UDP(
                sport=pkt.sport,
                dport=pkt.dport,
            )
        else:
            # Protocolo no soportado → encapsular payload directamente
            scapy_pkt = ip_layer / Raw(load=pkt.payload)
            scapy_pkt.time = pkt.timestamp
            return scapy_pkt

        payload_layer = Raw(load=pkt.payload) if pkt.payload else b""

        if payload_layer:
            scapy_pkt = ip_layer / transport / payload_layer
        else:
            scapy_pkt = ip_layer / transport

        scapy_pkt.time = pkt.timestamp
        return scapy_pkt

    except Exception as exc:
        logger.debug("[serialization] Error construyendo paquete Scapy: %s", exc)
        return None


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def flow_to_packets(flow: SyntheticSample) -> list:
    """
    Convierte un SyntheticSample en una lista de paquetes Scapy.

    Los paquetes con errores de construcción se omiten silenciosamente
    (se registra un aviso en DEBUG).

    Parameters
    ----------
    flow : SyntheticSample

    Returns
    -------
    Lista de paquetes Scapy.
    """
    if not _SCAPY_AVAILABLE:
        raise RuntimeError("Scapy no está disponible.")

    scapy_pkts = []
    for pkt in flow.packets:
        sp = _parsed_packet_to_scapy(pkt)
        if sp is not None:
            scapy_pkts.append(sp)
        else:
            logger.debug(
                "[serialization] Paquete omitido en muestra %s:%d→%s:%d",
                flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port,
            )

    return scapy_pkts


def samples_to_pcap(
    samples: List[SyntheticSample],
    output_path: Union[str, Path],
    include_eth: bool = False,
    sort_by_timestamp: bool = True,
) -> Path:
    """
    Escribe una lista de SyntheticSample en un archivo .pcap.

    Parameters
    ----------
    samples           : muestras a serializar.
    output_path     : ruta del fichero .pcap de salida.
    include_eth     : si True, añade cabecera Ethernet falsa (para herramientas
                      que requieren capa 2).
    sort_by_timestamp: si True, ordena todos los paquetes por timestamp antes
                      de escribir (útil para pcaps multi-muestra).

    Returns
    -------
    Path del fichero escrito.

    Raises
    ------
    RuntimeError si Scapy no está disponible.
    ImportError  si no se puede escribir el fichero.
    """
    if not _SCAPY_AVAILABLE:
        raise RuntimeError(
            "Scapy no está disponible. Instala con: pip install scapy"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pkts = []
    for flow in samples:
        pkts = flow_to_packets(flow)
        if include_eth:
            pkts = [Ether() / p for p in pkts]
        all_pkts.extend(pkts)

    if not all_pkts:
        logger.warning("[serialization] No se generó ningún paquete válido para %s", output_path)
        return output_path

    if sort_by_timestamp:
        all_pkts.sort(key=lambda p: float(p.time))

    wrpcap(str(output_path), all_pkts)
    logger.info(
        "[serialization] Escritos %d paquetes de %d muestras en %s",
        len(all_pkts), len(samples), output_path,
    )
    return output_path


def pcap_to_samples(pcap_path: Union[str, Path]) -> List[SyntheticSample]:
    """
    Lee un archivo .pcap y devuelve una lista de SyntheticSample.

    Los paquetes se agrupan por 5-tupla (src_ip, dst_ip, sport, dport, proto).
    Útil para comparar tráfico generado con tráfico real en evaluación.

    Parameters
    ----------
    pcap_path : ruta al fichero .pcap.

    Returns
    -------
    Lista de SyntheticSample reconstruidos desde el pcap.
    """
    if not _SCAPY_AVAILABLE:
        raise RuntimeError("Scapy no está disponible.")

    pcap_path = Path(pcap_path)
    raw_pkts = rdpcap(str(pcap_path))

    samples_dict: dict[tuple, SyntheticSample] = {}

    for sp in raw_pkts:
        if not sp.haslayer(IP):
            continue

        ip = sp[IP]
        proto = ip.proto

        if proto == 6 and sp.haslayer(TCP):
            t = sp[TCP]
            key = (ip.src, ip.dst, t.sport, t.dport, 6)
            pkt = ParsedPacket(
                ip_src=ip.src, ip_dst=ip.dst,
                ip_proto=6, ip_len=ip.len, ip_ttl=ip.ttl,
                sport=t.sport, dport=t.dport,
                tcp_seq=t.seq, tcp_ack=t.ack,
                tcp_flags=int(t.flags),
                tcp_window=t.window,
                payload=bytes(t.payload),
                timestamp=float(sp.time),
                reconstructed=False,
            )
        elif proto == 17 and sp.haslayer(UDP):
            u = sp[UDP]
            key = (ip.src, ip.dst, u.sport, u.dport, 17)
            pkt = ParsedPacket(
                ip_src=ip.src, ip_dst=ip.dst,
                ip_proto=17, ip_len=ip.len, ip_ttl=ip.ttl,
                sport=u.sport, dport=u.dport,
                udp_len=u.len,
                payload=bytes(u.payload),
                timestamp=float(sp.time),
                reconstructed=False,
            )
        else:
            key = (ip.src, ip.dst, 0, 0, proto)
            pkt = ParsedPacket(
                ip_src=ip.src, ip_dst=ip.dst,
                ip_proto=proto, ip_len=ip.len, ip_ttl=ip.ttl,
                payload=bytes(ip.payload),
                timestamp=float(sp.time),
                reconstructed=False,
            )

        if key not in samples_dict:
            samples_dict[key] = SyntheticSample(
                src_ip=key[0], dst_ip=key[1],
                src_port=key[2], dst_port=key[3],
                protocol=key[4],
            )
        samples_dict[key].packets.append(pkt)

    # Ordenar paquetes de cada muestra por timestamp
    for flow in samples_dict.values():
        flow.packets.sort(key=lambda p: p.timestamp)

    return list(samples_dict.values())
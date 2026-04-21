from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union, Iterator

from ...utils.logger_config import LOGGER

from scapy.all import (
    IP, rdpcap, PcapReader, Packet,
)

class PCAPReader:
    """
    Lee archivos PCAP y devuelve paquetes Scapy crudos de forma iterativa.

    Parameters
    ----------
    max_packets : int, optional
        Límite de paquetes a leer. None = sin límite.
    protocols   : list of str, optional
        Filtro de protocolos: ["TCP", "UDP", "ICMP"]. None = todos.
    streaming   : bool
        Si True usa PcapReader (streaming), si False rdpcap (en memoria).
    """

    PROTOCOL_MAP = {"TCP": 6, "UDP": 17, "ICMP": 1}

    def __init__(
        self,
        max_packets: Optional[int] = None,
        protocols:   Optional[List[str]] = None,
        streaming:   bool = True,
    ) -> None:
        self.max_packets = max_packets
        self.protocols   = protocols
        self.streaming   = streaming
        self._proto_nums = (
            {self.PROTOCOL_MAP[p] for p in protocols if p in self.PROTOCOL_MAP}
            if protocols else None
        )

    def read(self, pcap_path: Union[str, Path]) -> Iterator[Packet]:
        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP no encontrado: {pcap_path}")

        LOGGER.info("Leyendo PCAP: %s (streaming=%s)", pcap_path, self.streaming)
        count = 0

        reader = PcapReader(str(pcap_path)) if self.streaming else iter(rdpcap(str(pcap_path)))

        for pkt in reader:
            if self.max_packets and count >= self.max_packets:
                break
            if self._passes_filter(pkt):
                yield pkt
                count += 1

        LOGGER.info("Leídos %d paquetes de %s", count, pcap_path.name)

    def read_all(self, pcap_path: Union[str, Path]) -> List[Packet]:
        return list(self.read(pcap_path))

    def _passes_filter(self, pkt: Packet) -> bool:
        if self._proto_nums is None:
            return True
        if IP in pkt:
            return pkt[IP].proto in self._proto_nums
        return False
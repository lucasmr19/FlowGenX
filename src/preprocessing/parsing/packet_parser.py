from __future__ import annotations

from typing import List, Optional

from scapy.all import (
    IP, IPv6, TCP, UDP, ICMP, Ether, Raw,
    Packet,
)

from ..domain.packet import ParsedPacket

class PacketParser:
    """
    Convierte paquetes Scapy crudos en ParsedPacket normalizados.
    """

    def __init__(
        self,
        max_payload_bytes: int = 20,
        include_payload:   bool = True,
    ) -> None:
        self.max_payload_bytes = max_payload_bytes
        self.include_payload   = include_payload

    def parse(
        self,
        pkt: Packet,
        prev_timestamp: Optional[float] = None,
    ) -> Optional[ParsedPacket]:
        if IP not in pkt and IPv6 not in pkt:
            return None

        parsed = ParsedPacket()
        parsed.timestamp = float(pkt.time)

        if prev_timestamp is not None:
            parsed.iat = max(0.0, parsed.timestamp - prev_timestamp)

        # Ethernet
        if Ether in pkt:
            eth = pkt[Ether]
            parsed.eth_dhost    = bytes.fromhex(eth.dst.replace(":", ""))
            parsed.eth_shost    = bytes.fromhex(eth.src.replace(":", ""))
            parsed.eth_ethertype = eth.type

        # IPv4
        if IP in pkt:
            ip = pkt[IP]
            parsed.ipv4_ver   = ip.version
            parsed.ipv4_hl    = ip.ihl
            parsed.ipv4_tos   = ip.tos
            parsed.ipv4_tl    = ip.len
            parsed.ipv4_id    = ip.id
            parsed.ipv4_dfbit = int(ip.flags.DF)
            parsed.ipv4_mfbit = int(ip.flags.MF)
            parsed.ipv4_rbit  = 0
            parsed.ipv4_foff  = ip.frag
            parsed.ipv4_ttl   = ip.ttl
            parsed.ipv4_proto = ip.proto
            parsed.ipv4_cksum = ip.chksum
            parsed.ipv4_src   = ip.src
            parsed.ipv4_dst   = ip.dst
            if ip.options:
                parsed.ipv4_opt = (
                    bytes(ip.options)[:40]
                    if isinstance(ip.options, bytes) else None
                )

        # IPv6
        elif IPv6 in pkt:
            ip6 = pkt[IPv6]
            parsed.ipv6_ver = 6
            parsed.ipv6_tc  = ip6.tc
            parsed.ipv6_fl  = ip6.fl
            parsed.ipv6_len = ip6.plen
            parsed.ipv6_nh  = ip6.nh
            parsed.ipv6_hl  = ip6.hlim
            parsed.ipv6_src = ip6.src
            parsed.ipv6_dst = ip6.dst

        # TCP
        if TCP in pkt:
            tcp = pkt[TCP]
            parsed.tcp_sprt  = tcp.sport
            parsed.tcp_dprt  = tcp.dport
            parsed.sport     = tcp.sport
            parsed.dport     = tcp.dport
            parsed.tcp_seq   = tcp.seq
            parsed.tcp_ackn  = tcp.ack
            parsed.tcp_doff  = tcp.dataofs
            parsed.tcp_wsize = tcp.window
            parsed.tcp_cksum = tcp.chksum
            parsed.tcp_urp   = tcp.urgptr

            flags = int(tcp.flags)
            parsed.tcp_ns  = (flags >> 8) & 1
            parsed.tcp_cwr = (flags >> 7) & 1
            parsed.tcp_ece = (flags >> 6) & 1
            parsed.tcp_urg = (flags >> 5) & 1
            parsed.tcp_ackf= (flags >> 4) & 1
            parsed.tcp_psh = (flags >> 3) & 1
            parsed.tcp_rst = (flags >> 2) & 1
            parsed.tcp_syn = (flags >> 1) & 1
            parsed.tcp_fin = flags & 1

            if tcp.dataofs and tcp.dataofs > 5:
                tcp_raw    = bytes(tcp)
                header_len = tcp.dataofs * 4
                options    = tcp_raw[20:header_len]
                parsed.tcp_opt = options[:40]

        # UDP
        elif UDP in pkt:
            udp = pkt[UDP]
            parsed.udp_sport = udp.sport
            parsed.udp_dport = udp.dport
            parsed.udp_len   = udp.len
            parsed.udp_cksum = udp.chksum
            parsed.sport     = udp.sport
            parsed.dport     = udp.dport

        # ICMP
        elif ICMP in pkt:
            icmp = pkt[ICMP]
            parsed.icmp_type  = icmp.type
            parsed.icmp_code  = icmp.code
            parsed.icmp_cksum = icmp.chksum

        # Payload
        if self.include_payload and Raw in pkt:
            raw = bytes(pkt[Raw].load)
            parsed.payload_bytes = raw[:self.max_payload_bytes]
            parsed.payload_len   = len(raw)

        return parsed

    def parse_sequence(self, packets: List[Packet]) -> List[ParsedPacket]:
        result   = []
        prev_ts  = None
        for pkt in packets:
            p = self.parse(pkt, prev_timestamp=prev_ts)
            if p is not None:
                prev_ts = p.timestamp
                result.append(p)
        return result
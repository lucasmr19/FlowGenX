from __future__ import annotations

from dataclasses import dataclass
from typing import Optional



@dataclass(kw_only=True)
class TrafficSample:
    label: Optional[int] = None
    class_name: Optional[str] = None   # "Benign", "Malware", "Skype", etc.
    source: Optional[str] = None       # ruta del pcap origen
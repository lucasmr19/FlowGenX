"""
src/reconstruction/__init__.py
================================
API pública del módulo de reconstrucción.

Importaciones mínimas para uso cotidiano:

    from src.reconstruction import (
        ReconstructorRegistry,
        SyntheticSample,
        ParsedPacket,
    )

    reconstructor = ReconstructorRegistry.get_for_representation(
        representation=rep,
        model_name="ddpm",
    )
    flows = reconstructor.reconstruct(samples_gen, labels=y_gen)
"""

from src.reconstruction.base import (
    BaseReconstructor,
    ParsedPacket,
    SyntheticSample,
)
from src.reconstruction.registry import ReconstructorRegistry

# Reconstructores concretos — importados explícitamente para que los IDEs
# puedan resolverlos y para que el registro lazy funcione en tests.
from src.reconstruction.sequential import (
    FlatTokenizerReconstructor,
    SemanticByteReconstructor,
    ProtocolAwareReconstructor,
)
from src.reconstruction.vision import (
    GAFReconstructor,
    NprintImageReconstructor,
)

# Serialización — import condicional; Scapy puede no estar instalado.
try:
    from src.reconstruction.serialization import (
        flows_to_pcap,
        flow_to_packets,
        pcap_to_flows,
    )
    _SERIALIZATION_AVAILABLE = True
except ImportError:
    _SERIALIZATION_AVAILABLE = False

__all__ = [
    # Estructuras de datos
    "ParsedPacket",
    "SyntheticSample",
    # Contrato abstracto
    "BaseReconstructor",
    # Fábrica
    "ReconstructorRegistry",
    # Reconstructores concretos
    "FlatTokenizerReconstructor",
    "SemanticByteReconstructor",
    "ProtocolAwareReconstructor",
    "GAFReconstructor",
    "NprintImageReconstructor",
    # Serialización (disponible si Scapy instalado)
    "flows_to_pcap",
    "flow_to_packets",
    "pcap_to_flows",
]
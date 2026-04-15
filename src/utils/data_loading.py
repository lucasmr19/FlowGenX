from pathlib import Path
from src.representations import TrafficRepresentation
from src.data_utils.preprocessing import ParsedPacket
from src.reconstruction.base import PacketFields

def load_samples_from_directory(
    data_dir: Path,
    representation: TrafficRepresentation,
    max_packets: int = 100
):
    samples = []

    label_map = {
        "Benign": 0,
        "Malware": 1,
    }

    pipeline = representation.build_preprocessing_pipeline(max_packets=max_packets)

    for label_name, label_id in label_map.items():
        for pcap in (data_dir / label_name).glob("*.pcap"):

            pcap_samples = pipeline.process(pcap)

            for s in pcap_samples:
                s.label = label_id

            samples.extend(pcap_samples)

    return samples

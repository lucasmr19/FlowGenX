from ..representations import SequenceTrafficEncoder, VisionTrafficEncoder

def build_traffic_encoder(rep_name: str, **kwargs):
    if rep_name in ["semantic_byte", "flat_tokenizer", "protocol_aware"]:
        vocab_size = kwargs.get("vocab_size", None)
        return SequenceTrafficEncoder(vocab_size=vocab_size)

    elif rep_name in ["gasf", "nprint_image"]:
        return VisionTrafficEncoder(in_channels=kwargs.get("in_channels", 1))

    else:
        raise ValueError(f"Unknown representation: {rep_name}")
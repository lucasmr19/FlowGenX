"""
tests/integration/test_real_pcap_sequential_tokenizers.py
=============================================

Test de integración real para tokenizadores secuenciales usando PCAP.

Pipeline:
    PCAP -> Flows -> Tokenizer -> Tensor -> Tokens

Evalúa:
- Correctitud del pipeline
- Calidad de la secuencia
- Métricas tipo NLP (padding, vocab, diversidad)
"""

import numpy as np
import torch
from pathlib import Path

from src.preprocessing import PCAPPipeline
from src.representations.sequential.tokenizer import (
    FlatTokenizer,
    ProtocolAwareTokenizer,
    SemanticByteTokenizer,
    SequentialConfig,
    ProtocolAwareConfig,
    SemanticByteConfig,
)

PCAP_PATH = Path("data/pcap/Benign/Gmail.pcap")

print(f"\n[TEST] PCAP: {PCAP_PATH}")
print("=" * 60)


# -----------------------------------------------------------
# Métricas específicas para secuencias
# -----------------------------------------------------------

def padding_ratio(x: np.ndarray, pad_id: int) -> float:
    return float(np.mean(x == pad_id))


def unique_token_ratio(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(len(np.unique(x)) / x.size)


def sequence_length(x: np.ndarray, pad_id: int) -> int:
    return int(np.sum(x != pad_id))


# -----------------------------------------------------------
# Test principal
# -----------------------------------------------------------

def test_real_pcap_sequential_tokenizers():
    # -----------------------------
    # 1. Cargar datos reales
    # -----------------------------
    pipeline = PCAPPipeline(max_packets=None)
    flows = pipeline.process(PCAP_PATH)

    assert len(flows) > 0, "No se cargaron flows"

    flow = flows[0]

    print(f"\n[INFO] Nº flows: {len(flows)}")
    print(f"[INFO] Flow seleccionado: {len(flow.packets)} packets")

    # -----------------------------
    # 2. Inicializar tokenizadores
    # -----------------------------
    flat = FlatTokenizer(
        SequentialConfig(
            max_length=256,
            include_payload=False
        )
    )

    proto = ProtocolAwareTokenizer(
        ProtocolAwareConfig(
            max_length=256,
            include_payload=False
        )
    )

    semantic_byte = SemanticByteTokenizer(
        SemanticByteConfig(
            max_length=256,
            include_payload=True,
            max_payload_tokens=32
        )
    )

    tokenizers = {
        "Flat": flat,
        "ProtocolAware": proto,
        "SemanticByte": semantic_byte
    }

    # -----------------------------
    # 3. Fit (IMPORTANTE)
    # -----------------------------
    train_flows = flows[:50]

    for name, tok in tokenizers.items():
        print(f"\n[FIT] {name}")
        tok.fit(train_flows)

    # -----------------------------
    # 4. Evaluación
    # -----------------------------
    print("\n" + "=" * 60)
    print("EVALUACIÓN")
    print("=" * 60)

    for name, tok in tokenizers.items():
        print(f"\n--- {name} ---")

        encoded = tok.encode(flow)
        decoded = tok.decode(encoded)

        x = encoded.numpy()

        pad_ratio = padding_ratio(x, tok.vocab.pad_id)
        uniq_ratio = unique_token_ratio(x)
        eff_len = sequence_length(x, tok.vocab.pad_id)

        print(f"Shape:               {x.shape}")
        print(f"Effective length:    {eff_len}")
        print(f"Padding ratio:       {pad_ratio:.4f}")
        print(f"Unique token ratio:  {uniq_ratio:.4f}")
        print(f"Vocab size:          {tok.vocab.vocab_size}")

        # -----------------------------
        # Ejemplo legible
        # -----------------------------
        print("\nSample tokens:")
        print(" ".join(decoded[:40]))

        # -----------------------------
        # Sanity checks (clave)
        # -----------------------------
        assert x.shape[0] == tok.cfg.max_length
        assert tok.vocab.vocab_size > 10
        assert eff_len > 0

        # Evitar secuencias degeneradas
        assert uniq_ratio > 0.01


# -----------------------------------------------------------
# Test comparativo más profundo (TFG-level)
# -----------------------------------------------------------

def test_compare_tokenizers_statistics():
    pipeline = PCAPPipeline(max_packets=None)
    flows = pipeline.process(PCAP_PATH)[:50]

    flat = FlatTokenizer(SequentialConfig(max_length=96))
    proto = ProtocolAwareTokenizer(ProtocolAwareConfig(max_length=96))
    semantic_byte = SemanticByteTokenizer(SemanticByteConfig(
        max_length=96,
        include_payload=True,
        max_payload_tokens=32
    ))

    tokenizers = {
        "Flat": flat,
        "ProtocolAware": proto,
        "SemanticByte": semantic_byte
    }

    # -----------------------------
    # Fit
    # -----------------------------
    for tok in tokenizers.values():
        tok.fit(flows)

    # -----------------------------
    # Collect stats
    # -----------------------------
    stats = {name: [] for name in tokenizers}

    for flow in flows:
        for name, tok in tokenizers.items():
            stats[name].append(tok.encode(flow).numpy())

    print("\n" + "=" * 60)
    print("GLOBAL STATS")
    print("=" * 60)

    for name, seqs in stats.items():
        seqs = np.stack(seqs)
        tok = tokenizers[name]

        pad_id = tok.vocab.pad_id

        pad_ratio = np.mean(seqs == pad_id)

        uniq_ratio = np.mean([
            len(np.unique(s)) / len(s)
            for s in seqs
        ])

        eff_len = np.mean([
            np.sum(s != pad_id)
            for s in seqs
        ])

        vocab_size = tok.vocab.vocab_size

        print(f"\n{name}:")
        print(f"Mean padding:        {pad_ratio:.4f}")
        print(f"Mean diversity:      {uniq_ratio:.4f}")
        print(f"Mean effective len:  {eff_len:.2f}")
        print(f"Vocab size:          {vocab_size}")
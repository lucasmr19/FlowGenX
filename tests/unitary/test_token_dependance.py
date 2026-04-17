"""
tests/unitary/test_token_dependance.py
"""

import pytest
import torch
from pathlib import Path
import numpy as np

from src.models_ml.transformer.model import TrafficTransformer, TransformerConfig
from src.models_ml.gan.model import TrafficGAN, GANConfig
from src.models_ml.diffusion.ddpm import TrafficDDPM, DiffusionConfig

from src.representations import (
    FlatTokenizer, SequentialConfig,
    SemanticByteConfig, SemanticByteTokenizer,
    ProtocolAwareTokenizer, ProtocolAwareConfig,
    GAFConfig, GAFRepresentation,
    NprintImageConfig, NprintImageRepresentation,
)
from src.data_utils.loaders import build_datamodule_from_dir
from src.reconstruction.build import build_reconstructor

DATA_DIR = Path("data/pcap")

def train_model_quick(
    model,
    X: torch.Tensor,
    model_name: str,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Bucle de entrenamiento mínimo para smoke-tests de integración.
    """
    B = X.shape[0]
    batch_size = 8

    if model_name == "transformer":
        for i in range(0, B, batch_size):
            model.train_step(X[i : i + batch_size])

        return model.generate(
            n_samples=min(B, 16),
            max_new_tokens=X.shape[1],
            labels=labels,
        )

    elif model_name == "gan":
        optim = model.configure_optimizers(lr=1e-4)
        use_cond = getattr(model.cfg, "num_classes", 0) > 0

        if labels is not None and not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long)

        for i in range(0, B, batch_size):
            batch = X[i : i + batch_size]

            batch_labels = None
            if use_cond:
                batch_labels = (
                    labels[i : i + batch_size]
                    if labels is not None
                    else model._sample_labels(batch.shape[0])
                )

            model.train_step_discriminator(
                batch, optim["discriminator"], labels=batch_labels
            )
            model.train_step_generator(
                optim["generator"], batch_size=batch.shape[0], labels=batch_labels
            )

        n_gen = min(B, 16)
        gen_labels = None
        if use_cond:
            gen_labels = labels[:n_gen] if labels is not None else model._sample_labels(n_gen)

        return model.generate(n_samples=n_gen, labels=gen_labels)

    elif model_name == "ddpm":
        for i in range(0, B, 4):
            model.train_step(X[i : i + 4])

        return model.generate(n_samples=min(B, 16), labels=labels)

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

@pytest.mark.parametrize("model_name", ["transformer", "gan"])
def test_reconstruction_depends_on_tokens_semantic_byte(model_name):
    rep = SemanticByteTokenizer(SemanticByteConfig(max_length=64))

    dm = build_datamodule_from_dir(
        DATA_DIR,
        representation=rep,
        aggregator=rep.get_default_aggregator(),
        max_packets=100,
        batch_size=16,
        seed=42,
    )
    dm.setup()

    X, y = next(iter(dm.train_dataloader()))

    # Modelo
    vocab_size = int(X.max().item() + 1)

    if model_name == "transformer":
        model = TrafficTransformer(
            TransformerConfig(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2)
        ).build()
    elif model_name == "gan":
        model = TrafficGAN(
            GANConfig(vocab_size=vocab_size, seq_len=X.shape[1])
        ).build()

    # Generación rápida
    samples_gen = train_model_quick(model, X, model_name, labels=y)

    reconstructor = build_reconstructor("semantic_byte", model_name)

    # TEST CLAVE
    tokens1 = X[0].unsqueeze(0)
    tokens2 = X[1].unsqueeze(0)

    flow1 = reconstructor.reconstruct(tokens1, labels=y[:1])[0]
    flow2 = reconstructor.reconstruct(tokens2, labels=y[1:2])[0]

    # --- ASSERTS ---
    # 1. No deben ser idénticos
    assert flow1.src_ip != flow2.src_ip or flow1.dst_ip != flow2.dst_ip or flow1.sport != flow2.sport, \
        "La reconstrucción NO depende de los tokens (flows idénticos)"

    # 2. Payload debe diferir
    p1 = flow1.packets[0].payload_bytes
    p2 = flow2.packets[0].payload_bytes

    assert p1 != p2, "El payload no depende de los tokens"

@pytest.mark.parametrize("model_name", ["transformer", "gan"])
def test_model_does_not_collapse_semantic_byte(model_name):
    rep = SemanticByteTokenizer(SemanticByteConfig(max_length=64))

    dm = build_datamodule_from_dir(
        DATA_DIR,
        representation=rep,
        aggregator=rep.get_default_aggregator(),
        max_packets=100,
        batch_size=16,
        seed=42,
    )
    dm.setup()

    X, y = next(iter(dm.train_dataloader()))

    vocab_size = int(X.max().item() + 1)

    if model_name == "transformer":
        model = TrafficTransformer(
            TransformerConfig(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2)
        ).build()
    elif model_name == "gan":
        model = TrafficGAN(
            GANConfig(vocab_size=vocab_size, seq_len=X.shape[1])
        ).build()

    samples_gen = train_model_quick(model, X, model_name, labels=y)

    # TEST CLAVE: diversidad
    unique_tokens = torch.unique(samples_gen)

    assert len(unique_tokens) > 5, \
        "Colapso del modelo: genera tokens casi constantes"

@pytest.mark.parametrize("model_name", ["transformer", "gan"])
def test_reconstruction_has_structure_semantic_byte(model_name):
    rep = SemanticByteTokenizer(SemanticByteConfig(max_length=64))

    dm = build_datamodule_from_dir(
        DATA_DIR,
        representation=rep,
        aggregator=rep.get_default_aggregator(),
        max_packets=100,
        batch_size=16,
        seed=42,
    )
    dm.setup()

    X, y = next(iter(dm.train_dataloader()))

    vocab_size = int(X.max().item() + 1)

    if model_name == "transformer":
        model = TrafficTransformer(
            TransformerConfig(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2)
        ).build()
    elif model_name == "gan":
        model = TrafficGAN(
            GANConfig(vocab_size=vocab_size, seq_len=X.shape[1])
        ).build()

    samples_gen = train_model_quick(model, X, model_name, labels=y)

    reconstructor = build_reconstructor("semantic_byte", model_name)
    flows = reconstructor.reconstruct(samples_gen, labels=y[:samples_gen.shape[0]])

    # TEST CLAVE
    num_packets = [len(f.packets) for f in flows]

    assert any(n > 1 for n in num_packets), \
        "Todos los flows tienen un solo paquete → no hay estructura temporal"

def test_ddpm_generation_not_constant_gasf():
    rep = GAFRepresentation(GAFConfig(image_size=128, method="summation"))

    dm = build_datamodule_from_dir(
        DATA_DIR,
        representation=rep,
        aggregator=rep.get_default_aggregator(),
        batch_size=8,
    )
    dm.setup()

    X, y = next(iter(dm.train_dataloader()))

    B, C, H, W = X.shape

    model = TrafficDDPM(
        DiffusionConfig(in_channels=C, image_height=H, image_width=W, timesteps=10)
    ).build()

    samples = train_model_quick(model, X, "ddpm", labels=y)

    # diversidad básica
    assert not torch.allclose(samples[0], samples[1]), \
        "DDPM genera imágenes idénticas"
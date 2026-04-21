# tests/integration/test_data_representations_generative_model.py
# =================================================================
# Test de integración:
# pcap -> representaciones -> modelos -> mini training
#
# Incluye:
# - loops de entrenamiento (epochs + batches)
# - separación dataset/batch
# - checks de convergencia básica
# - generación de muestras
# =================================================================

import pytest
import torch
from pathlib import Path

from src.models_ml.transformer.model import TrafficTransformer, TransformerConfig
from src.models_ml.gan.model import TrafficGAN, GANConfig
from src.models_ml.diffusion.ddpm import TrafficDDPM, DiffusionConfig

from src.representations.sequential.tokenizer import FlatTokenizer, SequentialConfig
from src.representations.vision import (
    GAFRepresentation, GAFConfig,
    NprintImageRepresentation, NprintImageConfig
)

from src.preprocessing import PCAPPipeline, PacketWindowAggregator

PCAP_PATH = Path("data/pcap/Benign/Gmail.pcap")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def flows():
    pipeline = PCAPPipeline(max_packets=None)
    return pipeline.process(PCAP_PATH)

@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def windows():
    pipeline = PCAPPipeline(
        aggregator=PacketWindowAggregator,
        max_packets=None,
        window_size=256
    )
    return pipeline.process(PCAP_PATH)

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def assert_valid_tensor(x: torch.Tensor):
    assert torch.is_tensor(x)
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def train_loop(model, dataset, n_epochs=2, batch_size=4):
    losses = []

    for epoch in range(n_epochs):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]

            out = model.train_step(batch)
            loss = out["loss"]

            losses.append(loss.item())

    return losses


def train_gan(model, dataset, optimizers, n_epochs=2, batch_size=4):
    losses_g, losses_d = [], []
    wasserstein_vals = []

    for epoch in range(n_epochs):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]

            out_d = model.train_step_discriminator(
                batch, optimizers["discriminator"]
            )
            losses_d.append(out_d["loss_d"].item())
            wasserstein_vals.append(out_d["wasserstein_dist"].item())

            out_g = model.train_step_generator(
                optimizers["generator"]
            )
            losses_g.append(out_g["loss_g"].item())

    return losses_g, losses_d, wasserstein_vals


# ---------------------------------------------------------------------------
# TEST 1: Transformer + FlatTokenizer
# ---------------------------------------------------------------------------
def test_transformer_with_training(flows):
    print("\n[TEST] Transformer + FlatTokenizer (training loop)")

    # Representation
    rep = FlatTokenizer(SequentialConfig(max_length=64))
    rep.fit(flows[:80])

    dataset = rep.encode_batch(flows[:120])  # dataset real
    assert dataset.ndim == 2

    vocab_size = int(dataset.max().item() + 1)

    # Model
    model = TrafficTransformer(
        TransformerConfig(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2
        )
    ).build()

    # Train
    losses = train_loop(model, dataset, n_epochs=3)

    assert all(torch.isfinite(torch.tensor(losses)))
    print(f"  loss first: {losses[0]:.4f} | last: {losses[-1]:.4f}")

    # Convergencia débil
    assert losses[-1] <= losses[0] or abs(losses[-1] - losses[0]) < 1e-2

    # Generate
    samples = model.generate(n_samples=2, max_new_tokens=32)
    assert samples.ndim == 2

# ---------------------------------------------------------------------------
# TEST 2: GAN + FlatTokenizer
# ---------------------------------------------------------------------------

def test_gan_training(flows):
    print("\n[TEST] GAN + FlatTokenizer (training loop)")

    rep = FlatTokenizer(SequentialConfig(max_length=64))
    rep.fit(flows[:80])

    dataset = rep.encode_batch(flows[:120])
    vocab_size = int(dataset.max().item() + 1)
    seq_len = dataset.shape[1]

    model = TrafficGAN(
        GANConfig(
            vocab_size=vocab_size,
            seq_len=seq_len
        )
    ).build()

    optimizers = model.configure_optimizers(lr=1e-4)

    losses_g, losses_d, w_vals = train_gan(model, dataset, optimizers, n_epochs=2)

    print(
        f"  G loss: {losses_g[-1]:.4f} | "
        f"D loss: {losses_d[-1]:.4f} | "
        f"W distance: {w_vals[-1]:.4f}"
    )

    # Wasserstein debería tender a ser positivo si D aprende algo
    assert any(w > 0 for w in w_vals), "Wasserstein distance never became positive"

    # No NaNs
    assert all(torch.isfinite(torch.tensor(losses_g)))
    assert all(torch.isfinite(torch.tensor(losses_d)))

    # Generate
    samples = model.generate(n_samples=2)
    assert samples.shape[0] == 2


# ---------------------------------------------------------------------------
# TEST 3: DDPM + GAF / NprintImage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("representation_type", ["gasf", "nprint_image"])
def test_ddpm_training(flows, windows, representation_type):
    print(f"\n[TEST] DDPM + {representation_type} (training loop)")

    # Representation
    if representation_type == "gasf":
        rep = GAFRepresentation(GAFConfig(image_size=64, method="summation"))
        rep.fit(flows[:80])
        dataset = rep.encode_batch(flows[:120])

    else:
        rep = NprintImageRepresentation(
            NprintImageConfig(max_packets=64, patch_size=8)
        )
        rep.fit(windows[:80])
        dataset = rep.encode_batch(windows[:120])

        if dataset.ndim == 3:
            dataset = dataset.unsqueeze(1)

    assert dataset.ndim == 4
    assert_valid_tensor(dataset)

    B, C, H, W = dataset.shape

    model = TrafficDDPM(
        DiffusionConfig(
            in_channels=C,
            image_height=H,
            image_width=W,
            timesteps=30  # rápido
        )
    ).build()

    # Train
    losses = train_loop(model, dataset, n_epochs=2)

    assert all(torch.isfinite(torch.tensor(losses)))
    print(f"  loss first: {losses[0]:.4f} | last: {losses[-1]:.4f}")

    # Generate
    samples = model.generate(n_samples=2)

    assert samples.shape[1:] == (C, H, W)
    assert torch.isfinite(samples).all()
    assert samples.std() > 0  # evita colapso
"""
tests/integration/test_data_representations_generative_model.py
================================================================
Test de integración:
pcap -> representaciones -> modelos generativos

Modelos definidos en src.generative_models:
- Transformer (discreto)
- GAN (secuencial)
- DDPM (imagen)
"""

import pytest
import torch
from pathlib import Path

from src.models_ml.transformer.model import TrafficTransformer, TransformerConfig
from src.models_ml.gan.model import TrafficGAN, GANConfig
from src.models_ml.diffusion.ddpm import TrafficDDPM, DiffusionConfig

from src.representations.sequential.tokenizer import FlatTokenizer, SequentialConfig
from src.representations.vision import GASFRepresentation, GASFConfig
from src.representations.vision import NprintRepresentation, NprintConfig, NprintImageConfig, NprintImageRepresentation

from src.data_utils.preprocessing import PCAPPipeline, PacketWindowAggregator

PCAP_PATH = Path("data/pcap/Benign/Gmail.pcap")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_valid_tensor(x: torch.Tensor):
    assert torch.is_tensor(x)
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


# ---------------------------------------------------------------------------
# TEST 1: Transformer + FlatTokenizer
# ---------------------------------------------------------------------------

def test_transformer_with_flat_tokenizer():
    print("\n[TEST] Transformer + FlatTokenizer")

    # Data
    pipeline = PCAPPipeline(max_packets=None)
    flows = pipeline.process(PCAP_PATH)

    # Representation
    cfg = SequentialConfig(max_length=64)
    rep = FlatTokenizer(cfg)
    rep.fit(flows[:40])

    batch = rep.encode_batch(flows[40:45])  # (B, L)
    assert batch.ndim == 2

    vocab_size = int(batch.max().item() + 1)

    # Model
    model_cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=2
    )
    model = TrafficTransformer(model_cfg).build()

    # Train step
    out = model.train_step(batch)
    loss = out["loss"]

    assert_valid_tensor(loss)
    print(f"  loss: {loss.item():.4f}")

    # Generate
    samples = model.generate(n_samples=2, max_new_tokens=32)
    assert samples.ndim == 2
    print(f"  generate shape: {samples.shape}")


# ---------------------------------------------------------------------------
# TEST 2: GAN + FlatTokenizer / NprintImage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("representation_type", ["flat"])
def test_gan_with_valid_representations(representation_type):
    print(f"\n[TEST] GAN + {representation_type}")

    flow_pipeline = PCAPPipeline(max_packets=None)
    flows = flow_pipeline.process(PCAP_PATH)

    window_pipeline = PCAPPipeline(
        aggregator=PacketWindowAggregator,
        max_packets=None,
        window_size=256  # reducido para test
    )
    windows = window_pipeline.process(PCAP_PATH)

    # -------------------------
    # Representation selection
    # -------------------------
    if representation_type == "flat":
        rep = FlatTokenizer(SequentialConfig(max_length=64))
        rep.fit(flows[:40])
        batch = rep.encode_batch(flows[40:45])
        vocab_size = int(batch.max().item() + 1)
        seq_len = batch.shape[1]

    elif representation_type == "nprint_image":
        rep = NprintImageRepresentation(NprintImageConfig(max_packets=128, patch_size=4))
        rep.fit(windows[:40])
        batch = rep.encode_batch(windows[40:45])
        vocab_size = None
        # batch = (B, C, H, W), aplanamos temporalmente para GAN si es necesario
        batch = batch.permute(0, 2, 3, 1).reshape(len(batch), -1)  # (B, seq_len)
        seq_len = batch.shape[1]

    assert_valid_tensor(batch)
        
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # -------------------------
    # Model
    # -------------------------
    model_cfg = GANConfig(
        vocab_size=vocab_size if vocab_size else 0,
        seq_len=seq_len,
    )
    model = TrafficGAN(model_cfg).build()
    optimizers = model.configure_optimizers(lr=1e-4)

    # -------------------------
    # Train discriminator
    # -------------------------
    losses_d = model.train_step_discriminator(batch, optimizers["discriminator"])
    assert "loss_d" in losses_d or len(losses_d) > 0

    # -------------------------
    # Train generator
    # -------------------------
    losses_g = model.train_step_generator(optimizers["generator"])
    assert "loss_g" in losses_g or len(losses_g) > 0

    print("  GAN train steps OK")

    # -------------------------
    # Generate
    # -------------------------
    samples = model.generate(n_samples=2)
    assert samples.shape[0] == 2
    print(f"  generate shape: {samples.shape}")


# ---------------------------------------------------------------------------
# TEST 3: DDPM + GASF / NprintImage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("representation_type", ["gasf", "nprint_image"])
def test_ddpm_with_valid_representations(representation_type):
    print(f"\n[TEST] DDPM + {representation_type}")

    flow_pipeline = PCAPPipeline(max_packets=None)
    flows = flow_pipeline.process(PCAP_PATH)

    window_pipeline = PCAPPipeline(
        aggregator=PacketWindowAggregator,
        max_packets=None,
        window_size=256
    )
    windows = window_pipeline.process(PCAP_PATH)

    # -------------------------
    # Representation selection
    # -------------------------
    if representation_type == "gasf":
        rep = GASFRepresentation(GASFConfig(image_size=16, n_steps=16))
        rep.fit(flows[:40])
        batch = rep.encode_batch(flows[40:45])  # (B, C, H, W)

    elif representation_type == "nprint_image":
        rep = NprintImageRepresentation(NprintImageConfig(max_packets=64, patch_size=8))
        rep.fit(windows[:40])
        batch = rep.encode_batch(windows[40:45])
        if batch.ndim == 3:
            batch = batch.unsqueeze(1)  # (B, C, H, W)

    assert batch.ndim == 4
    assert_valid_tensor(batch)

    B, C, H, W = batch.shape

    # -------------------------
    # Model
    # -------------------------
    model_cfg = DiffusionConfig(
        in_channels=C,
        image_height=H,
        image_width=W,
        timesteps=50  # reducido para test rápido
    )
    model = TrafficDDPM(model_cfg).build()

    # -------------------------
    # Train step
    # -------------------------
    out = model.train_step(batch)
    loss = out["loss"]
    assert_valid_tensor(loss)
    print(f"  loss: {loss.item():.4f}")

    # -------------------------
    # Generate
    # -------------------------
    samples = model.generate(n_samples=2)
    assert samples.shape[1:] == (C, H, W)
    print(f"  generate shape: {samples.shape}")
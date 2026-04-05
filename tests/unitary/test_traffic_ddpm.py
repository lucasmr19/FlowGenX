import torch
import pytest

from src.models_ml.diffusion.ddpm import TrafficDDPM
from src.models_ml.diffusion.config import DiffusionConfig


@pytest.fixture
def small_config():
    """Config pequeña para tests rápidos."""
    return DiffusionConfig(
        in_channels=1,
        image_height=16,
        image_width=16,
        base_ch=16,
        channel_mults=(1, 2),
        n_res_per_level=1,
        attention_levels=(1,),
        num_classes=3,
        timesteps=50,
        ddim_steps=10,
        beta_schedule="linear",
        device="cpu",
    )


@pytest.fixture
def model(small_config):
    return TrafficDDPM(small_config).build()


def test_ddpm_build(model):
    """El modelo debe construirse correctamente."""
    assert model._built
    assert hasattr(model, "unet")
    assert hasattr(model, "scheduler")


def test_train_step_returns_loss(model):
    """train_step debe devolver loss."""
    B = 4
    x = torch.randn(B, 1, 16, 16)
    labels = torch.randint(0, model.cfg.num_classes, (B,))

    out = model.train_step((x, labels))

    assert "loss" in out
    assert "t_mean" in out
    assert torch.is_tensor(out["loss"])


def test_generate_ddim(model):
    """La generación DDIM debe producir imágenes del tamaño correcto."""
    samples = model.generate(
        n_samples=2,
        labels=torch.zeros(2, dtype=torch.long),
        use_ddim=True,
    )

    assert samples.shape == (
        2,
        model.cfg.in_channels,
        model.cfg.image_height,
        model.cfg.image_width,
    )


def test_generate_ddpm(model):
    """La generación DDPM estándar también debe funcionar."""
    samples = model.generate(
        n_samples=2,
        labels=torch.zeros(2, dtype=torch.long),
        use_ddim=False,
    )

    assert samples.shape == (
        2,
        model.cfg.in_channels,
        model.cfg.image_height,
        model.cfg.image_width,
    )
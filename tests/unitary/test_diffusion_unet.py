import torch
import pytest

from src.models_ml.diffusion.unet import UNet2D


@pytest.fixture
def small_unet():
    """UNet pequeño para tests rápidos."""
    return UNet2D(
        in_channels=1,
        base_ch=16,
        channel_mults=(1, 2),
        n_res_per_level=1,
        attention_levels=(1,),
        n_heads=2,
        dropout=0.0,
        num_classes=0,
    )


def test_unet_forward_shape(small_unet):
    """El UNet debe devolver un tensor con la misma forma que la entrada."""
    B, C, H, W = 2, 1, 16, 16

    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))

    out = small_unet(x, t)

    assert out.shape == (B, C, H, W)


def test_unet_time_embedding_affects_output(small_unet):
    """Cambiar el timestep debe cambiar la predicción."""
    B, C, H, W = 2, 1, 16, 16

    x = torch.randn(B, C, H, W)

    t1 = torch.zeros(B, dtype=torch.long)
    t2 = torch.ones(B, dtype=torch.long)

    out1 = small_unet(x, t1)
    out2 = small_unet(x, t2)

    assert not torch.allclose(out1, out2)


def test_unet_class_conditioning():
    """El UNet debe aceptar conditioning por clase."""
    model = UNet2D(
        in_channels=1,
        base_ch=16,
        channel_mults=(1, 2),
        n_res_per_level=1,
        attention_levels=(1,),
        n_heads=2,
        dropout=0.0,
        num_classes=5,
    )

    B, C, H, W = 2, 1, 16, 16

    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    labels = torch.randint(0, 5, (B,))

    out = model(x, t, labels)

    assert out.shape == (B, C, H, W)
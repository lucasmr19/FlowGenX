import torch
import pytest

from src.models_ml.diffusion.ddpm import NoiseScheduler


@pytest.fixture
def scheduler():
    return NoiseScheduler(
        timesteps=100,
        schedule="cosine",
        device="cpu",
    )


def test_scheduler_shapes(scheduler):
    """Los coeficientes deben tener tamaño T."""
    T = scheduler.T

    assert scheduler.betas.shape[0] == T
    assert scheduler.sqrt_alphas_bar.shape[0] == T
    assert scheduler.sqrt_one_minus_ab.shape[0] == T


def test_q_sample_output_shape(scheduler):
    """q_sample debe preservar la forma de la imagen."""
    x0 = torch.randn(4, 1, 16, 16)
    t = torch.randint(0, scheduler.T, (4,))

    x_t, noise = scheduler.q_sample(x0, t)

    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape


def test_predict_x0_reconstruction(scheduler):
    """
    Si usamos el ruido real, la reconstrucción de x0
    debe aproximarse al original.
    """
    x0 = torch.randn(2, 1, 16, 16)
    t = torch.randint(0, scheduler.T, (2,))

    x_t, noise = scheduler.q_sample(x0, t)

    x0_pred = scheduler.predict_x0_from_eps(x_t, t, noise)

    assert torch.allclose(x0, x0_pred, atol=1e-4)
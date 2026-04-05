import torch
import pytest

from src.models_ml.gan.model import (
    TrafficGAN,
    LSTMGenerator,
    TransformerDiscriminator,
)

from src.models_ml.gan.config import GANConfig


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def small_config():
    """Configuración pequeña para tests rápidos."""
    return GANConfig(
        vocab_size=50,
        seq_len=16,
        latent_dim=16,

        gen_hidden=32,
        gen_layers=1,
        gen_dropout=0.0,

        disc_d_model=32,
        disc_n_heads=4,
        disc_n_layers=1,
        disc_d_ff=64,
        disc_dropout=0.0,

        n_critic=2,
        use_gradient_penalty=True,
    )


@pytest.fixture
def model(small_config):
    """Modelo construido."""
    return TrafficGAN(small_config).build()


# ---------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------

def test_gan_build(model):
    """El modelo debe construir generator y discriminator."""
    assert model._built

    assert isinstance(model.generator, LSTMGenerator)
    assert isinstance(model.discriminator, TransformerDiscriminator)


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------

def test_generator_forward_shape(small_config):
    """El generador debe producir secuencias del tamaño correcto."""
    gen = LSTMGenerator(small_config)

    z = torch.randn(4, small_config.latent_dim)

    tokens, logits = gen(z, return_logits=True)

    assert tokens.shape == (4, small_config.seq_len)
    assert logits.shape == (4, small_config.seq_len, small_config.vocab_size)


def test_generator_generate(small_config):
    """generate() debe producir tokens discretos."""
    gen = LSTMGenerator(small_config)

    z = torch.randn(3, small_config.latent_dim)

    tokens = gen.generate(z)

    assert tokens.shape == (3, small_config.seq_len)
    assert tokens.dtype == torch.long


# ---------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------

def test_discriminator_forward_shape(small_config):
    """El discriminador debe devolver un score por secuencia."""
    disc = TransformerDiscriminator(small_config)

    tokens = torch.randint(0, small_config.vocab_size, (4, small_config.seq_len))

    scores = disc(tokens)

    assert scores.shape == (4,)


# ---------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------

def test_configure_optimizers(model):
    """Debe devolver optimizadores separados."""
    opts = model.configure_optimizers(lr=1e-4)

    assert "generator" in opts
    assert "discriminator" in opts


# ---------------------------------------------------------------------
# Discriminator step
# ---------------------------------------------------------------------

def test_train_step_discriminator(model):
    """El paso del discriminador debe devolver métricas válidas."""
    opts = model.configure_optimizers()

    batch = torch.randint(
        0,
        model.cfg.vocab_size,
        (4, model.cfg.seq_len),
    )

    out = model.train_step_discriminator(batch, opts["discriminator"])

    assert "loss_d" in out
    assert "d_real" in out
    assert "d_fake" in out
    assert "wasserstein_dist" in out

    assert torch.is_tensor(out["loss_d"])


# ---------------------------------------------------------------------
# Generator step
# ---------------------------------------------------------------------

def test_train_step_generator(model):
    """El paso del generador debe devolver loss_g."""
    opts = model.configure_optimizers()

    out = model.train_step_generator(opts["generator"], batch_size=4)

    assert "loss_g" in out
    assert torch.is_tensor(out["loss_g"])


# ---------------------------------------------------------------------
# Unified train_step
# ---------------------------------------------------------------------

def test_train_step_requires_optimizers(model):
    """train_step debe fallar si el Trainer no asigna optimizadores."""
    batch = torch.randint(
        0,
        model.cfg.vocab_size,
        (4, model.cfg.seq_len),
    )

    with pytest.raises(RuntimeError):
        model.train_step(batch)


def test_train_step_with_optimizers(model):
    """train_step debe funcionar cuando el Trainer asigna optimizadores."""
    opts = model.configure_optimizers()

    model._opt_discriminator = opts["discriminator"]
    model._opt_generator = opts["generator"]

    batch = torch.randint(
        0,
        model.cfg.vocab_size,
        (4, model.cfg.seq_len),
    )

    out = model.train_step(batch)

    assert "loss_d" in out
    assert "loss" in out


# ---------------------------------------------------------------------
# Gradient penalty
# ---------------------------------------------------------------------

def test_gradient_penalty_positive(model):
    """El gradient penalty debe ser >= 0."""
    real = torch.randint(
        0,
        model.cfg.vocab_size,
        (4, model.cfg.seq_len),
        device=model.device,
    )

    fake = torch.randint(
        0,
        model.cfg.vocab_size,
        (4, model.cfg.seq_len),
        device=model.device,
    )

    gp = model._gradient_penalty(real, fake)

    assert gp >= 0


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def test_generate(model):
    """La generación final debe devolver secuencias discretas."""
    tokens = model.generate(n_samples=3)

    assert tokens.shape == (3, model.cfg.seq_len)
    assert tokens.dtype == torch.long
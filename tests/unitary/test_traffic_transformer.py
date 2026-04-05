import torch
import pytest

from src.models_ml.transformer.model import TrafficTransformer
from src.models_ml.transformer.config import TransformerConfig


@pytest.fixture
def small_config():
    """Config pequeña para tests rápidos."""
    return TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
        seed=42,
    )


@pytest.fixture
def model(small_config):
    """Modelo construido listo para usar."""
    return TrafficTransformer(small_config).build()


def test_model_build(model):
    """El modelo debe construirse correctamente."""
    assert model._built
    assert hasattr(model, "token_emb")
    assert hasattr(model, "transformer")
    assert hasattr(model, "lm_head")


def test_forward_shape(model):
    """Forward debe devolver logits (B, L, vocab)."""
    batch = torch.randint(
        0,
        model.cfg.vocab_size,
        (4, 16),
        device=model.device
    )

    logits = model.forward(batch)

    assert logits.shape == (4, 16, model.cfg.vocab_size)


def test_weight_tying(model):
    """LM head y embedding deben compartir pesos."""
    assert model.lm_head.weight is model.token_emb.embedding.weight


def test_train_step_returns_loss_and_ppl(model):
    """train_step debe devolver loss y perplexity."""
    batch = torch.randint(
        0,
        model.cfg.vocab_size,
        (4, 16),
        device=model.device
    )

    out = model.train_step(batch)

    assert "loss" in out
    assert "perplexity" in out

    assert torch.is_tensor(out["loss"])
    assert torch.is_tensor(out["perplexity"])


def test_train_step_ignores_padding(model):
    """El padding no debe romper el loss."""
    batch = torch.randint(
        0,
        model.cfg.vocab_size,
        (4, 16),
        device=model.device
    )

    batch[:, -3:] = model.cfg.pad_token_id

    out = model.train_step(batch)

    assert torch.isfinite(out["loss"])

def test_causal_mask():
    """ La máscara causal debe tener la forma correcta de triangular superior."""
    mask = TrafficTransformer._make_causal_mask(4, torch.device("cpu"))

    expected = torch.tensor(
        [
            [False, True,  True,  True],
            [False, False, True,  True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )

    assert torch.equal(mask, expected)


def test_generate_runs(model):
    """La generación debe producir tensores de tamaño correcto."""
    samples = model.generate(
        n_samples=3,
        max_new_tokens=10,
        bos_token_id=2,
        eos_token_id=3,
    )

    assert samples.shape[0] == 3
    assert samples.shape[1] <= 10
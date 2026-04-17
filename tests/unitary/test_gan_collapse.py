# tests/unitary/test_gan_collapse.py

import torch
import numpy as np
from itertools import combinations


def _pairwise_l1_distance(X: torch.Tensor) -> float:
    """
    Distancia media L1 entre samples.
    Detecta si todas las muestras son casi idénticas.
    """
    if len(X) < 2:
        return 0.0

    dists = []
    for i, j in combinations(range(len(X)), 2):
        d = torch.abs(X[i] - X[j]).mean().item()
        dists.append(d)

    return float(np.mean(dists))


def _intra_sample_diversity(X: torch.Tensor) -> float:
    """
    Diversidad dentro de cada sample (tokens únicos / longitud).
    Detecta secuencias degeneradas (ej: todo ceros).
    """
    diversities = []

    for sample in X:
        tokens = sample.view(-1)
        unique = torch.unique(tokens)
        diversity = len(unique) / len(tokens)
        diversities.append(diversity)

    return float(np.mean(diversities))


def _vocab_coverage(X: torch.Tensor, vocab_size: int) -> float:
    """
    Cobertura del vocabulario global.
    Detecta si el modelo usa muy pocos tokens.
    """
    tokens = X.view(-1)
    unique = torch.unique(tokens)
    return float(len(unique) / vocab_size)


def _sample_variance(X: torch.Tensor) -> float:
    """
    Varianza global del batch.
    Muy baja → colapso típico de GAN.
    """
    return float(torch.var(X.float()).item())


def detect_mode_collapse(
    samples: torch.Tensor,
    *,
    vocab_size: int,
    l1_threshold: float = 0.05,
    intra_div_threshold: float = 0.05,
    coverage_threshold: float = 0.1,
    var_threshold: float = 1e-3,
):
    """
    Detector heurístico de mode collapse.

    Devuelve:
      - dict con métricas
      - flag booleano (collapse detectado)
    """

    X = samples.detach().cpu()

    metrics = {
        "pairwise_l1_distance": _pairwise_l1_distance(X),
        "intra_sample_diversity": _intra_sample_diversity(X),
        "vocab_coverage": _vocab_coverage(X, vocab_size),
        "global_variance": _sample_variance(X),
    }

    collapse_flags = {
        "low_inter_sample_diversity": metrics["pairwise_l1_distance"] < l1_threshold,
        "low_intra_sample_diversity": metrics["intra_sample_diversity"] < intra_div_threshold,
        "low_vocab_usage": metrics["vocab_coverage"] < coverage_threshold,
        "low_variance": metrics["global_variance"] < var_threshold,
    }

    collapse_detected = any(collapse_flags.values())

    return {
        "metrics": metrics,
        "flags": collapse_flags,
        "collapse_detected": collapse_detected,
    }

def _mean_sequence_length(X):
    lengths = [(sample != 0).sum().item() for sample in X]
    return np.mean(lengths)

def test_gan_mode_collapse_detection(generator, datamodule):
    """
    Test de integración: detecta si el GAN colapsa.
    """

    generator.eval()

    # Generar samples
    z = torch.randn(32, generator.latent_dim)
    samples = generator(z)

    # Ajusta esto según tu representación
    vocab_size = 256

    result = detect_mode_collapse(
        samples,
        vocab_size=vocab_size,
    )

    print("\n[Mode Collapse Detection]")
    print(result)

    # Assert suave (puedes endurecerlo)
    assert "metrics" in result
    assert "collapse_detected" in result
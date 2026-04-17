"""
tests/integration/test_evaluation.py
=================================================================
Test de integración:

PCAP
  ↓
Representation (tokenizer / image)
  ↓
Model (Transformer / GAN / DDPM)
  ↓
Generated samples
  ↓
Projection (representation space)
  ↓
┌─────────────────────────────────────┐
│ EvaluationSuite                     │
│ - Statistical                       │
│ - Structural (representation)       │
│ - URS                               │
└─────────────────────────────────────┘
  ↓
Reconstruction
  ↓
Flows / Packets
  ↓
┌─────────────────────────────────────┐
│ TrafficStructuralEvaluator          │
│ - Packet validity                   │
│ - Flow coherence                    │
│ - TCP semantics                     │
│ - Protocol distribution             │
└─────────────────────────────────────┘
 ↓
 PCAP sintético con validez estructural a nivel de tráfico
=================================================================
"""

import pytest
import torch
from pathlib import Path
import numpy as np

from src.models_ml.transformer.model import TrafficTransformer, TransformerConfig
from src.models_ml.gan.model import TrafficGAN, GANConfig
from src.models_ml.diffusion.ddpm import TrafficDDPM, DiffusionConfig

from src.representations.sequential.tokenizer import (
    FlatTokenizer, SequentialConfig,
    SemanticByteConfig, SemanticByteTokenizer,
    ProtocolAwareTokenizer, ProtocolAwareConfig,
)
from src.representations.vision import (
    GAFRepresentation, GAFConfig,
    NprintImageRepresentation, NprintImageConfig,
)
from src.data_utils.loaders import build_datamodule_from_dir
from src.evaluation import (
    EvaluationSuite,
    SuiteResult,
    URSEvaluator,
    StatisticalEvaluator,
    StructuralEvaluator,
    TrafficStructuralEvaluator,
    TSTREvaluator,
    TRTREvaluator,
    TSTRTRTRComparisonEvaluator,
    TaskRunnerEvaluator,
    ClassificationProbeTask,
    AnomalyDetectionTask,
)
from src.utils.build_traffic_encoder import build_traffic_encoder
from src.reconstruction.build import build_reconstructor


# ---------------------------------------------------------------------------
# Configuración de datos
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/pcap")

# Combinaciones compatibles representación ↔ modelo
COMPATIBILITY = {
    "flat_tokenizer":  ["transformer", "gan"],
    "protocol_aware":  ["transformer", "gan"],
    "semantic_byte":   ["transformer", "gan"],
    "gasf":            ["ddpm"],
    "nprint_image":    ["ddpm"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_valid_tensor(x: torch.Tensor) -> None:
    assert torch.is_tensor(x), "No es un tensor"
    assert not torch.isnan(x).any(), "Hay NaNs"
    assert not torch.isinf(x).any(), "Hay infs"


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


# ---------------------------------------------------------------------------
# Parametrización
# ---------------------------------------------------------------------------
representations = ["semantic_byte", "flat_tokenizer", "protocol_aware", "gasf", "nprint_image"]
#representations = ["semantic_byte",]
models = ["transformer", "gan", "ddpm"]


# ---------------------------------------------------------------------------
# Test de integración completo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rep_name", representations)
@pytest.mark.parametrize("model_name", models)
def test_full_integration_with_datamodule(rep_name, model_name):
    if model_name not in COMPATIBILITY[rep_name]:
        pytest.skip(f"Incompatible: {rep_name} + {model_name}")

    print(f"\n[Integration Test] Representation={rep_name} | Model={model_name}")

    is_sequence = rep_name in ["semantic_byte", "flat_tokenizer", "protocol_aware"]

    # -----------------------------------------------------------------------
    # 1. Preparar representación
    # -----------------------------------------------------------------------
    if rep_name == "flat_tokenizer":
        rep = FlatTokenizer(SequentialConfig(max_length=64))
    elif rep_name == "semantic_byte":
        rep = SemanticByteTokenizer(SemanticByteConfig(max_length=64))
    elif rep_name == "protocol_aware":
        rep = ProtocolAwareTokenizer(ProtocolAwareConfig(max_length=64))
    elif rep_name == "gasf":
        rep = GAFRepresentation(GAFConfig(image_size=128, method="summation"))
    elif rep_name == "nprint_image":
        rep = NprintImageRepresentation(NprintImageConfig(max_packets=64, patch_size=8))
    else:
        raise ValueError(f"Representación desconocida: {rep_name}")

    # -----------------------------------------------------------------------
    # 2. Construir DataModule — usa el agregador nativo de la representación
    #    para evitar enviar Flow donde se espera PacketWindow y viceversa.
    # -----------------------------------------------------------------------
    dm = build_datamodule_from_dir(
        DATA_DIR,
        representation=rep,
        aggregator=rep.get_default_aggregator(),
        max_packets=100,
        batch_size=16,
        seed=42,
    )
    dm.setup()

    X, y_real = next(iter(dm.train_dataloader()))
    assert_valid_tensor(X)

    unique, counts = torch.unique(y_real, return_counts=True)
    print(f"Distribución de clases: {dict(zip(unique.tolist(), counts.tolist()))}")

    # -----------------------------------------------------------------------
    # 3. Inicializar modelo
    # -----------------------------------------------------------------------
    vocab_size: int | None = None  # solo válido para representaciones secuenciales

    if model_name == "transformer":
        vocab_size = int(X.max().item() + 1)
        model = TrafficTransformer(
            TransformerConfig(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2)
        ).build()

    elif model_name == "gan":
        vocab_size = int(X.max().item() + 1)
        seq_len = X.shape[1]
        model = TrafficGAN(GANConfig(vocab_size=vocab_size, seq_len=seq_len)).build()

    elif model_name == "ddpm":
        B, C, H, W = X.shape
        model = TrafficDDPM(
            DiffusionConfig(in_channels=C, image_height=H, image_width=W, timesteps=30)
        ).build()

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    # Encoder para URSEvaluator
    encoder = build_traffic_encoder(
        rep_name,
        vocab_size=vocab_size if is_sequence else None,
        in_channels=X.shape[1] if X.dim() == 4 else None,
    )

    # -----------------------------------------------------------------------
    # 4. Entrenamiento rápido + generación
    # -----------------------------------------------------------------------
    samples_gen = train_model_quick(model, X, model_name, labels=y_real)

    # Alinear tamaño del batch generado con el batch real
    n_real = X.shape[0]
    if samples_gen.shape[0] != n_real:
        samples_gen = samples_gen[:n_real]

    # Proyectar al espacio de la representación (no-op en la mayoría de casos)
    samples_gen = rep.project(samples_gen)
    assert_valid_tensor(samples_gen)
    print(f"[Projection] samples shape: {samples_gen.shape}")

    # -----------------------------------------------------------------------
    # 5. Evaluación estadística + estructural (nivel representación)
    # -----------------------------------------------------------------------
    evaluators = [
        StatisticalEvaluator(n_features_cap=64),
        StructuralEvaluator(
            representation_type=rep_name,
            vocab_size=vocab_size if is_sequence else None,
        ),
        URSEvaluator(encoder=encoder),
    ]

    suite = EvaluationSuite(
        evaluators=evaluators,
        representation_name=rep_name,
        model_name=model_name,
        verbose=False,
    )
    result = suite.run(X, samples_gen)
    summary = result.summary()
    print(f"[EvaluationSuite] {summary}")

    assert isinstance(result, SuiteResult)
    finite_vals = [v for v in summary.values() if not np.isnan(v)]
    assert all(np.isfinite(finite_vals)), "Alguna métrica no es finita"

    # -----------------------------------------------------------------------
    # 6. Reconstrucción a tráfico real
    # -----------------------------------------------------------------------
    reconstructor = build_reconstructor(rep_name, model_name)

    flows_gen  = reconstructor.reconstruct(samples_gen, labels=y_real[:samples_gen.shape[0]])
    flows_real = reconstructor.reconstruct(X,           labels=y_real) # ??

    assert isinstance(flows_gen, list) and len(flows_gen) > 0

    total_packets = sum(
        len(f.packets) if hasattr(f, "packets") else len(f)
        for f in flows_gen
    )
    assert total_packets > 0
    #for f in flows_gen[:3]:
    #    print(f"Ejemplo de flujo reconstruido: {f}")
    print(f"[Reconstruction] {len(flows_gen)} flows | {total_packets} packets")
    #print("REAL TOKENS:", X[0][:20])
    #print("GEN TOKENS:", samples_gen[0][:20])

    # -----------------------------------------------------------------------
    # 7. Evaluación estructural a nivel de tráfico
    # -----------------------------------------------------------------------
    traffic_evaluator = TrafficStructuralEvaluator()
    traffic_report = traffic_evaluator.evaluate(flows_real, flows_gen)
    #flows_reconstructed = reconstructor.reconstruct(X, labels=y_real)
    #traffic_evaluator.evaluate(flows_real, flows_reconstructed)
    traffic_summary = traffic_report.summary()
    print(f"[Traffic Structural] {traffic_summary}")

    finite_traffic = [v for v in traffic_summary.values() if not np.isnan(v)]
    assert all(np.isfinite(finite_traffic)), "Alguna métrica de tráfico no es finita"
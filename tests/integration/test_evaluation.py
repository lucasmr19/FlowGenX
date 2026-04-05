"""
tests/integration/test_evaluation.py
======================================
Test de integración del módulo de evaluación.

Cubre el pipeline completo:
    make_dataset → representación → modelo generativo (dummy) → evaluación

Para cada combinación representación-modelo se verifica:
1. StatisticalEvaluator  : métricas EMD, JS y correlación presentes y numéricas.
2. StructuralEvaluator   : métricas de validez presentes; valid_sample_rate ∈ [0,1].
3. TSTREvaluator         : accuracy y f1 ∈ [0,1] con etiquetas sintéticas.
4. EvaluationSuite       : orquestación correcta, summary() sin NaN críticos.

Nota sobre los datos de test
-----------------------------
Los datos son sintéticos (make_dataset), no .pcap reales.
Los tensores "sintéticos" son ruido aleatorio para simular un
modelo generativo imperfecto. Esto garantiza que las métricas
devuelvan valores no triviales y el test sea informativamente útil.
"""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Datos y representaciones
# ---------------------------------------------------------------------------
from src.utils.make_synthetic_flow import make_dataset
from src.representations.sequential.tokenizer import FlatTokenizer, SequentialConfig
from src.representations.vision import GASFRepresentation, GASFConfig
from src.representations.vision import NprintRepresentation, NprintConfig

# ---------------------------------------------------------------------------
# Módulo de evaluación
# ---------------------------------------------------------------------------
from src.evaluation import (
    EvaluationSuite,
    StatisticalEvaluator,
    StructuralEvaluator,
    TSTREvaluator,
    SuiteResult,
)

# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------

N_FLOWS = 80
N_TRAIN = 50
N_TEST = 20
N_SYNTH = 20
N_CLASSES = 4


def make_labels(n: int, n_classes: int = N_CLASSES) -> np.ndarray:
    """Etiquetas sintéticas balanceadas."""
    return np.array([i % n_classes for i in range(n)])


def make_noisy_synth(real: torch.Tensor, noise_scale: float = 0.3) -> torch.Tensor:
    """
    Simula un generador imperfecto: datos reales + ruido gaussiano.
    Más realista que ruido puro para que las métricas sean no triviales.
    """
    return real + noise_scale * torch.randn_like(real.float())


# ---------------------------------------------------------------------------
# Test 1: StatisticalEvaluator sobre las tres representaciones
# ---------------------------------------------------------------------------

class TestStatisticalEvaluator:

    def _run(self, real: torch.Tensor, synth: torch.Tensor):
        evaluator = StatisticalEvaluator(n_features_cap=128, bins=32)
        report = evaluator.evaluate(real, synth)
        return report

    def test_flat_tokenizer(self):
        flows = make_dataset(N_FLOWS)
        cfg = SequentialConfig(max_length=64)
        rep = FlatTokenizer(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        synth = make_noisy_synth(real)

        report = self._run(real, synth)
        summary = report.summary()

        print(f"\n[StatisticalEvaluator | FlatTokenizer] {summary}")

        assert "mean_emd" in summary
        assert "mean_js_divergence" in summary
        assert not np.isnan(summary["mean_emd"])
        assert not np.isnan(summary["mean_js_divergence"])
        assert summary["mean_emd"] >= 0.0
        assert 0.0 <= summary["mean_js_divergence"] <= 1.0

    def test_gasf_representation(self):
        flows = make_dataset(N_FLOWS)
        cfg = GASFConfig(image_size=16, feature="combined", n_steps=16)
        rep = GASFRepresentation(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        synth = make_noisy_synth(real)

        report = self._run(real, synth)
        summary = report.summary()

        print(f"\n[StatisticalEvaluator | GASF] {summary}")

        assert "mean_emd" in summary
        assert "mean_js_divergence" in summary
        # Con suficientes muestras debería haber la métrica de correlación
        if "pearson_corr_matrix_distance" in summary:
            assert summary["pearson_corr_matrix_distance"] >= 0.0

    def test_nprint_representation(self):
        flows = make_dataset(N_FLOWS)
        cfg = NprintConfig(max_packets=10)
        rep = NprintRepresentation(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        synth = make_noisy_synth(real)

        report = self._run(real, synth)
        summary = report.summary()

        print(f"\n[StatisticalEvaluator | Nprint] {summary}")

        assert "mean_emd" in summary
        assert "mean_js_divergence" in summary
        assert summary["mean_emd"] >= 0.0


# ---------------------------------------------------------------------------
# Test 2: StructuralEvaluator — validez por representación
# ---------------------------------------------------------------------------

class TestStructuralEvaluator:

    def test_sequential_valid_data(self):
        """Datos reales deberían tener valid_sample_rate ≈ 1.0."""
        flows = make_dataset(N_FLOWS)
        cfg = SequentialConfig(max_length=64)
        rep = FlatTokenizer(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST])
        vocab_size = rep.vocab  # asumimos que FlatTokenizer expone vocab_size

        evaluator = StructuralEvaluator(
            representation_type="sequential",
            vocab_size=vocab_size,
        )
        # Evaluar datos reales contra sí mismos
        report = evaluator.evaluate(real, real)
        summary = report.summary()

        print(f"\n[StructuralEvaluator | sequential | real vs real] {summary}")

        assert "valid_sample_rate" in summary
        assert 0.0 <= summary["valid_sample_rate"] <= 1.0
        # Los datos reales deben ser casi 100% válidos
        assert summary["valid_sample_rate"] > 0.9, (
            f"Los datos reales tienen valid_sample_rate={summary['valid_sample_rate']:.3f}, "
            "se esperaba > 0.9."
        )

    def test_sequential_random_noise_degrades(self):
        """Ruido aleatorio debe tener valid_sample_rate < datos reales."""
        flows = make_dataset(N_FLOWS)
        cfg = SequentialConfig(max_length=64)
        rep = FlatTokenizer(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST])
        vocab_size = rep.vocab

        # Generamos "sintético" con valores flotantes continuos (modelo de difusión sin redondear)
        synth_bad = torch.randn(N_SYNTH, 64) * vocab_size

        evaluator = StructuralEvaluator(
            representation_type="sequential",
            vocab_size=vocab_size,
        )
        report = evaluator.evaluate(real, synth_bad)
        summary = report.summary()

        print(f"\n[StructuralEvaluator | sequential | ruido] {summary}")

        # El ruido debe tener peor válida que los datos reales
        assert summary["valid_sample_rate"] < 0.5, (
            "Se esperaba que el ruido tuviera valid_sample_rate bajo."
        )

    def test_gasf_real_data_valid(self):
        """Imágenes GASF reales deben estar en [-1, 1]."""
        flows = make_dataset(N_FLOWS)
        cfg = GASFConfig(image_size=16, feature="combined", n_steps=16)
        rep = GASFRepresentation(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        evaluator = StructuralEvaluator(representation_type="gasf")
        report = evaluator.evaluate(real, real)
        summary = report.summary()

        print(f"\n[StructuralEvaluator | gasf | real vs real] {summary}")

        assert "valid_sample_rate" in summary
        assert summary["valid_sample_rate"] > 0.9

    def test_nprint_metrics_present(self):
        """nprint debe devolver las tres métricas características."""
        flows = make_dataset(N_FLOWS)
        cfg = NprintConfig(max_packets=10)
        rep = NprintRepresentation(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        # Sintético continuo (salida de difusión sin binarizar)
        synth = torch.sigmoid(torch.randn_like(real))

        evaluator = StructuralEvaluator(
            representation_type="nprint",
            binary_threshold=0.1,
        )
        report = evaluator.evaluate(real, synth)
        summary = report.summary()

        print(f"\n[StructuralEvaluator | nprint] {summary}")

        assert "non_binary_field_rate" in summary
        assert "valid_sample_rate" in summary
        assert "binarization_confidence" in summary
        assert 0.0 <= summary["non_binary_field_rate"] <= 1.0
        assert 0.0 <= summary["valid_sample_rate"] <= 1.0
        assert 0.0 <= summary["binarization_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Test 3: TSTREvaluator — protocolo TSTR + baseline TRTR
# ---------------------------------------------------------------------------

class TestTSTREvaluator:

    def test_tstr_basic(self):
        """TSTR básico sin baseline TRTR."""
        flows = make_dataset(N_FLOWS)
        cfg = GASFConfig(image_size=8, feature="combined", n_steps=8)
        rep = GASFRepresentation(cfg)
        rep.fit(flows[:N_TRAIN])

        real_test = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        synth_train = make_noisy_synth(
            rep.encode_batch(flows[:N_SYNTH]).float()
        )

        y_real = make_labels(N_TEST)
        y_synth = make_labels(N_SYNTH)

        evaluator = TSTREvaluator(max_features=64, include_trtr=False)
        report = evaluator.evaluate(
            real=real_test,
            synthetic=synth_train,
            real_labels=y_real,
            synthetic_labels=y_synth,
        )
        summary = report.summary()

        print(f"\n[TSTREvaluator | GASF | sin TRTR] {summary}")

        assert "tstr_accuracy" in summary
        assert "tstr_f1_macro" in summary
        assert 0.0 <= summary["tstr_accuracy"] <= 1.0
        assert 0.0 <= summary["tstr_f1_macro"] <= 1.0

    def test_tstr_with_trtr_baseline(self):
        """TSTR con baseline TRTR: verifica que accuracy_gap esté presente."""
        flows = make_dataset(N_FLOWS)
        cfg = GASFConfig(image_size=8, feature="combined", n_steps=8)
        rep = GASFRepresentation(cfg)
        rep.fit(flows[:N_TRAIN])

        real_train = rep.encode_batch(flows[:N_TRAIN]).float()
        real_test = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        synth_train = make_noisy_synth(real_train[:N_SYNTH])

        y_train = make_labels(N_TRAIN)
        y_test = make_labels(N_TEST)
        y_synth = make_labels(N_SYNTH)

        evaluator = TSTREvaluator(max_features=64, include_trtr=True)
        report = evaluator.evaluate(
            real=real_test,
            synthetic=synth_train,
            real_labels=y_test,
            synthetic_labels=y_synth,
            real_train=real_train,
            real_train_labels=y_train,
        )
        summary = report.summary()

        print(f"\n[TSTREvaluator | GASF | con TRTR] {summary}")

        assert "tstr_accuracy" in summary
        assert "trtr_accuracy" in summary
        assert "accuracy_gap" in summary
        assert "f1_gap" in summary

        # El gap puede ser negativo (sintético mejor que real en datos pequeños),
        # pero debe ser un número válido
        assert not np.isnan(summary["accuracy_gap"])

    def test_tstr_without_labels_returns_nan(self):
        """Sin etiquetas, TSTR debe devolver NaN con metadato de error."""
        flows = make_dataset(40)
        real = torch.randn(10, 16)
        synth = torch.randn(10, 16)

        evaluator = TSTREvaluator()
        report = evaluator.evaluate(real, synth)  # sin labels

        result = report.get("tstr_accuracy")
        assert result is not None
        assert np.isnan(result.value)
        assert "error" in result.metadata


# ---------------------------------------------------------------------------
# Test 4: EvaluationSuite — integración completa R×M
# ---------------------------------------------------------------------------

class TestEvaluationSuite:

    def _build_suite(self, representation_type: str, vocab_size=None) -> EvaluationSuite:
        evaluators = [
            StatisticalEvaluator(n_features_cap=64, bins=32),
            StructuralEvaluator(
                representation_type=representation_type,
                vocab_size=vocab_size,
            ),
            TSTREvaluator(max_features=64, include_trtr=False),
        ]
        return EvaluationSuite(
            evaluators=evaluators,
            representation_name=representation_type,
            model_name="DummyModel",
            verbose=True,
        )

    def test_suite_nprint(self):
        """Suite completa sobre representación nprint."""
        print("\n[EvaluationSuite | nprint × DummyModel]")

        flows = make_dataset(N_FLOWS)
        cfg = NprintConfig(max_packets=10)
        rep = NprintRepresentation(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        synth = torch.sigmoid(torch.randn_like(real))

        y_real = make_labels(N_TEST)
        y_synth = make_labels(N_TEST)

        suite = self._build_suite("nprint")
        result: SuiteResult = suite.run(
            real, synth,
            real_labels=y_real,
            synthetic_labels=y_synth,
        )

        assert isinstance(result, SuiteResult)
        assert len(result.reports) == 3

        summary = result.summary()
        print(f"\nResumen final:\n{summary}")

        # Verificar que hay métricas de los tres evaluadores
        keys = list(summary.keys())
        assert any("StatisticalEvaluator" in k for k in keys)
        assert any("StructuralEvaluator" in k for k in keys)
        assert any("TSTREvaluator" in k for k in keys)

        # Verificar que ningún valor es infinito
        for k, v in summary.items():
            assert not np.isinf(v), f"Métrica {k} es infinita."

    def test_suite_gasf(self):
        """Suite completa sobre representación GASF."""
        print("\n[EvaluationSuite | gasf x DummyModel]")

        flows = make_dataset(N_FLOWS)
        cfg = GASFConfig(image_size=8, feature="combined", n_steps=8)
        rep = GASFRepresentation(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        synth = make_noisy_synth(real, noise_scale=0.5)

        y_real = make_labels(N_TEST)
        y_synth = make_labels(N_TEST)

        suite = self._build_suite("gasf")
        result = suite.run(real, synth, real_labels=y_real, synthetic_labels=y_synth)

        summary = result.summary()
        print(f"\nResumen GASF:\n{summary}")

        assert result.get_metric("mean_emd") is not None
        assert result.get_metric("valid_sample_rate") is not None

    def test_suite_sequential(self):
        """Suite completa sobre representación secuencial (FlatTokenizer)."""
        print("\n[EvaluationSuite | sequential x DummyModel]")

        flows = make_dataset(N_FLOWS)
        cfg = SequentialConfig(max_length=64)
        rep = FlatTokenizer(cfg)
        rep.fit(flows[:N_TRAIN])

        real = rep.encode_batch(flows[N_TRAIN:N_TRAIN + N_TEST]).float()
        # Sintético: tokens enteros aleatorios dentro del vocab
        vocab_size = rep.vocab
        synth = torch.randint(0, vocab_size, real.shape).float()

        y_real = make_labels(N_TEST)
        y_synth = make_labels(N_TEST)

        suite = self._build_suite("sequential", vocab_size=vocab_size)
        result = suite.run(real, synth, real_labels=y_real, synthetic_labels=y_synth)

        summary = result.summary()
        print(f"\nResumen Sequential:\n{summary}")

        assert result.get_metric("token_out_of_range_rate") is not None

    def test_suite_to_dataframe(self):
        """Verificar que to_dataframe() devuelve un DataFrame válido."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas no instalado")

        flows = make_dataset(40)
        cfg = GASFConfig(image_size=8, feature="combined", n_steps=8)
        rep = GASFRepresentation(cfg)
        rep.fit(flows[:25])

        real = rep.encode_batch(flows[25:35]).float()
        synth = make_noisy_synth(real)

        suite = EvaluationSuite(
            evaluators=[StatisticalEvaluator(n_features_cap=32)],
            representation_name="gasf",
            model_name="DDPM",
            verbose=False,
        )
        result = suite.run(real, synth)
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "representation" in df.columns
        assert "model" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns
        assert len(df) > 0
        print(f"\n[to_dataframe]\n{df.to_string()}")


# ---------------------------------------------------------------------------
# Runner para debug rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: StatisticalEvaluator")
    print("=" * 60)
    t = TestStatisticalEvaluator()
    t.test_flat_tokenizer()
    t.test_gasf_representation()
    t.test_nprint_representation()

    print("\n" + "=" * 60)
    print("TEST: StructuralEvaluator")
    print("=" * 60)
    t2 = TestStructuralEvaluator()
    t2.test_sequential_valid_data()
    t2.test_sequential_random_noise_degrades()
    t2.test_gasf_real_data_valid()
    t2.test_nprint_metrics_present()

    print("\n" + "=" * 60)
    print("TEST: TSTREvaluator")
    print("=" * 60)
    t3 = TestTSTREvaluator()
    t3.test_tstr_basic()
    t3.test_tstr_with_trtr_baseline()
    t3.test_tstr_without_labels_returns_nan()

    print("\n" + "=" * 60)
    print("TEST: EvaluationSuite")
    print("=" * 60)
    t4 = TestEvaluationSuite()
    t4.test_suite_nprint()
    t4.test_suite_gasf()
    t4.test_suite_sequential()

    print("\n✓ Todos los tests del módulo de evaluación ejecutados correctamente")
"""
src/visualization/__init__.py
==============================
Módulo de visualización de FlowGenX.

Exporta:
    ExperimentPlotter    — Plots de un experimento individual
    ExperimentComparator — Comparativa entre múltiples experimentos

Uso rápido (experimento individual)
------------------------------------
    from src.visualization import ExperimentPlotter

    plotter = ExperimentPlotter(
        plots_dir = "experiments/runs/2025-01-01_ddpm_gasf/plots/",
        exp_name  = "ddpm_gasf_v1",
    )
    plotter.plot_all(
        metrics_csv = Path("experiments/runs/.../metrics/ddpm_gasf_v1_metrics.csv"),
        timing_json = Path("experiments/runs/.../metrics/ddpm_gasf_v1_timing.json"),
        result      = result_obj,
    )

Uso rápido (comparativa)
-------------------------
    from src.visualization import ExperimentComparator

    comp = ExperimentComparator(
        run_dirs = [
            "experiments/runs/2025-01-01_ddpm_gasf",
            "experiments/runs/2025-01-01_gan_flat",
            "experiments/runs/2025-01-01_transformer_semantic",
        ],
        labels = ["DDPM+GAF", "GAN+Flat", "Transformer+Semantic"],
    )
    comp.plot_all(output_dir="experiments/comparison/")
"""

from .plotter    import ExperimentPlotter
from .comparator import ExperimentComparator

__all__ = ["ExperimentPlotter", "ExperimentComparator"]

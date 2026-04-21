"""
scripts/run_experiment.py
==========================
Punto de entrada CLI para FlowGenX.

Subcomandos
-----------
train   : Lanza un experimento completo desde un YAML de configuración.
compare : Genera plots comparativos entre múltiples runs ya finalizadas.

Ejemplos
--------
# Entrenar un experimento
python scripts/run_experiment.py train configs/exp_ddpm_gasf.yaml

# Entrenar con device y data_dir personalizados
python scripts/run_experiment.py train configs/exp_transformer_flat.yaml \
    --data_dir data/pcap/ \
    --device cuda

# Comparar varios runs
python scripts/run_experiment.py compare \
    experiments/runs/2025-01-01_ddpm_gasf \
    experiments/runs/2025-01-01_gan_flat \
    experiments/runs/2025-01-01_transformer_semantic \
    --labels "DDPM+GAF" "GAN+Flat" "Transformer+Semantic" \
    --output_dir experiments/comparison/

# Modo legado (compatibilidad: primer arg es YAML sin subcomando)
python scripts/run_experiment.py configs/exp_ddpm_gasf.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_train(args: argparse.Namespace) -> None:
    """Ejecuta el pipeline completo: train → reconstruct → evaluate → plot."""
    from src.training import run_experiment

    result = run_experiment(
        yaml_path = args.config,
        data_dir  = args.data_dir,
        device    = args.device,
    )

    print("\n" + "═" * 60)
    print(f"  Experimento : {result.experiment_name}")
    print(f"  Modelo      : {result.model}")
    print(f"  Representa. : {result.representation}")
    print(f"  Mejor época : {result.best_epoch}")
    print(f"  Mejor {result.best_metric_name:12s}: {result.best_metric_value:.4f}")
    print(f"  Tiempo total: {result.training_time_s:.1f}s")
    print(f"  Run dir     : {result.run_dir}")
    print("═" * 60)

    if result.eval_metrics:
        print("\n  Métricas de evaluación:")
        for group, vals in result.eval_metrics.items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    try:
                        print(f"    {group}.{k:<30} {float(v):.4f}")
                    except (TypeError, ValueError):
                        pass

    if result.reconstruction_paths:
        print("\n  Tráfico reconstruido:")
        for p in result.reconstruction_paths:
            print(f"    {p}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Genera plots comparativos entre múltiples runs ya finalizadas."""
    from src.visualization import ExperimentComparator

    run_dirs = args.run_dirs
    labels   = args.labels if args.labels else None
    out_dir  = args.output_dir

    if labels and len(labels) != len(run_dirs):
        print(f"[ERROR] --labels debe tener el mismo número de elementos que run_dirs "
              f"({len(labels)} vs {len(run_dirs)}).", file=sys.stderr)
        sys.exit(1)

    print(f"Comparando {len(run_dirs)} experimentos → {out_dir}")
    comp = ExperimentComparator(
        run_dirs  = run_dirs,
        labels    = labels,
        dpi       = args.dpi,
        also_pdf  = args.pdf,
    )
    saved = comp.plot_all(output_dir=out_dir)
    print(f"\nGenerados {len(saved)} plots:")
    for p in saved:
        print(f"  {p}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "run_experiment",
        description = "FlowGenX — Orquestador de experimentos de tráfico sintético",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = __doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="<subcomando>")

    # train
    p_train = sub.add_parser("train", help="Lanzar un experimento.")
    p_train.add_argument("config", type=str,
                         help="Ruta al YAML de configuración del experimento.")
    p_train.add_argument("--data_dir", type=str, default=None,
                         help="Sobreescribe data.data_dir del YAML.")
    p_train.add_argument("--device", type=str, default=None,
                         choices=["cpu", "cuda", "mps"],
                         help="Dispositivo de cómputo (default: auto).")

    # compare
    p_cmp = sub.add_parser("compare", help="Comparar múltiples runs.")
    p_cmp.add_argument("run_dirs", nargs="+",
                       help="Directorios de run a comparar.")
    p_cmp.add_argument("--labels", nargs="*", default=None,
                       help="Etiquetas para cada run (mismo orden).")
    p_cmp.add_argument("--output_dir", type=str,
                       default="experiments/comparison/",
                       help="Directorio de salida (default: experiments/comparison/).")
    p_cmp.add_argument("--dpi", type=int, default=300,
                       help="Resolución en DPI (default: 300).")
    p_cmp.add_argument("--pdf", action="store_true",
                       help="Guardar también versión PDF.")

    return parser


def _is_legacy_call(argv: list) -> bool:
    if not argv:
        return False
    first = argv[0]
    return (
        first not in ("train", "compare", "-h", "--help")
        and (first.endswith(".yaml") or first.endswith(".yml") or Path(first).exists())
    )


def main() -> None:
    argv = sys.argv[1:]

    if _is_legacy_call(argv):
        parser = argparse.ArgumentParser(description="Run experiment from YAML config")
        parser.add_argument("config", type=str)
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--device",   type=str, default=None)
        args = parser.parse_args(argv)
        args.command = "train"
        cmd_train(args)
        return

    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
src/visualization/comparator.py
================================
Comparación visual entre múltiples experimentos de FlowGenX.

Uso típico
----------
    from src.visualization import ExperimentComparator

    comp = ExperimentComparator(
        run_dirs=[
            "experiments/runs/2025-01-01_12-00-00_ddpm_gasf",
            "experiments/runs/2025-01-01_13-00-00_gan_flat",
            "experiments/runs/2025-01-01_14-00-00_transformer_semantic",
        ]
    )
    comp.plot_all(output_dir="experiments/comparison/")

Plots generados
---------------
- comparison_loss.png       : curvas de loss de todos los experimentos
- comparison_timing.png     : tiempo total y por época entre experimentos
- comparison_eval.png       : métricas de evaluación lado a lado
- comparison_summary.png    : tabla resumen con ranking por métrica
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Helpers de carga
# ---------------------------------------------------------------------------

def _load_result_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "result.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _load_metrics_csv(run_dir: Path, exp_name: str) -> List[Dict[str, str]]:
    p = run_dir / "metrics" / f"{exp_name}_metrics.csv"
    if not p.exists():
        return []
    try:
        with open(p, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _load_timing_json(run_dir: Path, exp_name: str) -> Dict[str, Any]:
    p = run_dir / "metrics" / f"{exp_name}_timing.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Paleta y estilo
# ---------------------------------------------------------------------------

_COLORS = [
    "#2563EB", "#DC2626", "#059669", "#7C3AED",
    "#F59E0B", "#06B6D4", "#84CC16", "#EC4899",
]

def _apply_style() -> None:
    try:
        import matplotlib as mpl
        mpl.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "axes.grid":        True,
            "grid.alpha":       0.3,
            "grid.linestyle":   "--",
            "axes.spines.top":  False,
            "axes.spines.right":False,
            "font.size":        11,
            "axes.titlesize":   13,
            "legend.fontsize":  10,
            "lines.linewidth":  2.0,
        })
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# ExperimentComparator
# ---------------------------------------------------------------------------

class ExperimentComparator:
    """
    Carga múltiples runs de FlowGenX y genera plots comparativos.

    Parameters
    ----------
    run_dirs  : Lista de directorios de run (cada uno con result.json).
    labels    : Etiquetas cortas para cada experimento (None = usar exp_name).
    dpi       : Resolución de salida (300 para publicación).
    also_pdf  : Si True, guarda también versión PDF.
    """

    def __init__(
        self,
        run_dirs: List[str | Path],
        labels:   Optional[List[str]] = None,
        dpi:      int  = 300,
        also_pdf: bool = False,
    ) -> None:
        self.run_dirs = [Path(d) for d in run_dirs]
        self.dpi      = dpi
        self.also_pdf = also_pdf

        # Cargar datos de cada run
        self._results: List[Dict[str, Any]] = []
        self._metrics: List[List[Dict[str, str]]] = []
        self._timings: List[Dict[str, Any]] = []
        self._labels:  List[str] = []

        for i, rd in enumerate(self.run_dirs):
            result = _load_result_json(rd)
            if result is None:
                continue
            exp_name = result.get("experiment_name", rd.name)
            label    = labels[i] if labels and i < len(labels) else exp_name

            self._results.append(result)
            self._metrics.append(_load_metrics_csv(rd, exp_name))
            self._timings.append(_load_timing_json(rd, exp_name))
            self._labels.append(label)

        _apply_style()

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def plot_all(self, output_dir: str | Path = ".") -> List[Path]:
        """Genera todos los plots comparativos en ``output_dir``."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []
        if not self._results:
            return saved

        for fn in (
            self.plot_loss_comparison,
            self.plot_timing_comparison,
            self.plot_pipeline_timing_comparison,
            self.plot_eval_comparison,
            self.plot_summary_table,
        ):
            p = fn(out)
            if p:
                saved.append(p)

        return saved

    # ------------------------------------------------------------------
    # Plot 1: Curvas de loss comparadas
    # ------------------------------------------------------------------

    def plot_loss_comparison(self, out: Path) -> Optional[Path]:
        """
        Una línea por experimento mostrando train_loss y val_loss
        en paneles separados.
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            for i, (metrics, label) in enumerate(zip(self._metrics, self._labels)):
                if not metrics:
                    continue
                color  = _COLORS[i % len(_COLORS)]
                epochs = [int(r["epoch"]) for r in metrics if r.get("epoch")]

                train_loss = [_safe_float(r.get("train_loss")) for r in metrics]
                valid_pairs = [(e, v) for e, v in zip(epochs, train_loss) if v is not None]
                if valid_pairs:
                    ep, tl = zip(*valid_pairs)
                    ax1.plot(ep, tl, color=color, label=label, zorder=3)

                val_loss = [_safe_float(r.get("val_loss")) for r in metrics]
                valid_pairs = [(e, v) for e, v in zip(epochs, val_loss) if v is not None]
                if valid_pairs:
                    ep, vl = zip(*valid_pairs)
                    ax2.plot(ep, vl, color=color, label=label, zorder=3)

            ax1.set_title("Train Loss — Comparativa")
            ax1.set_xlabel("Época")
            ax1.set_ylabel("Loss")
            ax1.legend()

            ax2.set_title("Val Loss — Comparativa")
            ax2.set_xlabel("Época")
            ax2.set_ylabel("Loss")
            ax2.legend()

            fig.suptitle("FlowGenX — Comparativa de curvas de loss",
                         fontweight="bold")
            fig.tight_layout()
            return self._save(fig, out, "comparison_loss")
        except Exception as e:
            _warn(f"plot_loss_comparison: {e}")
            return None

    # ------------------------------------------------------------------
    # Plot 2: Timing comparado
    # ------------------------------------------------------------------

    def plot_timing_comparison(self, out: Path) -> Optional[Path]:
        """
        Panel izquierdo: tiempo total de entrenamiento por experimento.
        Panel derecho: distribución de tiempos por época (boxplot).
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            totals  = [t.get("total_training_s", 0) for t in self._timings]
            means   = [t.get("mean_epoch_s", 0)     for t in self._timings]
            ep_data = [t.get("epoch_times", [])      for t in self._timings]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # ── Barras: tiempo total ─────────────────────────────────────
            colors = [_COLORS[i % len(_COLORS)] for i in range(len(self._labels))]
            x = np.arange(len(self._labels))
            bars = ax1.bar(x, totals, color=colors, alpha=0.85, zorder=3)
            ax1.set_xticks(x)
            ax1.set_xticklabels(self._labels, rotation=20, ha="right")
            ax1.set_ylabel("Segundos")
            ax1.set_title("Tiempo total de entrenamiento")
            for bar, val in zip(bars, totals):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(totals, default=1) * 0.01,
                    f"{val:.0f}s",
                    ha="center", va="bottom", fontsize=9,
                )

            # ── Boxplot: distribución por época ──────────────────────────
            valid = [(lbl, ep) for lbl, ep in zip(self._labels, ep_data) if ep]
            if valid:
                lbls, data = zip(*valid)
                bp = ax2.boxplot(
                    data,
                    labels    = lbls,
                    patch_artist = True,
                    medianprops  = dict(color="white", linewidth=2),
                )
                for i, patch in enumerate(bp["boxes"]):
                    patch.set_facecolor(_COLORS[i % len(_COLORS)])
                    patch.set_alpha(0.75)
                ax2.set_ylabel("Segundos / época")
                ax2.set_title("Distribución de tiempo por época")
                ax2.tick_params(axis="x", rotation=20)

            fig.suptitle("FlowGenX — Comparativa de tiempos", fontweight="bold")
            fig.tight_layout()
            return self._save(fig, out, "comparison_timing")
        except Exception as e:
            _warn(f"plot_timing_comparison: {e}")
            return None

    # ------------------------------------------------------------------
    # Plot 3: Métricas de evaluación comparadas
    # ------------------------------------------------------------------

    def plot_eval_comparison(self, out: Path) -> Optional[Path]:
        """
        Barras agrupadas por métrica de evaluación.
        Solo incluye métricas numéricas presentes en al menos un run.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Recopilar todas las métricas flat
            all_flat: List[Dict[str, float]] = []
            for result in self._results:
                flat: Dict[str, float] = {}
                for group, vals in result.get("eval_metrics", {}).items():
                    if isinstance(vals, dict):
                        for k, v in vals.items():
                            try:
                                flat[f"{group}.{k}"] = float(v)
                            except (TypeError, ValueError):
                                pass
                all_flat.append(flat)

            # Métricas comunes a todos los experimentos
            all_keys = sorted({k for d in all_flat for k in d})
            if not all_keys:
                return None

            n_exp = len(self._labels)
            n_met = len(all_keys)
            x     = np.arange(n_met)
            width = 0.8 / max(n_exp, 1)

            fig, ax = plt.subplots(figsize=(max(10, n_met * 1.2), 6))

            for i, (label, flat) in enumerate(zip(self._labels, all_flat)):
                vals = [flat.get(k, float("nan")) for k in all_keys]
                offset = (i - n_exp / 2 + 0.5) * width
                ax.bar(x + offset, vals, width=width * 0.9,
                       color=_COLORS[i % len(_COLORS)],
                       label=label, alpha=0.85, zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels(all_keys, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Valor de métrica")
            ax.set_title("FlowGenX — Comparativa de métricas de evaluación")
            ax.legend()

            fig.tight_layout()
            return self._save(fig, out, "comparison_eval")
        except Exception as e:
            _warn(f"plot_eval_comparison: {e}")
            return None

    # ------------------------------------------------------------------
    # Plot 4: Tabla resumen
    # ------------------------------------------------------------------

    def plot_summary_table(self, out: Path) -> Optional[Path]:
        """
        Tabla visual con las métricas clave de todos los experimentos,
        con resaltado de mejor valor por columna.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Columnas fijas del resumen
            COLS = [
                ("Modelo",      lambda r: r.get("model", "—")),
                ("Representa.", lambda r: r.get("representation", "—")),
                ("Best epoch",  lambda r: str(r.get("best_epoch", "—"))),
                ("Best metric", lambda r: f"{r.get('best_metric_value', float('nan')):.4f}"),
                ("Total épocas",lambda r: str(r.get("total_epochs_ran", "—"))),
                ("Tiempo (s)",  lambda r: f"{r.get('training_time_s', 0):.1f}"),
                ("N params",    lambda r: f"{r.get('extra', {}).get('n_params', 0):,}"),
            ]

            col_headers = [c[0] for c in COLS]
            rows = []
            for result, label in zip(self._results, self._labels):
                row = [label] + [fn(result) for _, fn in COLS]
                rows.append(row)

            if not rows:
                return None

            all_headers = ["Experimento"] + col_headers
            n_rows = len(rows)
            n_cols = len(all_headers)

            fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.8), n_rows * 0.7 + 2))
            ax.axis("off")

            table = ax.table(
                cellText  = rows,
                colLabels = all_headers,
                cellLoc   = "center",
                loc       = "center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.auto_set_column_width(list(range(n_cols)))

            # Estilo: cabecera gris oscuro
            for j in range(n_cols):
                table[0, j].set_facecolor("#374151")
                table[0, j].set_text_props(color="white", fontweight="bold")

            # Filas alternas
            for i in range(1, n_rows + 1):
                bg = "#F9FAFB" if i % 2 == 0 else "white"
                for j in range(n_cols):
                    table[i, j].set_facecolor(bg)

            ax.set_title(
                "FlowGenX — Tabla comparativa de experimentos",
                fontsize=13, fontweight="bold", pad=20,
            )

            fig.tight_layout()
            return self._save(fig, out, "comparison_summary")
        except Exception as e:
            _warn(f"plot_summary_table: {e}")
            return None


    # ------------------------------------------------------------------
    # Plot nuevo: Pipeline timing (generacion + reconstruccion)
    # ------------------------------------------------------------------

    def plot_pipeline_timing_comparison(self, out: Path) -> Optional[Path]:
        """
        Barras apiladas mostrando generation_time + reconstruction_time
        por experimento. Permite comparar el coste del pipeline post-train.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            gen_times   = []
            recon_times = []
            labels      = []

            for result, label in zip(self._results, self._labels):
                pt = result.get("pipeline_timing", {})
                gt = pt.get("generation_time_s", result.get("extra", {}).get("generation_time_s", None))
                rt = pt.get("reconstruction_time_s", result.get("extra", {}).get("reconstruction_time_s", None))
                if gt is None and rt is None:
                    continue
                gen_times.append(float(gt or 0))
                recon_times.append(float(rt or 0))
                labels.append(label)

            if not labels:
                return None

            x = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))

            b1 = ax.bar(x, gen_times,   label="Generacion", color=_COLORS[0], alpha=0.85)
            b2 = ax.bar(x, recon_times, label="Reconstruccion", color=_COLORS[1],
                        alpha=0.85, bottom=gen_times)

            # Etiquetas de valor
            for i, (g, r) in enumerate(zip(gen_times, recon_times)):
                total = g + r
                ax.text(
                    i,
                    total + max(gen_times + recon_times, default=1) * 0.01,
                    f"{total:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("Segundos")
            ax.set_title("FlowGenX — Coste del pipeline post-entrenamiento\n"
                         "(generacion + reconstruccion)")
            ax.legend()
            fig.tight_layout()
            return self._save(fig, out, "comparison_pipeline_timing")
        except Exception as e:
            _warn(f"plot_pipeline_timing_comparison: {e}")
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save(self, fig, out: Path, name: str) -> Path:
        import matplotlib.pyplot as plt
        path = out / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if self.also_pdf:
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        return path


def _warn(msg: str) -> None:
    import logging
    logging.getLogger(__name__).warning("[Comparator] %s", msg)


__all__ = ["ExperimentComparator"]

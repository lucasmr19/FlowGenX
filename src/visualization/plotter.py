"""
src/visualization/plotter.py
=============================
Módulo de visualización para un experimento individual de FlowGenX.

Genera gráficas de publicación (300 dpi, estilo limpio) listas para
incluir en un TFG o artículo. Todos los plots se exportan en PNG y,
opcionalmente, en PDF.

Plots disponibles
-----------------
- loss_curves.png          : train_loss y val_* por época
- timing.png               : tiempo por época + acumulado
- eval_metrics.png         : métricas de evaluación en barras/radar
- training_summary.png     : dashboard 4-panel (combinado)

Dependencias
------------
Requiere matplotlib (>=3.5). Si no está instalado, los métodos fallan
silenciosamente salvo que ``strict=True``.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..training.experiment import ExperimentResult


# ---------------------------------------------------------------------------
# Estilo global
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False

def _apply_style() -> None:
    """Aplica un estilo limpio y consistente para publicación."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mpl.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "axes.grid":        True,
            "grid.alpha":       0.3,
            "grid.linestyle":   "--",
            "axes.spines.top":  False,
            "axes.spines.right":False,
            "font.family":      "sans-serif",
            "font.size":        11,
            "axes.titlesize":   13,
            "axes.labelsize":   11,
            "legend.fontsize":  10,
            "xtick.labelsize":  10,
            "ytick.labelsize":  10,
            "lines.linewidth":  2.0,
            "lines.markersize": 5,
        })
        _STYLE_APPLIED = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Paleta de colores consistente
# ---------------------------------------------------------------------------

_PALETTE = {
    "train":      "#2563EB",   # azul
    "val":        "#DC2626",   # rojo
    "time":       "#059669",   # verde
    "accent":     "#7C3AED",   # morado
    "neutral":    "#6B7280",   # gris
    "bar_good":   "#10B981",   # verde claro
    "bar_warn":   "#F59E0B",   # naranja
    "bar_bad":    "#EF4444",   # rojo claro
}

_METRIC_COLORS = [
    "#2563EB", "#DC2626", "#059669", "#7C3AED",
    "#F59E0B", "#06B6D4", "#84CC16", "#EC4899",
]


# ---------------------------------------------------------------------------
# ExperimentPlotter
# ---------------------------------------------------------------------------

class ExperimentPlotter:
    """
    Genera y guarda todos los plots de un experimento.

    Parameters
    ----------
    plots_dir : Directorio donde se guardarán los PNGs.
    exp_name  : Nombre del experimento (usado en títulos y nombres de archivo).
    dpi       : Resolución de salida (300 para publicación).
    also_pdf  : Si True, guarda también versión PDF junto al PNG.
    """

    def __init__(
        self,
        plots_dir: Path,
        exp_name:  str  = "experiment",
        dpi:       int  = 300,
        also_pdf:  bool = False,
    ) -> None:
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name  = exp_name
        self.dpi       = dpi
        self.also_pdf  = also_pdf

        _apply_style()

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def plot_all(
        self,
        metrics_csv:  Optional[Path] = None,
        timing_json:  Optional[Path] = None,
        result:       Optional[Any]  = None,
    ) -> List[Path]:
        """
        Genera todos los plots disponibles y devuelve sus rutas.

        Parameters
        ----------
        metrics_csv : CSV generado por MetricsLoggerCallback.
        timing_json : JSON generado por TimingCallback.
        result      : ExperimentResult con métricas de evaluación.
        """
        saved: List[Path] = []

        metrics = self._load_metrics_csv(metrics_csv)
        timing  = self._load_timing_json(timing_json)

        if metrics:
            p = self.plot_loss_curves(metrics)
            if p:
                saved.append(p)

        if timing:
            p = self.plot_timing(timing)
            if p:
                saved.append(p)

        if result and getattr(result, "eval_metrics", None):
            p = self.plot_eval_metrics(result.eval_metrics)
            if p:
                saved.append(p)

        # Dashboard: solo si tenemos métricas de entrenamiento
        if metrics:
            p = self.plot_training_summary(
                metrics = metrics,
                timing  = timing,
                result  = result,
            )
            if p:
                saved.append(p)

        return saved

    # ------------------------------------------------------------------
    # Plot 1: Curvas de loss
    # ------------------------------------------------------------------

    def plot_loss_curves(self, metrics: List[Dict[str, str]]) -> Optional[Path]:
        """
        Grafica train_loss y todas las métricas val_* por época.

        Genera dos sub-gráficas apiladas si hay métricas val adicionales.
        """
        try:
            import matplotlib.pyplot as plt

            epochs     = [int(r["epoch"]) for r in metrics if r.get("epoch")]
            train_loss = [_safe_float(r.get("train_loss")) for r in metrics]

            # Detectar columnas val_*
            val_keys = sorted({
                k for r in metrics for k in r
                if k.startswith("val_") and k != "val_loss"
            })
            has_val_loss = any(r.get("val_loss") for r in metrics)

            n_panels = 1 + (1 if val_keys else 0)
            fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels),
                                     sharex=True)
            if n_panels == 1:
                axes = [axes]

            # Panel superior: train + val_loss
            ax = axes[0]
            ax.plot(epochs, train_loss, color=_PALETTE["train"],
                    label="train_loss", zorder=3)
            if has_val_loss:
                val_loss = [_safe_float(r.get("val_loss")) for r in metrics]
                ax.plot(epochs, val_loss, color=_PALETTE["val"],
                        label="val_loss", linestyle="--", zorder=3)
            ax.set_ylabel("Loss")
            ax.set_title(f"{self.exp_name} — Curvas de entrenamiento")
            ax.legend()

            # Panel inferior: otras métricas val
            if val_keys and len(axes) > 1:
                ax2 = axes[1]
                for i, key in enumerate(val_keys):
                    vals = [_safe_float(r.get(key)) for r in metrics]
                    ax2.plot(epochs, vals, color=_METRIC_COLORS[i % len(_METRIC_COLORS)],
                             label=key, zorder=3)
                ax2.set_ylabel("Métrica")
                ax2.set_xlabel("Época")
                ax2.legend()
            else:
                axes[-1].set_xlabel("Época")

            fig.tight_layout()
            path = self._save(fig, "loss_curves")
            plt.close(fig)
            return path
        except Exception as e:
            _warn(f"plot_loss_curves falló: {e}")
            return None

    # ------------------------------------------------------------------
    # Plot 2: Timing
    # ------------------------------------------------------------------

    def plot_timing(self, timing: Dict[str, Any]) -> Optional[Path]:
        """
        Dos paneles: tiempo por época (barras) y tiempo acumulado (línea).
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            epoch_times = timing.get("epoch_times", [])
            if not epoch_times:
                return None

            epochs   = list(range(1, len(epoch_times) + 1))
            cumul    = list(np.cumsum(epoch_times))
            mean_t   = timing.get("mean_epoch_s", 0)
            total_t  = timing.get("total_training_s", 0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

            # ── Panel 1: barras por época ────────────────────────────────
            colors = [_PALETTE["bar_warn"] if t > mean_t * 1.2 else _PALETTE["time"]
                      for t in epoch_times]
            ax1.bar(epochs, epoch_times, color=colors, alpha=0.85, zorder=3)
            ax1.axhline(mean_t, color=_PALETTE["neutral"], linestyle="--",
                        linewidth=1.5, label=f"Media: {mean_t:.2f}s")
            ax1.set_ylabel("Segundos / época")
            ax1.set_title(f"{self.exp_name} — Tiempo por época\n"
                          f"Total: {total_t:.1f}s  |  "
                          f"Media: {mean_t:.2f}s  |  "
                          f"Médiana: {timing.get('median_epoch_s', 0):.2f}s")
            ax1.legend()

            # ── Panel 2: tiempo acumulado ────────────────────────────────
            ax2.fill_between(epochs, cumul, alpha=0.2, color=_PALETTE["accent"])
            ax2.plot(epochs, cumul, color=_PALETTE["accent"], zorder=3)
            ax2.set_ylabel("Tiempo acumulado (s)")
            ax2.set_xlabel("Época")

            fig.tight_layout()
            path = self._save(fig, "timing")
            plt.close(fig)
            return path
        except Exception as e:
            _warn(f"plot_timing falló: {e}")
            return None

    # ------------------------------------------------------------------
    # Plot 3: Métricas de evaluación
    # ------------------------------------------------------------------

    def plot_eval_metrics(self, eval_metrics: Dict[str, Any]) -> Optional[Path]:
        """
        Barras horizontales por grupo de evaluador (estadístico, estructural, URS).
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Aplanar métricas: "grupo.metrica" → valor
            flat: Dict[str, float] = {}
            for group, vals in eval_metrics.items():
                if isinstance(vals, dict):
                    for k, v in vals.items():
                        try:
                            flat[f"{group}.{k}"] = float(v)
                        except (TypeError, ValueError):
                            pass

            if not flat:
                return None

            labels = list(flat.keys())
            values = [flat[k] for k in labels]
            n      = len(labels)

            fig, ax = plt.subplots(figsize=(9, max(4, n * 0.55)))
            colors  = [_PALETTE["bar_good"] if v <= 0.5 else _PALETTE["bar_warn"]
                       for v in values]
            bars = ax.barh(labels, values, color=colors, alpha=0.85, zorder=3)

            # Añadir valores al final de cada barra
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center", ha="left", fontsize=9,
                    color=_PALETTE["neutral"],
                )

            ax.set_xlabel("Valor")
            ax.set_title(f"{self.exp_name} — Métricas de evaluación")
            ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
            ax.invert_yaxis()

            fig.tight_layout()
            path = self._save(fig, "eval_metrics")
            plt.close(fig)
            return path
        except Exception as e:
            _warn(f"plot_eval_metrics falló: {e}")
            return None

    # ------------------------------------------------------------------
    # Plot 4: Dashboard de resumen (4 paneles)
    # ------------------------------------------------------------------

    def plot_training_summary(
        self,
        metrics: List[Dict[str, str]],
        timing:  Optional[Dict[str, Any]] = None,
        result:  Optional[Any]            = None,
    ) -> Optional[Path]:
        """
        Dashboard 4-panel:
          ┌──────────────────┬──────────────────┐
          │  loss curves     │  timing barras   │
          ├──────────────────┼──────────────────┤
          │  val_metrics     │  info text card  │
          └──────────────────┴──────────────────┘
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np

            fig = plt.figure(figsize=(16, 10))
            gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])   # loss curves
            ax2 = fig.add_subplot(gs[0, 1])   # timing
            ax3 = fig.add_subplot(gs[1, 0])   # val metrics
            ax4 = fig.add_subplot(gs[1, 1])   # info card

            epochs     = [int(r["epoch"]) for r in metrics if r.get("epoch")]
            train_loss = [_safe_float(r.get("train_loss")) for r in metrics]

            # ── Panel 1: loss ────────────────────────────────────────────
            ax1.plot(epochs, train_loss, color=_PALETTE["train"], label="train_loss")
            if any(r.get("val_loss") for r in metrics):
                val_loss = [_safe_float(r.get("val_loss")) for r in metrics]
                ax1.plot(epochs, val_loss, color=_PALETTE["val"],
                         linestyle="--", label="val_loss")
            ax1.set_title("Loss de entrenamiento")
            ax1.set_xlabel("Época")
            ax1.set_ylabel("Loss")
            ax1.legend()

            # ── Panel 2: timing ──────────────────────────────────────────
            if timing and timing.get("epoch_times"):
                et = timing["epoch_times"]
                ep = list(range(1, len(et) + 1))
                mean_t = timing.get("mean_epoch_s", 0)
                colors = [_PALETTE["bar_warn"] if t > mean_t * 1.2
                          else _PALETTE["time"] for t in et]
                ax2.bar(ep, et, color=colors, alpha=0.85)
                ax2.axhline(mean_t, color=_PALETTE["neutral"],
                            linestyle="--", linewidth=1.5,
                            label=f"Media {mean_t:.1f}s")
                ax2.set_title("Tiempo por época (s)")
                ax2.set_xlabel("Época")
                ax2.set_ylabel("Segundos")
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "Timing no disponible",
                         ha="center", va="center", transform=ax2.transAxes)
                ax2.axis("off")

            # ── Panel 3: val metrics evolution ───────────────────────────
            val_keys = sorted({
                k for r in metrics for k in r
                if k.startswith("val_") and _safe_float(r.get(k)) is not None
            })
            if val_keys:
                for i, key in enumerate(val_keys[:6]):
                    vals = [_safe_float(r.get(key)) for r in metrics]
                    ax3.plot(epochs, vals,
                             color=_METRIC_COLORS[i % len(_METRIC_COLORS)],
                             label=key)
                ax3.set_title("Métricas de validación")
                ax3.set_xlabel("Época")
                ax3.legend(fontsize=8)
            else:
                ax3.text(0.5, 0.5, "Sin métricas de validación",
                         ha="center", va="center", transform=ax3.transAxes)
                ax3.axis("off")

            # ── Panel 4: info card ───────────────────────────────────────
            ax4.axis("off")
            if result is not None:
                lines_info = [
                    f"Experimento:   {getattr(result, 'experiment_name', '—')}",
                    f"Representación:{getattr(result, 'representation', '—')}",
                    f"Modelo:        {getattr(result, 'model', '—')}",
                    f"Seed:          {getattr(result, 'seed', '—')}",
                    "",
                    f"Mejor época:   {getattr(result, 'best_epoch', '—')}",
                    f"Mejor {getattr(result, 'best_metric_name', 'métrica')}:   "
                    f"{getattr(result, 'best_metric_value', float('nan')):.4f}",
                    f"Épocas totales:{getattr(result, 'total_epochs_ran', '—')}",
                    f"Tiempo total:  {getattr(result, 'training_time_s', 0):.1f}s",
                ]
                pt = getattr(result, "pipeline_timing", {})
                if pt:
                    lines_info += [
                        "",
                        "── Pipeline timing ──",
                        f"  Gen:    {pt.get('generation_time_s', 0):.2f}s",
                        f"  Recon:  {pt.get('reconstruction_time_s', 0):.2f}s",
                        f"  Total:  {pt.get('total_pipeline_time_s', 0):.2f}s",
                    ]
                lines_info += [
                    "",
                    f"N params:      "
                    f"{getattr(result, 'extra', {}).get('n_params', 0):,}",
                ]
                eval_m = getattr(result, "eval_metrics", {})
                if eval_m:
                    lines_info.append("")
                    lines_info.append("── Evaluación ──")
                    for group, vals in eval_m.items():
                        if isinstance(vals, dict):
                            for k, v in list(vals.items())[:3]:
                                try:
                                    lines_info.append(f"  {group}.{k}: {float(v):.4f}")
                                except (TypeError, ValueError):
                                    pass

                text = "\n".join(lines_info)
                ax4.text(
                    0.05, 0.95, text,
                    transform    = ax4.transAxes,
                    va           = "top",
                    ha           = "left",
                    fontsize     = 9,
                    fontfamily   = "monospace",
                    bbox         = dict(boxstyle="round,pad=0.5",
                                        facecolor="#F9FAFB",
                                        edgecolor="#D1D5DB",
                                        linewidth=1),
                )
                ax4.set_title("Resumen del experimento")

            fig.suptitle(
                f"FlowGenX — {self.exp_name}",
                fontsize=15, fontweight="bold", y=1.01,
            )

            path = self._save(fig, "training_summary")
            plt.close(fig)
            return path
        except Exception as e:
            _warn(f"plot_training_summary falló: {e}")
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save(self, fig, name: str) -> Path:
        """Guarda figura en PNG (y PDF si also_pdf=True)."""
        path = self.plots_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if self.also_pdf:
            fig.savefig(self.plots_dir / f"{name}.pdf", bbox_inches="tight")
        return path

    @staticmethod
    def _load_metrics_csv(path: Optional[Path]) -> List[Dict[str, str]]:
        if path is None or not Path(path).exists():
            return []
        try:
            with open(path, newline="") as f:
                return list(csv.DictReader(f))
        except Exception:
            return []

    @staticmethod
    def _load_timing_json(path: Optional[Path]) -> Dict[str, Any]:
        if path is None or not Path(path).exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _warn(msg: str) -> None:
    import logging
    logging.getLogger(__name__).warning("[Plotter] %s", msg)


__all__ = ["ExperimentPlotter"]

"""
training/experiment.py
=======================
Orquestador de alto nivel: YAML → datos → representación → modelo → entrenamiento → artefactos.

Pipeline post-entrenamiento
---------------------------
  generate()  →  samples (generados UNA sola vez, cacheados)
      ↓
  reconstruction(samples)  →  reconstructed_flows  +  .pcap
      ↓
  evaluation(samples, reconstructed_flows)  →  métricas completas

Evaluadores activos (orden de prioridad)
----------------------------------------
  PRIMARY   StatisticalEvaluator       (EMD, JS, Pearson)
  PRIMARY   URSEvaluator               (FID, MMD)
  CRITICAL  ConditionalEvaluator       (accuracy, f1_macro sobre labels condicionantes)
  SECONDARY StructuralEvaluator        (validez de la representacion)
  SECONDARY TrafficStructuralEvaluator (nivel paquete/flujo — raw o reconstruido)
  TERTIARY  TSTREvaluator / TRTREvaluator / AnomalyDetectionTask (utilidad)
"""

from __future__ import annotations

import json
import time
import yaml
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..utils.config_loader import load_yaml
from ..utils.builders import build_model_from_dict, build_representation_from_dict
from ..datamodules import TrafficDataModule, build_datamodule_from_dir
from ..models_ml.base import GenerativeModel
from ..representations.base import TrafficRepresentation
from ..utils.logger_config import LOGGER

from .callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    EMACallback,
    MetricsLoggerCallback,
    TensorBoardCallback,
    TimingCallback,
    TrainerCallback,
    WandbCallback,
)
from .config import TrainingConfig
from .trainer import Trainer


# ---------------------------------------------------------------------------
# Resultado del experimento
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    Resumen completo de un experimento de FlowGenX.
    Serializable a JSON vía to_dict() / save_json().
    """
    run_dir:              str
    experiment_name:      str
    representation:       str
    model:                str
    best_epoch:           int
    best_metric_name:     str
    best_metric_value:    float
    total_epochs_ran:     int
    training_time_s:      float
    checkpoint_path:      Optional[str]
    val_metrics:          Dict[str, float] = field(default_factory=dict)
    extra:                Dict[str, Any]   = field(default_factory=dict)
    eval_metrics:         Dict[str, Any]   = field(default_factory=dict)
    reconstruction_paths: List[str]        = field(default_factory=list)
    epoch_times:          List[float]      = field(default_factory=list)
    seed:                 int              = 42
    pipeline_timing:      Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: "str | Path") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        LOGGER.info("[ExperimentResult] Guardado en %s", path)


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Orquestador de experimento completo para FlowGenX.

    Parameters
    ----------
    yaml_path       : Ruta al YAML de configuracion.
    data_dir        : Directorio raiz con .pcap por clase.
    experiment_name : Nombre del experimento.
    extra_callbacks : Callbacks adicionales.
    device          : Dispositivo ("cpu"|"cuda"|"mps"). None=auto.
    """

    def __init__(
        self,
        yaml_path:        str,
        data_dir:         Optional[str] = None,
        experiment_name:  Optional[str] = None,
        extra_callbacks:  Optional[List[TrainerCallback]] = None,
        device:           Optional[str] = None,
    ) -> None:
        self._raw_cfg   = load_yaml(yaml_path)
        self._yaml_stem = Path(yaml_path).stem
        self._data_dir  = data_dir
        self._exp_name  = (
            experiment_name
            or self._raw_cfg.get("experiment_name")
            or self._yaml_stem
        )
        self._extra_callbacks = extra_callbacks or []
        self._device          = device

    @classmethod
    def from_dict(
        cls,
        config_dict:     Dict[str, Any],
        data_dir:        Optional[str] = None,
        experiment_name: str = "exp",
        extra_callbacks: Optional[List[TrainerCallback]] = None,
        device:          Optional[str] = None,
    ) -> "ExperimentRunner":
        """Crea un ExperimentRunner desde un dict."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            tmp_path = f.name
        runner = cls(tmp_path, data_dir, experiment_name, extra_callbacks, device)
        runner._raw_cfg = config_dict
        return runner

    # ------------------------------------------------------------------
    # Punto de entrada
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResult:
        """
        Ejecuta el experimento completo.

        Flujo de datos post-entrenamiento:
            generate() -> samples
                -> reconstruction(samples) -> flows + .pcap
                    -> evaluation(samples, flows) -> metricas
        """
        # Directorios de salida
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir   = Path("experiments/runs") / f"{timestamp}_{self._exp_name}"
        metrics_dir = run_dir / "metrics"
        plots_dir   = run_dir / "plots"
        tb_dir      = run_dir / "logs"      # creado ANTES de callbacks
        recon_dir   = run_dir / "reconstructed"

        for d in (run_dir, metrics_dir, plots_dir, tb_dir, recon_dir):
            d.mkdir(parents=True, exist_ok=True)

        t_start = time.time()
        cfg_raw = self._raw_cfg

        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg_raw, f)

        # 1. Configs
        train_cfg = TrainingConfig.from_dict(cfg_raw.get("training", {}))
        train_cfg.checkpoint_dir = str(run_dir / "checkpoints")
        data_cfg  = cfg_raw.get("data", {})
        rep_cfg   = cfg_raw.get("representation", {})
        model_cfg = cfg_raw.get("model", {})
        recon_cfg = cfg_raw.get("reconstruction", {})
        eval_cfg  = cfg_raw.get("evaluation", {})
        seed      = getattr(train_cfg, "seed", 42)

        # 2. Dispositivo
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg.setdefault("device", device)

        # 3. Representacion
        LOGGER.info("[ExperimentRunner] Representacion: %s", rep_cfg.get("representation_type"))
        rep: TrafficRepresentation = build_representation_from_dict(rep_cfg)

        # 4. Datos
        data_dir = Path(self._data_dir or data_cfg.get("data_dir", "data/pcap/"))
        LOGGER.info("[ExperimentRunner] Datos desde %s", data_dir)
        dm: TrafficDataModule = build_datamodule_from_dir(
            root_path         = data_dir,
            representation    = rep,
            aggregator        = rep.get_default_aggregator(),
            max_packets       = data_cfg.get("max_packets", None),
            protocols         = data_cfg.get("protocols", None),
            max_payload_bytes = data_cfg.get("max_payload_bytes", 20),
            train_ratio       = train_cfg.train_ratio,
            val_ratio         = train_cfg.val_ratio,
            test_ratio        = train_cfg.test_ratio,
            batch_size        = train_cfg.batch_size,
            num_workers       = train_cfg.num_workers,
            seed              = seed,
        )
        dm.setup()
        summary = dm.summary()
        LOGGER.info(
            "[ExperimentRunner] DataModule: train=%d val=%d test=%d",
            summary["train_samples"], summary["val_samples"], summary["test_samples"],
        )

        # 5. vocab_size
        vocab_size = getattr(rep, "vocab_size", None)
        if vocab_size is not None and "vocab_size" not in model_cfg:
            model_cfg["vocab_size"] = vocab_size
            LOGGER.info("[ExperimentRunner] vocab_size=%d sincronizado.", vocab_size)

        # 6. Modelo
        LOGGER.info("[ExperimentRunner] Modelo: %s", model_cfg.get("model_type"))
        model: GenerativeModel = build_model_from_dict(model_cfg)
        model.build()
        params = model.get_num_parameters()
        LOGGER.info("[ExperimentRunner] Parametros: %s", f"{params.get('total', 0):,}")

        # 7. Callbacks (tb_dir ya existe)
        callbacks = self._build_callbacks(train_cfg, model, rep, run_dir, tb_dir)
        callbacks.extend(self._extra_callbacks)

        # 8. Entrenamiento
        trainer = Trainer(model=model, config=train_cfg, callbacks=callbacks)
        LOGGER.info(
            "[ExperimentRunner] Iniciando '%s'  (%s + %s)",
            self._exp_name,
            rep_cfg.get("representation_type"),
            model_cfg.get("model_type"),
        )
        state   = trainer.fit(dm.train_dataloader(), dm.val_dataloader())
        elapsed = time.time() - t_start

        result = ExperimentResult(
            run_dir           = str(run_dir),
            experiment_name   = self._exp_name,
            representation    = rep_cfg.get("representation_type", "unknown"),
            model             = model_cfg.get("model_type", "unknown"),
            best_epoch        = state.best_epoch,
            best_metric_name  = train_cfg.checkpoint_metric,
            best_metric_value = state.best_metric,
            total_epochs_ran  = state.epoch,
            training_time_s   = round(elapsed, 2),
            checkpoint_path   = state.checkpoint_path,
            val_metrics       = state.val_metrics,
            epoch_times       = state.epoch_times,
            seed              = seed,
            extra             = {"n_params": params.get("total", 0)},
        )

        # 9. Generar muestras UNA sola vez
        n_samples      = recon_cfg.get("n_samples", eval_cfg.get("n_synthetic", 32))
        synth_samples, synth_labels, gen_time = self._generate_samples(
            model=model, model_cfg=model_cfg, n_samples=n_samples, seed=seed,
        )

        # 10. Reconstruccion (usa samples cacheados)
        recon_paths, synth_flows, recon_time = self._run_reconstruction(
            rep=rep, rep_cfg=rep_cfg, model_cfg=model_cfg,
            recon_dir=recon_dir, samples=synth_samples, labels=synth_labels,
        )
        result.reconstruction_paths = recon_paths

        # 11. Evaluacion (usa samples + flows, sin segunda llamada a generate)
        real_flows, real_flows_is_raw = self._get_real_flows(
            dm=dm, rep_cfg=rep_cfg, model_cfg=model_cfg
        )
        result.eval_metrics = self._run_evaluation(
            dm=dm, rep=rep, rep_cfg=rep_cfg, model_cfg=model_cfg, metrics_dir=metrics_dir,
            synth_samples=synth_samples, synth_flows=synth_flows, synth_labels=synth_labels,
            real_flows=real_flows, real_flows_is_raw=real_flows_is_raw, seed=seed,
        )

        # Pipeline timing
        pipeline_timing = {
            "generation_time_s":     round(gen_time,   3),
            "reconstruction_time_s": round(recon_time, 3),
            "total_pipeline_time_s": round(gen_time + recon_time, 3),
        }
        result.pipeline_timing = pipeline_timing
        result.extra.update(pipeline_timing)
        with open(metrics_dir / f"{self._exp_name}_pipeline_timing.json", "w") as f:
            json.dump(pipeline_timing, f, indent=2)

        # 12. Visualizacion
        self._run_visualization(run_dir=run_dir, metrics_dir=metrics_dir,
                                plots_dir=plots_dir, result=result)

        # 13. Guardar resultado
        result.save_json(run_dir / "result.json")
        LOGGER.info(
            "[ExperimentRunner] Experimento '%s' completado en %.1fs "
            "(gen=%.1fs recon=%.1fs).",
            self._exp_name, elapsed, gen_time, recon_time,
        )
        return result

    # ------------------------------------------------------------------
    # Construccion de callbacks
    # ------------------------------------------------------------------

    def _build_callbacks(
        self,
        train_cfg: TrainingConfig,
        model:     GenerativeModel,
        rep:       TrafficRepresentation,
        run_dir:   Path,
        tb_dir:    Path,
    ) -> List[TrainerCallback]:
        cbs: List[TrainerCallback] = []

        cbs.append(MetricsLoggerCallback(
            log_dir=str(run_dir / "metrics"),
            experiment_name=self._exp_name,
            log_every=train_cfg.log_every,
        ))
        cbs.append(TimingCallback(
            log_dir=str(run_dir / "metrics"),
            experiment_name=self._exp_name,
        ))
        cbs.append(CheckpointCallback(
            checkpoint_dir=str(run_dir / "checkpoints"),
            metric=train_cfg.checkpoint_metric,
            mode=train_cfg.checkpoint_mode,
            model=model,
            representation=rep,
            save_last=train_cfg.save_last,
            save_top_k=train_cfg.save_top_k,
            experiment_name=self._exp_name,
        ))
        if train_cfg.early_stopping:
            cbs.append(EarlyStoppingCallback(
                metric=train_cfg.early_stopping_metric,
                mode=train_cfg.checkpoint_mode,
                patience=train_cfg.early_stopping_patience,
                min_delta=train_cfg.early_stopping_delta,
            ))
        if train_cfg.use_ema:
            cbs.append(EMACallback(
                model=model,
                decay=train_cfg.ema_decay,
                start_epoch=train_cfg.ema_start,
            ))

        # TensorBoard: tb_dir ya existe en disco, sin riesgo de "not a directory"
        if train_cfg.use_tensorboard:
            try:
                cbs.append(TensorBoardCallback(
                    log_dir=str(tb_dir),
                    experiment_name=self._exp_name,
                ))
            except Exception as exc:
                LOGGER.warning("[ExperimentRunner] TensorBoard no disponible: %s", exc)

        if train_cfg.use_wandb:
            try:
                cbs.append(WandbCallback(
                    project=train_cfg.wandb_project,
                    run_name=train_cfg.wandb_run_name or self._exp_name,
                    config_dict=self._raw_cfg,
                    tags=train_cfg.extra_tags,
                ))
            except Exception as exc:
                LOGGER.warning("[ExperimentRunner] wandb no disponible: %s", exc)

        return cbs

    # ------------------------------------------------------------------
    # Fase 9: Generar muestras (UNA sola vez)
    # ------------------------------------------------------------------

    def _generate_samples(
        self,
        model:     GenerativeModel,
        model_cfg: Dict[str, Any],
        n_samples: int = 32,
        seed:      int = 42,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
        """
        Genera n_samples del modelo. Muestreo condicional si num_classes>1.

        Returns
        -------
        (samples, labels, elapsed_seconds)
        """
        LOGGER.info("[ExperimentRunner] Fase 9: Generando %d muestras.", n_samples)
        generate_fn = getattr(model, "generate", None)
        if generate_fn is None:
            LOGGER.warning("[Generate] Modelo sin generate(). Saltando.")
            return None, None, 0.0

        num_classes = model_cfg.get("num_classes", 0)
        model.eval_mode()
        t0 = time.time()

        try:
            if num_classes and num_classes > 1:
                per_class = max(1, n_samples // num_classes)
                parts_s, parts_y = [], []
                with torch.no_grad():
                    for lid in range(num_classes):
                        y = torch.full((per_class,), lid, dtype=torch.long)
                        try:
                            s = generate_fn(n_samples=per_class, labels=y)
                        except TypeError:
                            s = generate_fn(n_samples=per_class)
                        parts_s.append(s)
                        parts_y.append(y)
                samples = torch.cat(parts_s, dim=0)
                labels  = torch.cat(parts_y, dim=0)
            else:
                with torch.no_grad():
                    samples = generate_fn(n_samples=n_samples)
                labels = None

            elapsed = time.time() - t0
            LOGGER.info(
                "[Generate] %d muestras en %.2fs  shape=%s",
                samples.shape[0], elapsed, tuple(samples.shape),
            )
            return samples, labels, elapsed

        except Exception as exc:
            elapsed = time.time() - t0
            LOGGER.warning("[Generate] Error (no bloqueante): %s", exc, exc_info=True)
            return None, None, elapsed

    # ------------------------------------------------------------------
    # Fase 10: Reconstruccion -> .pcap
    # ------------------------------------------------------------------

    def _run_reconstruction(
        self,
        rep:       TrafficRepresentation,
        rep_cfg:   Dict[str, Any],
        model_cfg: Dict[str, Any],
        recon_dir: Path,
        samples:   Optional[torch.Tensor],
        labels:    Optional[torch.Tensor],
    ) -> Tuple[List[str], List[Any], float]:
        """
        Reconstruye flows a partir de los samples pre-generados.

        Returns
        -------
        (paths, flows, elapsed_seconds)
        """
        LOGGER.info("[ExperimentRunner] Fase 10: Reconstruccion -> .pcap.")
        paths: List[str] = []
        flows: List[Any] = []

        if samples is None:
            LOGGER.warning("[Reconstruction] Sin samples. Saltando.")
            return paths, flows, 0.0

        t0 = time.time()
        try:
            from ..reconstruction.build import build_reconstructor
            from ..reconstruction.serialization import samples_to_pcap

            rec = build_reconstructor(
                rep_name   = rep_cfg.get("representation_type", "unknown"),
                model_name = model_cfg.get("model_type", "unknown"),
                verbose    = False,
            )

            unique_labels = labels.unique().tolist() if labels is not None else [None]

            if len(unique_labels) > 1:
                for lid in unique_labels:
                    lid_int = int(lid)
                    mask = (labels == lid_int)
                    s = samples[mask]
                    y = labels[mask]
                    cls_flows = rec.reconstruct(s, labels=y)
                    flows.extend(cls_flows)
                    out = recon_dir / f"class_{lid_int}.pcap"
                    samples_to_pcap(cls_flows, str(out))
                    paths.append(str(out))
                    LOGGER.info("[Reconstruction] Clase %d -> %d flows -> %s",
                                lid_int, len(cls_flows), out)
            else:
                flows = rec.reconstruct(samples, labels=labels)
                out   = recon_dir / "synthetic.pcap"
                samples_to_pcap(flows, str(out))
                paths.append(str(out))
                LOGGER.info("[Reconstruction] %d flows -> %s", len(flows), out)

        except Exception as exc:
            LOGGER.warning("[Reconstruction] Error (no bloqueante): %s", exc, exc_info=True)

        elapsed = time.time() - t0
        LOGGER.info("[Reconstruction] Completada en %.2fs.", elapsed)
        return paths, flows, elapsed

    # ------------------------------------------------------------------
    # Helper: obtener flows reales del test set
    # ------------------------------------------------------------------

    def _get_real_flows(
        self,
        dm:        TrafficDataModule,
        rep_cfg:   Dict[str, Any],
        model_cfg: Dict[str, Any],
    ) -> Tuple[List[Any], bool]:
        """
        Obtiene flows reales del test set para TrafficStructuralEvaluator.

        Estrategia (en orden de preferencia):
        1. Acceso directo a dm.test_samples si contienen objetos Flow con
           información estructural suficiente (fuente no sesgada). [PREFERIDO]
        2. Si no es posible: reconstrucción desde representaciones del test set
           vía el decoder (introduce sesgo del decoder). En ese caso:
           - La clave de métricas se renombra a ``traffic_structural_reconstructed``.
           - Se emite un WARNING explícito.

        Returns
        -------
        (flows, is_raw)
            flows   : lista de flows reales (o reconstruidos).
            is_raw  : True si son flows crudos originales; False si son
                      reconstruidos (métricas potencialmente sesgadas).
        """
        # --- Intento 1: acceso a flows crudos via dm.test_samples ----------
        try:
            from ..preprocessing.domain.flow import Flow as TrafficFlow
            test_samples = dm.test_samples           # List[TrafficSample]
            raw_flows = [s for s in test_samples if isinstance(s, TrafficFlow)]
            if raw_flows:
                LOGGER.info(
                    "[RealFlows] Usando %d flows crudos del test set (sin sesgo del decoder).",
                    len(raw_flows),
                )
                return raw_flows, True
            else:
                LOGGER.warning(
                    "[RealFlows] dm.test_samples no contiene objetos Flow (%d muestras, "
                    "tipo: %s). Recurriendo a reconstrucción.",
                    len(test_samples),
                    type(test_samples[0]).__name__ if test_samples else "vacío",
                )
        except Exception as exc:
            LOGGER.warning(
                "[RealFlows] No se pudo acceder a dm.test_samples directamente: %s. "
                "Recurriendo a reconstrucción.",
                exc,
            )

        # --- Intento 2 (fallback): reconstrucción desde representaciones ----
        LOGGER.warning(
            "[RealFlows] *** AVISO DE SESGO ***: Las métricas de tráfico se calculan "
            "sobre datos REALES RECONSTRUIDOS por el decoder. Esto introduce el sesgo "
            "del decoder y puede invalidar la comparación. "
            "Se recomienda usar flows crudos (dm.test_samples). "
            "La clave de métricas será 'traffic_structural_reconstructed'."
        )
        flows: List[Any] = []
        try:
            from ..reconstruction.build import build_reconstructor
            rec = build_reconstructor(
                rep_name   = rep_cfg.get("representation_type", "unknown"),
                model_name = model_cfg.get("model_type", "unknown"),
                verbose    = False,
            )
            real_ts, real_ys = [], []
            for batch in dm.test_dataloader():
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                y = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else None
                real_ts.append(x)
                if y is not None:
                    real_ys.append(y)
            if not real_ts:
                return flows, False
            real_data = torch.cat(real_ts, dim=0)
            real_y    = torch.cat(real_ys, dim=0) if real_ys else None
            flows = rec.reconstruct(real_data, labels=real_y)
        except Exception as exc:
            LOGGER.warning("[RealFlows] Error en reconstrucción (no bloqueante): %s", exc)
        return flows, False

    # Alias de compatibilidad hacia atrás
    def _reconstruct_real(
        self,
        dm:        TrafficDataModule,
        rep_cfg:   Dict[str, Any],
        model_cfg: Dict[str, Any],
    ) -> List[Any]:
        """
        Alias de compatibilidad. Usa _get_real_flows internamente.

        .. deprecated::
            Usar _get_real_flows() que devuelve también el flag ``is_raw``.
        """
        flows, _ = self._get_real_flows(dm, rep_cfg, model_cfg)
        return flows

    # ------------------------------------------------------------------
    # Fase 11: Evaluacion composable
    # ------------------------------------------------------------------

    def _run_evaluation(
        self,
        dm:                 TrafficDataModule,
        rep:                TrafficRepresentation,
        rep_cfg:            Dict[str, Any],
        model_cfg:          Dict[str, Any],
        metrics_dir:        Path,
        synth_samples:      Optional[torch.Tensor],
        synth_flows:        List[Any],
        real_flows:         List[Any],
        synth_labels:       Optional[torch.Tensor] = None,
        real_flows_is_raw:  bool = False,
        seed:               int = 42,
    ) -> Dict[str, Any]:
        """
        Suite de evaluacion composable con prioridad de metricas explícita.

        Niveles y prioridad
        -------------------
        PRIMARY   [calidad generativa]
          1. statistical  - JS, EMD, Pearson           (nivel representacion)
          2. urs          - FID, MMD                   (nivel representacion)

        CRITICAL  [fidelidad condicional]
          3. conditional  - accuracy, f1_macro          (solo modelos condicionales)

        SECONDARY [validez estructural]
          4. structural   - validez de representacion   (nivel representacion)
          5. traffic_structural[_reconstructed]          (nivel trafico/paquete)

        TERTIARY  [utilidad downstream - no miden calidad generativa]
          6. tstr / trtr / tstr_trtr / anomaly

        Muestreo
        --------
        Se usa muestreo estratificado por clase (en lugar de truncacion) para
        evitar sesgo en la distribucion de clases durante la evaluacion.

        Parametros
        ----------
        real_flows_is_raw : bool
            True si los flows reales son crudos (sin sesgo del decoder).
            False si son reconstruidos (se usa clave ``traffic_structural_reconstructed``
            y se emite advertencia en metricas y logs).
        """
        LOGGER.info("[ExperimentRunner] Fase 11: Evaluacion composable (estratificada).")
        LOGGER.info(
            "[Evaluation] Prioridad: PRIMARY=(statistical,urs) | "
            "CRITICAL=(conditional) | SECONDARY=(structural,traffic_structural) | "
            "TERTIARY=(downstream tasks)"
        )

        all_metrics: Dict[str, Any] = {}
        rep_type    = rep_cfg.get("representation_type", "sequential")
        num_classes = model_cfg.get("num_classes", 0)

        # vocab_size
        vocab_size = getattr(rep, "vocab_size", None)
        if vocab_size is None:
            vocab_size = model_cfg.get("vocab_size", None)
        if vocab_size is not None:
            LOGGER.info("[Evaluation] vocab_size resuelto: %d", vocab_size)
        else:
            LOGGER.warning("[Evaluation] vocab_size no disponible para rep_type='%s'.", rep_type)

        # ----------------------------------------------------------------
        # Cargar test set real
        # ----------------------------------------------------------------
        real_data:   Optional[torch.Tensor] = None
        real_labels: Optional[torch.Tensor] = None

        if synth_samples is not None:
            real_ts: List[torch.Tensor] = []
            real_ys: List[torch.Tensor] = []
            for batch in dm.test_dataloader():
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                y = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else None
                real_ts.append(x)
                if y is not None:
                    real_ys.append(y)

            if real_ts:
                real_data = torch.cat(real_ts, dim=0)
                if real_ys:
                    real_labels = torch.cat(real_ys, dim=0)
                else:
                    LOGGER.warning(
                        "[Evaluation] Test set sin etiquetas. "
                        "Evaluadores condicionales y downstream se saltaran."
                    )

        # ----------------------------------------------------------------
        # Muestreo estratificado (evita sesgo de clase)
        # ----------------------------------------------------------------
        real_eval:         Optional[torch.Tensor] = None
        synth_eval:        Optional[torch.Tensor] = None
        real_labels_eval:  Optional[torch.Tensor] = None
        synth_labels_eval: Optional[torch.Tensor] = None

        if real_data is not None and synth_samples is not None:
            rng = np.random.RandomState(seed)
            (
                real_eval, synth_eval,
                real_labels_eval, synth_labels_eval,
            ) = self._stratified_sample(
                real_data, real_labels,
                synth_samples, synth_labels,
                rng=rng,
            )
            LOGGER.info(
                "[Evaluation] Muestreo estratificado: real=%d synth=%d",
                len(real_eval), len(synth_eval),
            )

        # ----------------------------------------------------------------
        # PRIMARY: statistical (JS, EMD) + URS (FID, MMD)
        # ----------------------------------------------------------------
        if real_eval is not None and synth_eval is not None:
            LOGGER.info("[Evaluation] [PRIMARY] StatisticalEvaluator (JS, EMD, Pearson)...")
            all_metrics["statistical"] = self._eval_statistical(real_eval, synth_eval)

            LOGGER.info("[Evaluation] [PRIMARY] URSEvaluator (FID, MMD)...")
            all_metrics["urs"] = self._eval_urs(real_eval, synth_eval, rep_type, vocab_size)

            # ----------------------------------------------------------------
            # CRITICAL: condicional (solo si el modelo es condicional)
            # ----------------------------------------------------------------
            is_conditional = bool(num_classes and num_classes > 1)
            if is_conditional:
                LOGGER.info(
                    "[Evaluation] [CRITICAL] ConditionalEvaluator (num_classes=%d)...",
                    num_classes,
                )
                if real_labels_eval is None:
                    LOGGER.warning(
                        "[Evaluation] Modelo condicional (num_classes=%d) pero "
                        "real_labels no disponibles. Saltando evaluacion condicional.",
                        num_classes,
                    )
                    all_metrics["conditional"] = {
                        "conditional_accuracy": float("nan"),
                        "conditional_f1_macro": float("nan"),
                        "_warning": "real_labels no disponibles",
                    }
                elif synth_labels_eval is None:
                    LOGGER.warning(
                        "[Evaluation] Modelo condicional pero synthetic_labels no "
                        "disponibles. Saltando evaluacion condicional."
                    )
                    all_metrics["conditional"] = {
                        "conditional_accuracy": float("nan"),
                        "conditional_f1_macro": float("nan"),
                        "_warning": "synthetic_labels no disponibles",
                    }
                else:
                    all_metrics["conditional"] = self._eval_conditional(
                        real_data=real_eval,
                        real_labels=real_labels_eval.cpu().numpy(),
                        synth_samples=synth_eval,
                        synth_labels=synth_labels_eval.cpu().numpy(),
                    )
            else:
                LOGGER.info(
                    "[Evaluation] Modelo no condicional (num_classes=%s). "
                    "Saltando ConditionalEvaluator.", num_classes,
                )

            # ----------------------------------------------------------------
            # SECONDARY: structural (representacion)
            # ----------------------------------------------------------------
            LOGGER.info("[Evaluation] [SECONDARY] StructuralEvaluator...")
            all_metrics["structural"] = self._eval_structural(
                real_eval, synth_eval, rep_type, vocab_size
            )

            # ----------------------------------------------------------------
            # TERTIARY: downstream tasks (utilidad, no calidad generativa)
            # ----------------------------------------------------------------
            eval_cfg  = self._raw_cfg.get("evaluation", {})
            tasks_cfg = eval_cfg.get("tasks", {})
            active_tasks = [k for k, v in tasks_cfg.items() if v and k != "max_features"]
            if tasks_cfg and active_tasks:
                LOGGER.info(
                    "[Evaluation] [TERTIARY] Tareas downstream "
                    "(miden UTILIDAD, no calidad generativa): %s", active_tasks,
                )
                if real_labels_eval is not None:
                    synth_labels_np = (
                        synth_labels_eval.cpu().numpy()
                        if synth_labels_eval is not None else None
                    )
                    if synth_labels_np is None:
                        LOGGER.warning(
                            "[Evaluation] Tareas downstream requieren synthetic_labels. "
                            "Saltando."
                        )
                    else:
                        real_all_labels = (
                            real_labels.cpu().numpy()
                            if real_labels is not None
                            else real_labels_eval.cpu().numpy()
                        )
                        task_metrics = self._eval_tasks(
                            real_eval, synth_eval,
                            real_labels_eval.cpu().numpy(),
                            synth_labels_np,
                            real_data, real_all_labels,
                            tasks_cfg,
                        )
                        all_metrics.update(task_metrics)
                else:
                    LOGGER.warning(
                        "[Evaluation] tasks_cfg definido pero no hay etiquetas. "
                        "Saltando tareas downstream."
                    )
        else:
            LOGGER.warning(
                "[Evaluation] Sin synth_samples o test set vacio. "
                "Saltando evaluadores PRIMARY, CRITICAL, SECONDARY y TERTIARY."
            )

        # ----------------------------------------------------------------
        # SECONDARY: traffic structural (nivel trafico/paquete)
        # ----------------------------------------------------------------
        traffic_key = (
            "traffic_structural" if real_flows_is_raw
            else "traffic_structural_reconstructed"
        )

        if synth_flows and real_flows:
            LOGGER.info(
                "[Evaluation] [SECONDARY] TrafficStructuralEvaluator "
                "(clave='%s', fuente_real=%s)...",
                traffic_key,
                "flows_crudos" if real_flows_is_raw else "reconstruidos_por_decoder",
            )
            traffic_result = self._eval_traffic_structural(real_flows, synth_flows)
            if not real_flows_is_raw:
                traffic_result["_warning"] = (
                    "Metricas calculadas sobre datos REALES RECONSTRUIDOS por el decoder. "
                    "Pueden estar sesgadas por el proceso de reconstruccion."
                )
            all_metrics[traffic_key] = traffic_result
        else:
            LOGGER.warning(
                "[Evaluation] Flows insuficientes para TrafficStructuralEvaluator "
                "(real=%d synth=%d). Saltando.",
                len(real_flows), len(synth_flows),
            )

        # ----------------------------------------------------------------
        # Metadatos del run de evaluacion
        # ----------------------------------------------------------------
        all_metrics["_meta"] = {
            "metric_priority": {
                "primary":   ["statistical", "urs"],
                "critical":  ["conditional"],
                "secondary": ["structural", traffic_key],
                "tertiary":  ["tstr", "trtr", "tstr_trtr", "anomaly"],
            },
            "traffic_metrics_source": (
                "raw_flows" if real_flows_is_raw else "reconstructed_by_decoder"
            ),
            "sampling_strategy":     "stratified_per_class",
            "is_conditional_model":  bool(num_classes and num_classes > 1),
            "num_classes":           num_classes,
        }

        # ----------------------------------------------------------------
        # Persistir y loguear resumen
        # ----------------------------------------------------------------
        if any(k for k in all_metrics if not k.startswith("_")):
            eval_path = metrics_dir / "evaluation.json"
            with open(eval_path, "w") as f:
                json.dump(all_metrics, f, indent=2, default=str)
            LOGGER.info("[Evaluation] Metricas guardadas en %s", eval_path)
            self._log_metrics_summary(all_metrics)

        return all_metrics

    def _stratified_sample(
        self,
        real_data:    torch.Tensor,
        real_labels:  Optional[torch.Tensor],
        synth_data:   torch.Tensor,
        synth_labels: Optional[torch.Tensor],
        rng:          np.random.RandomState,
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Muestreo estratificado por clase para evitar sesgo de distribucion.

        Garantias
        ---------
        - Misma distribucion de clases en real y sintetico.
        - Mismo numero total de muestras en ambos splits.
        - Si no hay etiquetas, recae en truncacion simple con aviso.
        """
        if real_labels is None or synth_labels is None:
            LOGGER.warning(
                "[Sampling] Sin etiquetas disponibles. Usando truncacion simple "
                "(puede introducir sesgo de clase)."
            )
            n = min(len(real_data), len(synth_data))
            return (
                real_data[:n], synth_data[:n],
                real_labels[:n] if real_labels is not None else None,
                synth_labels[:n] if synth_labels is not None else None,
            )

        real_y  = real_labels.cpu().numpy()
        synth_y = synth_labels.cpu().numpy()

        real_classes   = set(np.unique(real_y).tolist())
        synth_classes  = set(np.unique(synth_y).tolist())
        common_classes = sorted(real_classes & synth_classes)

        missing_in_synth = real_classes - synth_classes
        missing_in_real  = synth_classes - real_classes
        if missing_in_synth:
            LOGGER.warning(
                "[Sampling] Clases en real ausentes en sintetico: %s. "
                "Solo se usaran clases comunes.", sorted(missing_in_synth),
            )
        if missing_in_real:
            LOGGER.warning(
                "[Sampling] Clases en sintetico ausentes en real: %s. "
                "Solo se usaran clases comunes.", sorted(missing_in_real),
            )

        if not common_classes:
            LOGGER.warning(
                "[Sampling] Sin clases comunes. Recayendo en truncacion simple."
            )
            n = min(len(real_data), len(synth_data))
            return real_data[:n], synth_data[:n], real_labels[:n], synth_labels[:n]

        per_class_counts = [
            min(int(np.sum(real_y == c)), int(np.sum(synth_y == c)))
            for c in common_classes
        ]
        n_per_class = min(per_class_counts)

        if n_per_class == 0:
            LOGGER.warning(
                "[Sampling] Una o mas clases tienen 0 muestras. Recayendo en truncacion."
            )
            n = min(len(real_data), len(synth_data))
            return real_data[:n], synth_data[:n], real_labels[:n], synth_labels[:n]

        # Detectar desbalance
        max_count = max(
            max(int(np.sum(real_y == c)) for c in common_classes),
            max(int(np.sum(synth_y == c)) for c in common_classes),
        )
        if max_count > 5 * n_per_class:
            LOGGER.warning(
                "[Sampling] Desbalance de clases detectado. "
                "n_per_class=%d (limitado por la clase minoritaria). "
                "real=%s synth=%s",
                n_per_class,
                {c: int(np.sum(real_y == c))  for c in common_classes},
                {c: int(np.sum(synth_y == c)) for c in common_classes},
            )

        real_idxs, synth_idxs = [], []
        for c in common_classes:
            r_idx = np.where(real_y == c)[0]
            s_idx = np.where(synth_y == c)[0]
            real_idxs.append( rng.choice(r_idx, n_per_class, replace=False))
            synth_idxs.append(rng.choice(s_idx, n_per_class, replace=False))

        real_idx  = np.concatenate(real_idxs)
        synth_idx = np.concatenate(synth_idxs)

        LOGGER.info(
            "[Sampling] Estratificado: %d clases x %d muestras/clase = %d total.",
            len(common_classes), n_per_class, len(real_idx),
        )
        return (
            real_data[real_idx],
            synth_data[synth_idx],
            real_labels[real_idx],
            synth_labels[synth_idx],
        )

    def _log_metrics_summary(self, all_metrics: Dict[str, Any]) -> None:
        """Imprime un resumen priorizado de todas las metricas calculadas."""
        priority_order = [
            "statistical", "urs", "conditional",
            "structural", "traffic_structural", "traffic_structural_reconstructed",
            "tstr", "trtr", "tstr_trtr", "anomaly",
        ]
        extra_keys = [
            k for k in all_metrics
            if k not in priority_order and not k.startswith("_")
        ]
        ordered = [k for k in priority_order if k in all_metrics] + extra_keys

        LOGGER.info("[Evaluation] ===== RESUMEN DE METRICAS =====")
        for key in ordered:
            val = all_metrics[key]
            if isinstance(val, dict):
                numeric = {
                    k: v for k, v in val.items()
                    if isinstance(v, (int, float)) and not k.startswith("_")
                }
                if numeric:
                    LOGGER.info("[Evaluation]  [%s] %s", key.upper(), numeric)
        LOGGER.info("[Evaluation] ===================================")

    def _eval_conditional(
        self,
        real_data:     torch.Tensor,
        real_labels:   "np.ndarray",
        synth_samples: torch.Tensor,
        synth_labels:  "np.ndarray",
    ) -> Dict[str, Any]:
        """
        Evalua la fidelidad condicional del modelo generativo.

        Protocolo
        ---------
        1. clf.fit(real_data, real_labels)
        2. preds = clf.predict(synth_samples)
        3. accuracy = accuracy_score(synth_labels, preds)
           f1      = f1_score(synth_labels, preds, average="macro")

        Una accuracy alta indica que el modelo genera muestras que el
        clasificador real asocia a la clase condicionante correcta.
        """
        try:
            from ..evaluation import ConditionalEvaluator
            r = ConditionalEvaluator().evaluate(
                real=real_data,
                synthetic=synth_samples,
                real_labels=real_labels,
                synthetic_labels=synth_labels,
            )
            summary = r.summary()
            LOGGER.info("[Evaluation] Condicional: %s", summary)
            return summary
        except Exception as e:
            LOGGER.warning("[Evaluation] ConditionalEvaluator: %s", e)
            return {"error": str(e)}

    def _eval_statistical(self, real: torch.Tensor, synth: torch.Tensor) -> Dict[str, Any]:
        try:
            from ..evaluation import StatisticalEvaluator
            r = StatisticalEvaluator().evaluate(real, synth)
            summary = r.summary()
            LOGGER.info("[Evaluation] Estadistica: %s", summary)
            return summary
        except Exception as e:
            LOGGER.warning("[Evaluation] StatisticalEvaluator: %s", e)
            return {"error": str(e)}

    def _eval_structural(
        self, real: torch.Tensor, synth: torch.Tensor,
        rep_type: str, vocab_size: Optional[int],
    ) -> Dict[str, Any]:
        try:
            from ..evaluation import StructuralEvaluator
            r = StructuralEvaluator(
                representation_type=rep_type, vocab_size=vocab_size
            ).evaluate(real, synth)
            summary = r.summary()
            LOGGER.info("[Evaluation] Estructural: %s", summary)
            return summary
        except Exception as e:
            LOGGER.warning("[Evaluation] StructuralEvaluator: %s", e)
            return {"error": str(e)}

    def _eval_urs(
        self, real: torch.Tensor, synth: torch.Tensor,
        rep_type: str, vocab_size: Optional[int],
    ) -> Dict[str, Any]:
        try:
            from ..evaluation import URSEvaluator
            from ..utils.build_traffic_encoder import build_traffic_encoder
            is_seq      = rep_type in ("flat_tokenizer", "protocol_aware", "semantic_byte")
            in_channels = synth.shape[1] if synth.dim() == 4 else None
            if is_seq and vocab_size is None:
                raise ValueError(
                    f"vocab_size es necesario para URSEvaluator con rep_type='{rep_type}'. "
                    "Añade vocab_size en la sección model del YAML."
                )
            encoder = build_traffic_encoder(
                rep_type,
                vocab_size  = vocab_size if is_seq else None,
                in_channels = in_channels,
            )
            r = URSEvaluator(encoder=encoder).evaluate(real, synth)
            summary = r.summary()
            LOGGER.info("[Evaluation] URS: %s", summary)
            return summary
        except Exception as e:
            LOGGER.warning("[Evaluation] URSEvaluator: %s", e)
            return {"error": str(e)}

    def _eval_traffic_structural(
        self, real_flows: List[Any], synth_flows: List[Any],
    ) -> Dict[str, Any]:
        try:
            from ..evaluation import TrafficStructuralEvaluator
            r = TrafficStructuralEvaluator().evaluate(real_flows, synth_flows)
            summary = r.summary()
            LOGGER.info("[Evaluation] TrafficStructural: %s", summary)
            return summary
        except Exception as e:
            LOGGER.warning("[Evaluation] TrafficStructuralEvaluator: %s", e)
            return {"error": str(e)}
    
    # ------------------------------------------------------------------
    # Fase 11b: Tareas downstream (clasificacion / anomaly detection)
    # ------------------------------------------------------------------

    def _eval_tasks(
        self,
        real_eval:       torch.Tensor,
        synth_eval:      torch.Tensor,
        real_labels_np:  np.ndarray,
        synth_labels_np: Optional[np.ndarray],
        real_all:        torch.Tensor,
        real_all_labels: np.ndarray,
        tasks_cfg:       Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ejecuta tareas downstream configuradas en evaluation.tasks del YAML.

        Tareas soportadas
        -----------------
        tstr      : Train on Synthetic, Test on Real (clasificacion).
        trtr      : Train on Real, Test on Real (baseline de clasificacion).
        tstr_trtr : Ambas + gap entre ellas.
        anomaly   : Deteccion de anomalias con IsolationForest.

        Configuracion YAML ejemplo
        --------------------------
        evaluation:
          tasks:
            tstr: true
            trtr: true
            tstr_trtr: false
            anomaly: true
            max_features: 256
        """
        import numpy as np
        from ..evaluation import (
            TSTREvaluator, TRTREvaluator,
            TSTRTRTRComparisonEvaluator,
            TaskRunnerEvaluator, AnomalyDetectionTask,
        )

        task_metrics: Dict[str, Any] = {}
        max_features = int(tasks_cfg.get("max_features", 256))

        def _run(evaluator, **kwargs) -> Dict[str, Any]:
            try:
                r = evaluator.evaluate(
                    real=real_eval,
                    synthetic=synth_eval,
                    real_labels=real_labels_np,
                    synthetic_labels=synth_labels_np,
                    **kwargs,
                )
                return r.summary()
            except Exception as exc:
                LOGGER.warning("[Evaluation] Task %s: %s", evaluator.name, exc)
                return {"error": str(exc)}

        if tasks_cfg.get("tstr_trtr", False):
            LOGGER.info("[Evaluation] Task: TSTR+TRTR comparison.")
            ev = TSTRTRTRComparisonEvaluator(max_features=max_features)
            task_metrics["tstr_trtr"] = _run(
                ev,
                real_train=real_all,
                real_train_labels=real_all_labels,
            )
        else:
            if tasks_cfg.get("tstr", False):
                LOGGER.info("[Evaluation] Task: TSTR.")
                task_metrics["tstr"] = _run(TSTREvaluator(max_features=max_features))

            if tasks_cfg.get("trtr", False):
                LOGGER.info("[Evaluation] Task: TRTR.")
                task_metrics["trtr"] = _run(
                    TRTREvaluator(max_features=max_features),
                    real_train=real_all,
                    real_train_labels=real_all_labels,
                )

        if tasks_cfg.get("anomaly", False):
            LOGGER.info("[Evaluation] Task: AnomalyDetection.")
            ev = TaskRunnerEvaluator(
                task=AnomalyDetectionTask(max_features=max_features),
                name="AnomalyDetection",
                category="utility",
            )
            task_metrics["anomaly"] = _run(ev)

        if task_metrics:
            LOGGER.info("[Evaluation] Tasks downstream completadas: %s", list(task_metrics.keys()))
        return task_metrics

    # ------------------------------------------------------------------
    # Fase 12: Visualizacion
    # ------------------------------------------------------------------

    def _run_visualization(
        self,
        run_dir:     Path,
        metrics_dir: Path,
        plots_dir:   Path,
        result:      ExperimentResult,
    ) -> None:
        """Genera todos los plots del experimento via ExperimentPlotter."""
        LOGGER.info("[ExperimentRunner] Fase 12: Generando visualizaciones.")
        try:
            from ..visualization import ExperimentPlotter
            metrics_csv = metrics_dir / f"{self._exp_name}_metrics.csv"
            timing_json = metrics_dir / f"{self._exp_name}_timing.json"
            plotter = ExperimentPlotter(plots_dir=plots_dir, exp_name=self._exp_name)
            plotter.plot_all(
                metrics_csv = metrics_csv if metrics_csv.exists() else None,
                timing_json = timing_json if timing_json.exists() else None,
                result      = result,
            )
        except Exception as exc:
            LOGGER.warning("[Visualization] Error (no bloqueante): %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Funcion de conveniencia
# ---------------------------------------------------------------------------

def run_experiment(
    yaml_path: str,
    data_dir:  Optional[str] = None,
    device:    Optional[str] = None,
) -> ExperimentResult:
    """
    Shortcut para lanzar un experimento desde una sola linea.

    >>> result = run_experiment("src/configs/exp_ddpm_nprint.yaml")
    """
    return ExperimentRunner(yaml_path, data_dir=data_dir, device=device).run()


__all__ = ["ExperimentRunner", "ExperimentResult", "run_experiment"]
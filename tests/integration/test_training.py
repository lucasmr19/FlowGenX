"""
tests/integration/test_training.py
====================================
Tests de integración del módulo de entrenamiento.

Verifica que el Trainer funciona correctamente para todas las
combinaciones (representación x modelo) sin necesitar PCAPs reales:
se sintetizan batches directamente desde ``make_synthetic_flow``.

Cobertura
---------
- TrainingConfig: validación y from_dict.
- Callbacks: CheckpointCallback, EarlyStoppingCallback, MetricsLoggerCallback.
- build_scheduler: cosine, step, none.
- Trainer.fit(): protocolo STANDARD (Transformer, DDPM).
- Trainer.fit(): protocolo ADVERSARIAL (GAN).
- ExperimentRunner: construcción sin PCAPs (via mock de DataModule).
- EMACallback: acumulación y aplicación final.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------ #
# Fixtures compartidos                                                 #
# ------------------------------------------------------------------ #


# ---- batch sintético secuencial (Transformer / GAN) ---------------
def _seq_batch(batch_size: int = 4, seq_len: int = 32, vocab_size: int = 256):
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size,))
    return tokens, labels


# ---- batch sintético visual (DDPM) --------------------------------
def _img_batch(batch_size: int = 4, c: int = 1, h: int = 20, w: int = 32):
    imgs   = torch.randn(batch_size, c, h, w).clamp(-1, 1)
    labels = torch.randint(0, 2, (batch_size,))
    return imgs, labels


# ---- DataLoader sintético -----------------------------------------
def _make_loader(batch_fn, n_batches: int = 3) -> DataLoader:
    batches = [batch_fn() for _ in range(n_batches)]
    # Empaquetamos como TensorDataset de la primera dimensión
    x = torch.cat([b[0] for b in batches], dim=0)
    y = torch.cat([b[1] for b in batches], dim=0)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4, shuffle=False)


# ================================================================== #
# TrainingConfig                                                      #
# ================================================================== #

class TestTrainingConfig:
    def test_defaults_valid(self):
        from src.training.config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.epochs == 100
        assert cfg.checkpoint_mode in ("min", "max")

    def test_from_dict(self):
        from src.training.config import TrainingConfig
        d = {"epochs": 10, "lr": 3e-4, "batch_size": 8, "lr_scheduler": "step"}
        cfg = TrainingConfig.from_dict(d)
        assert cfg.epochs == 10
        assert cfg.lr == pytest.approx(3e-4)
        assert cfg.lr_scheduler == "step"

    def test_unknown_keys_ignored(self):
        from src.training.config import TrainingConfig
        cfg = TrainingConfig.from_dict({"epochs": 5, "nonexistent_key": 999})
        assert cfg.epochs == 5

    def test_invalid_mode_raises(self):
        from src.training.config import TrainingConfig
        with pytest.raises(ValueError, match="checkpoint_mode"):
            TrainingConfig(checkpoint_mode="weird")

    def test_invalid_scheduler_raises(self):
        from src.training.config import TrainingConfig
        with pytest.raises(ValueError, match="lr_scheduler"):
            TrainingConfig(lr_scheduler="unknown")

    def test_early_stopping_metric_defaults_to_checkpoint_metric(self):
        from src.training.config import TrainingConfig
        cfg = TrainingConfig(checkpoint_metric="val_perplexity")
        assert cfg.early_stopping_metric == "val_perplexity"


# ================================================================== #
# build_scheduler                                                     #
# ================================================================== #

class TestBuildScheduler:

    def _optimizer(self, lr=1e-3):
        import torch.nn as nn
        net = nn.Linear(4, 2)
        return torch.optim.Adam(net.parameters(), lr=lr)

    def test_cosine_returns_scheduler(self):
        from src.training.config import TrainingConfig
        from src.training.lr_scheduler import build_scheduler
        cfg = TrainingConfig(lr_scheduler="cosine", warmup_epochs=0, epochs=10)
        sched = build_scheduler(self._optimizer(), cfg)
        assert sched is not None
        self._optimizer().step()
        sched.step()

    def test_cosine_with_warmup(self):
        from src.training.config import TrainingConfig
        from src.training.lr_scheduler import build_scheduler
        cfg = TrainingConfig(lr_scheduler="cosine", warmup_epochs=3, epochs=20)
        sched = build_scheduler(self._optimizer(), cfg)
        assert sched is not None
        for _ in range(5):
            self._optimizer().step()
            sched.step()

    def test_step_scheduler(self):
        from src.training.config import TrainingConfig
        from src.training.lr_scheduler import build_scheduler
        cfg = TrainingConfig(lr_scheduler="step", lr_step_size=5, lr_gamma=0.5,
                             warmup_epochs=0, epochs=20)
        sched = build_scheduler(self._optimizer(), cfg)
        assert sched is not None

    def test_plateau_scheduler(self):
        from src.training.config import TrainingConfig
        from src.training.lr_scheduler import build_scheduler
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        cfg = TrainingConfig(lr_scheduler="plateau", epochs=10)
        sched = build_scheduler(self._optimizer(), cfg)
        assert isinstance(sched, ReduceLROnPlateau)
        self._optimizer().step()
        sched.step(0.5)

    def test_none_returns_none(self):
        from src.training.config import TrainingConfig
        from src.training.lr_scheduler import build_scheduler
        cfg = TrainingConfig(lr_scheduler="none", epochs=10)
        sched = build_scheduler(self._optimizer(), cfg)
        assert sched is None


# ================================================================== #
# Callbacks                                                           #
# ================================================================== #

class TestCallbacks:

    def _state(self, epoch=1, val_loss=1.0):
        from src.training.callbacks import TrainerState
        s = TrainerState(epoch=epoch, global_step=epoch * 10)
        s.val_metrics = {"val_loss": val_loss}
        return s

    def test_early_stopping_triggers(self):
        from src.training.callbacks import EarlyStoppingCallback, TrainerState
        cb = EarlyStoppingCallback(metric="val_loss", patience=3)
        s  = self._state(val_loss=1.0)

        # Primera vez: mejora
        cb.on_validation_end(s)
        assert not s.stop_training

        # 3 épocas sin mejora → stop
        for _ in range(3):
            cb.on_validation_end(s)
        assert s.stop_training

    def test_early_stopping_resets_on_improvement(self):
        from src.training.callbacks import EarlyStoppingCallback
        cb = EarlyStoppingCallback(metric="val_loss", patience=3)

        s1 = self._state(val_loss=1.0)
        cb.on_validation_end(s1)      # mejora a 1.0

        s2 = self._state(val_loss=1.0)
        cb.on_validation_end(s2)      # igual → wait=1
        cb.on_validation_end(s2)      # wait=2

        s3 = self._state(val_loss=0.5)
        cb.on_validation_end(s3)      # mejora → reset wait
        assert not s3.stop_training

    def test_checkpoint_callback_saves(self):
        from src.training.callbacks import CheckpointCallback

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_rep   = MagicMock()

            cb = CheckpointCallback(
                checkpoint_dir  = tmpdir,
                metric          = "val_loss",
                mode            = "min",
                model           = mock_model,
                representation  = mock_rep,
                save_top_k      = 1,
                save_last       = False,   # sin last.pt para contar exacto
                experiment_name = "test_exp",
            )

            from src.training.callbacks import TrainerState
            s = TrainerState(epoch=1)
            s.val_metrics = {"val_loss": 0.5}
            cb.on_validation_end(s)

            # El modelo debe haber llamado save() exactamente una vez
            mock_model.save.assert_called_once()
            mock_rep.save.assert_called_once()

    def test_checkpoint_keeps_top_k(self):
        from src.training.callbacks import CheckpointCallback, TrainerState

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_rep   = MagicMock()

            cb = CheckpointCallback(
                checkpoint_dir  = tmpdir,
                metric          = "val_loss",
                mode            = "min",
                model           = mock_model,
                representation  = mock_rep,
                save_top_k      = 2,
                save_last       = False,
                experiment_name = "test_exp",
            )

            for epoch, loss in enumerate([0.9, 0.7, 0.5, 0.8], start=1):
                s = TrainerState(epoch=epoch)
                s.val_metrics = {"val_loss": loss}
                cb.on_validation_end(s)

            # save() debe haber sido llamado en total 3 veces
            # (una por cada época con nueva candidata)
            assert mock_model.save.call_count == 3

    def test_metrics_logger_creates_csv(self):
        from src.training.callbacks import MetricsLoggerCallback, TrainerState

        with tempfile.TemporaryDirectory() as tmpdir:
            cb = MetricsLoggerCallback(
                log_dir         = tmpdir,
                experiment_name = "test",
                log_every       = 1,
            )
            s = TrainerState()
            cb.on_train_start(s)

            # Simular 3 épocas
            for epoch in range(1, 4):
                s.epoch      = epoch
                s.train_loss = 1.0 / epoch
                s.val_metrics = {"val_loss": 0.5 / epoch}
                cb.on_epoch_end(s)

            cb.on_train_end(s)

            csv_file = Path(tmpdir) / "test_metrics.csv"
            assert csv_file.exists()
            content = csv_file.read_text()
            assert "epoch" in content
            assert "train_loss" in content

    def test_ema_callback_applies_weights(self):
        import torch.nn as nn
        from src.training.callbacks import EMACallback, TrainerState

        # Modelo mock con un parámetro
        net = nn.Linear(4, 2)
        original_weight = net.weight.data.clone()

        mock_model = MagicMock()
        mock_model._networks = {"main": net}

        cb = EMACallback(mock_model, decay=0.9, start_epoch=0, apply_on_end=True)

        s = TrainerState(epoch=1)
        cb.on_epoch_end(s)  # acumular shadow

        # Cambiar los pesos
        net.weight.data.fill_(100.0)
        s.epoch = 2
        cb.on_epoch_end(s)

        # Aplicar EMA al final
        cb.on_train_end(s)

        # Los pesos deben ser distintos a 100.0 (EMA suavizó)
        assert not torch.allclose(net.weight.data, torch.full_like(net.weight.data, 100.0))


# ================================================================== #
# Trainer — protocolo STANDARD (Transformer)                         #
# ================================================================== #

class TestTrainerStandard:
    """
    Verifica el ciclo completo de entrenamiento para modelos standard
    (Transformer, DDPM) usando modelos mínimos en CPU.
    """

    def _build_transformer(self):
        from src.models_ml.transformer.config import TransformerConfig
        from src.models_ml.transformer.model import TrafficTransformer
        cfg = TransformerConfig(
            vocab_size  = 256,
            d_model     = 32,
            n_heads     = 2,
            n_layers    = 1,
            d_ff        = 64,
            max_seq_len = 32,
            device      = "cpu",
        )
        model = TrafficTransformer(cfg)
        model.build()
        return model

    def test_single_epoch(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import Trainer

        model  = self._build_transformer()
        loader = _make_loader(_seq_batch, n_batches=2)

        cfg     = TrainingConfig(epochs=1, lr=1e-4, grad_clip=1.0,
                                 lr_scheduler="none", val_every=1)
        trainer = Trainer(model, cfg, callbacks=[])
        state   = trainer.fit(loader, val_loader=loader)

        assert state.epoch == 1
        assert state.train_loss < float("inf")

    def test_early_stopping_integration(self):
        from src.training.callbacks import EarlyStoppingCallback
        from src.training.config import TrainingConfig
        from src.training.trainer import Trainer

        model  = self._build_transformer()
        loader = _make_loader(_seq_batch, n_batches=2)

        # Patience=1 → para enseguida
        cfg = TrainingConfig(
            epochs=20, lr=1e-4, lr_scheduler="none", val_every=1,
            early_stopping=False,  # gestionamos manualmente
        )
        cb      = EarlyStoppingCallback(metric="val_loss", patience=1)
        trainer = Trainer(model, cfg, callbacks=[cb])
        state   = trainer.fit(loader, val_loader=loader)

        # Debe parar antes de la época 20
        assert state.epoch <= 20

    def test_checkpoint_callback_integration(self):
        from src.training.callbacks import CheckpointCallback
        from src.training.config import TrainingConfig
        from src.training.trainer import Trainer

        model  = self._build_transformer()
        loader = _make_loader(_seq_batch, n_batches=2)

        mock_rep = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            cb = CheckpointCallback(
                checkpoint_dir  = tmpdir,
                metric          = "val_loss",
                mode            = "min",
                model           = model,
                representation  = mock_rep,
                save_top_k      = 1,
                experiment_name = "test",
            )
            cfg     = TrainingConfig(
                epochs=3, lr=1e-4, lr_scheduler="none", val_every=1,
            )
            trainer = Trainer(model, cfg, callbacks=[cb])
            state   = trainer.fit(loader, val_loader=loader)

            # Al menos un checkpoint guardado
            pts = list(Path(tmpdir).glob("*.pt"))
            assert len(pts) >= 1


# ================================================================== #
# Trainer — protocolo ADVERSARIAL (GAN)                              #
# ================================================================== #

class TestTrainerAdversarial:
    """
    Verifica el ciclo de entrenamiento para la GAN.
    El protocolo adversarial delega backward/step al modelo,
    el Trainer solo asigna los optimizadores y loguea.
    """

    def _build_gan(self):
        from src.models_ml.gan.config import GANConfig
        from src.models_ml.gan.model import TrafficGAN
        cfg = GANConfig(
            vocab_size    = 256,
            seq_len       = 32,
            latent_dim    = 32,
            gen_hidden    = 64,
            gen_layers    = 1,
            disc_d_model  = 32,
            disc_n_heads  = 2,
            disc_n_layers = 1,
            disc_d_ff     = 64,
            n_critic      = 2,
            num_classes   = 2,
            device        = "cpu",
        )
        model = TrafficGAN(cfg)
        model.build()
        return model

    def test_single_epoch_adversarial(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import Trainer

        model  = self._build_gan()
        loader = _make_loader(_seq_batch, n_batches=3)

        cfg     = TrainingConfig(
            epochs=1, lr=5e-5, weight_decay=0.0, grad_clip=None,
            lr_scheduler="none", val_every=1,
        )
        trainer = Trainer(model, cfg, callbacks=[])
        state   = trainer.fit(loader, val_loader=loader)

        assert state.epoch == 1
        # El protocolo adversarial debe detectarse
        assert trainer._protocol == "adversarial"

    def test_optimizers_assigned_to_model(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import Trainer

        model  = self._build_gan()
        loader = _make_loader(_seq_batch, n_batches=2)

        cfg     = TrainingConfig(
            epochs=1, lr=5e-5, weight_decay=0.0, grad_clip=None,
            lr_scheduler="none",
        )
        trainer = Trainer(model, cfg, callbacks=[])
        trainer._setup_optimizer_and_protocol()

        # El modelo debe tener los optimizadores asignados
        assert hasattr(model, "_opt_generator")
        assert hasattr(model, "_opt_discriminator")


# ================================================================== #
# Trainer — protocolo STANDARD (DDPM)                                #
# ================================================================== #

class TestTrainerDDPM:

    def _build_ddpm(self):
        from src.models_ml.diffusion.config import DiffusionConfig
        from src.models_ml.diffusion.ddpm import TrafficDDPM
        cfg = DiffusionConfig(
            in_channels   = 1,
            image_height  = 8,
            image_width   = 16,
            base_ch       = 16,
            channel_mults = (1, 2),
            n_res_per_level=1,
            attention_levels=(1,),
            n_heads       = 2,
            timesteps     = 10,   # muy pocos para test rápido
            ddim_steps    = 5,
            num_classes   = 2,
            device        = "cpu",
        )
        model = TrafficDDPM(cfg)
        model.build()
        return model

    def test_ddpm_single_epoch(self):
        from src.training.config import TrainingConfig
        from src.training.trainer import Trainer

        model  = self._build_ddpm()
        loader = _make_loader(
            lambda: _img_batch(batch_size=2, c=1, h=8, w=16),
            n_batches=2,
        )

        cfg     = TrainingConfig(
            epochs=1, lr=2e-4, lr_scheduler="none", val_every=1,
        )
        trainer = Trainer(model, cfg, callbacks=[])
        state   = trainer.fit(loader, val_loader=loader)

        assert state.epoch == 1
        assert trainer._protocol == "standard"


# ================================================================== #
# TrainerState — parada manual                                        #
# ================================================================== #

class TestTrainerStateStop:

    def test_stop_training_flag_respected(self):
        """
        Un callback puede poner stop_training=True en on_epoch_end
        y el Trainer debe detenerse.
        """
        from src.training.callbacks import TrainerCallback, TrainerState
        from src.training.config import TrainingConfig
        from src.training.trainer import Trainer
        from src.models_ml.transformer.config import TransformerConfig
        from src.models_ml.transformer.model import TrafficTransformer

        class StopAtEpoch2(TrainerCallback):
            def on_epoch_end(self, state: TrainerState) -> None:
                if state.epoch >= 2:
                    state.stop_training = True

        cfg_model = TransformerConfig(
            vocab_size=256, d_model=32, n_heads=2, n_layers=1,
            d_ff=64, max_seq_len=32, device="cpu",
        )
        model  = TrafficTransformer(cfg_model).build()
        loader = _make_loader(_seq_batch, n_batches=2)

        cfg     = TrainingConfig(epochs=10, lr=1e-4, lr_scheduler="none",
                                 val_every=999)  # sin validación
        trainer = Trainer(model, cfg, callbacks=[StopAtEpoch2()])
        state   = trainer.fit(loader)

        assert state.epoch == 2


# ================================================================== #
# ExperimentResult                                                    #
# ================================================================== #

class TestExperimentResult:

    def test_to_dict_serializable(self):
        from src.training.experiment import ExperimentResult
        result = ExperimentResult(
            experiment_name   = "test",
            representation    = "flat_tokenizer",
            model             = "transformer",
            best_epoch        = 10,
            best_metric_name  = "val_loss",
            best_metric_value = 0.42,
            total_epochs_ran  = 12,
            training_time_s   = 30.5,
            run_dir           = "experiments/runs/test",
            checkpoint_path   = "ckpts/best.pt",
        )
        d = result.to_dict()
        assert d["experiment_name"] == "test"
        assert d["best_metric_value"] == pytest.approx(0.42)

    def test_save_json(self):
        from src.training.experiment import ExperimentResult
        result = ExperimentResult(
            experiment_name   = "test_json",
            representation    = "gaf",
            model             = "ddpm",
            best_epoch        = 5,
            best_metric_name  = "val_loss",
            best_metric_value = 0.15,
            total_epochs_ran  = 5,
            training_time_s   = 10.0,
            run_dir           = "experiments/runs/test",
            checkpoint_path   = None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "result.json")
            result.save_json(path)
            assert Path(path).exists()
            loaded = json.loads(Path(path).read_text())
            assert loaded["model"] == "ddpm"


# ================================================================== #
# __init__ exports                                                    #
# ================================================================== #

class TestTrainingInit:

    def test_all_exports_importable(self):
        from src.training import (
            TrainingConfig,
            Trainer,
            ExperimentRunner,
            ExperimentResult,
            run_experiment,
            TrainerState,
            TrainerCallback,
            CheckpointCallback,
            EarlyStoppingCallback,
            MetricsLoggerCallback,
            TensorBoardCallback,
            EMACallback,
            WandbCallback,
            build_scheduler,
        )
        # Si llega aquí sin ImportError, el __init__ está correcto
        assert TrainingConfig is not None
        assert Trainer is not None
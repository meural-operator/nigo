"""
Tests for training infrastructure: checkpointing, resumability, logger.
"""
import os
import json
import tempfile
import pytest
import torch
import numpy as np

from turbo_nigo.models import GlobalTurboNIGO
from turbo_nigo.utils.logger import ExperimentLogger, CSVLogger, JSONLinesLogger
from turbo_nigo.utils.misc import count_parameters, get_system_info, format_time


class TestCheckpointing:
    """Tests for full-state checkpoint save/load roundtrip."""

    def _make_trainer_state(self, tmp_dir):
        """Creates a minimal trainer-like state dict."""
        model = GlobalTurboNIGO(
            latent_dim=8, num_bases=2, cond_dim=2, width=8,
            in_channels=2, spatial_size=32,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(device="cpu", enabled=False)

        # Do a fake forward+backward to populate optimizer state
        x = torch.randn(2, 2, 32, 32)
        cond = torch.randn(2, 2)
        ts = torch.arange(1, 4).float() * 0.1
        out, *_ = model(x, ts, cond)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        return model, optimizer, scaler

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model, optimizer, scaler = self._make_trainer_state(tmp_dir)

            # Save state
            epoch = 42
            best_loss = 0.001234
            state = {
                "epoch": epoch,
                "best_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            ckpt_path = os.path.join(tmp_dir, "test_ckpt.pth")
            torch.save(state, ckpt_path)

            # Load and verify
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert ckpt["epoch"] == epoch
            assert abs(ckpt["best_loss"] - best_loss) < 1e-10

            # Load into fresh model
            model2 = GlobalTurboNIGO(
                latent_dim=8, num_bases=2, cond_dim=2, width=8,
                in_channels=2, spatial_size=32,
            )
            model2.load_state_dict(ckpt["model_state_dict"])

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"


class TestLogger:
    """Tests for ExperimentLogger backends."""

    def test_csv_logger(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "test.csv")
            logger = CSVLogger(path, ["epoch", "loss"])
            logger.log([1, 0.5])
            logger.log([2, 0.3])

            with open(path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 rows

    def test_jsonl_logger(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "test.jsonl")
            logger = JSONLinesLogger(path)
            logger.log({"epoch": 1, "loss": 0.5})
            logger.log({"epoch": 2, "loss": 0.3})

            with open(path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 2
            rec = json.loads(lines[0])
            assert rec["epoch"] == 1

    def test_experiment_logger_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(tmp_dir, use_tensorboard=False)
            logger.save_metadata(
                config={"lr": 1e-3, "epochs": 100},
                model_info={"total": 1000, "trainable": 1000},
            )

            meta_path = os.path.join(tmp_dir, "experiment_metadata.json")
            assert os.path.exists(meta_path)

            with open(meta_path) as f:
                meta = json.load(f)
            assert "config" in meta
            assert "system" in meta
            assert meta["config"]["lr"] == 1e-3

    def test_experiment_logger_log_epoch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = ExperimentLogger(tmp_dir, use_tensorboard=False)
            logger.log_epoch(1, {"train_loss": 0.5, "val_loss": 0.4, "lr": 1e-3})
            logger.log_epoch(2, {"train_loss": 0.3, "val_loss": 0.25, "lr": 9e-4})
            logger.close()

            # CSV
            csv_path = os.path.join(tmp_dir, "metrics.csv")
            assert os.path.exists(csv_path)

            # JSONL
            jsonl_path = os.path.join(tmp_dir, "metrics.jsonl")
            assert os.path.exists(jsonl_path)
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]
            assert len(records) == 2
            assert records[1]["epoch"] == 2


class TestMiscUtils:
    """Tests for utility functions."""

    def test_count_parameters(self):
        model = GlobalTurboNIGO(
            latent_dim=8, num_bases=2, cond_dim=2, width=8,
            in_channels=2, spatial_size=32,
        )
        info = count_parameters(model)
        assert info["total"] > 0
        assert info["trainable"] == info["total"]
        assert info["non_trainable"] == 0
        assert isinstance(info["total_million"], float)

    def test_get_system_info(self):
        info = get_system_info()
        assert "python_version" in info
        assert "torch_version" in info
        assert "numpy_version" in info
        # Should NOT contain PII
        for key in info:
            if isinstance(info[key], str):
                assert "DIAT" not in info[key]
                assert "ashish" not in info[key]

    def test_format_time(self):
        assert "s" in format_time(5.3)
        assert "m" in format_time(125)
        assert "h" in format_time(3700)

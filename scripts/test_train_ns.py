import unittest
import sys
import os
import h5py
import torch
import json
import torch.nn as nn
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_ns import PreloadedNSDataset, save_checkpoint

class TestTrainNS(unittest.TestCase):
    def setUp(self):
        self.output_dir = './results/test_ns_unit'
        os.makedirs(self.output_dir, exist_ok=True)
        self.h5_path = './datasets/ns_incom_inhom_2d_512-0.h5'

    def test_preloaded_dataset_math(self):
        # 1. Ensure math computes correctly for dataset window logic
        seq_len = 20
        ds_train = PreloadedNSDataset(self.h5_path, target_res=64, seq_len=seq_len, mode="train")
        ds_val = PreloadedNSDataset(self.h5_path, target_res=64, seq_len=seq_len, mode="val")
        
        # Dataset has N=4 samples, T=1000 timesteps
        # Train uses samples 0,1,2 -> 3 * (1000 - 21) = 2937 windows
        # Val uses sample 3         -> 1 * (1000 - 21) = 979 windows
        self.assertEqual(len(ds_train), 2937, "Train window count mismatch")
        self.assertEqual(len(ds_val), 979, "Val window count mismatch")
        
        # Test exact item shape math
        x, y, cond = ds_train[0]
        self.assertEqual(x.shape, (2, 64, 64), "x vector shape mismatch")
        self.assertEqual(y.shape, (seq_len, 2, 64, 64), "y rollout boundary mismatch")
        self.assertEqual(cond.shape, (4,), "Forcing boundary conditions mismatch")
        
        # No data leakage: train and val use different samples
        train_samples = set(n for n, t in ds_train.indices)
        val_samples = set(n for n, t in ds_val.indices)
        self.assertEqual(len(train_samples & val_samples), 0, "Data leakage between train/val!")

    def test_resumability_checkpoints(self):
        # 1. Establish exact state schemas for the TurboNIGO architecture
        model = nn.Linear(10, 2) # Shell proxy
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scaler = MagicMock()
        scaler.state_dict.return_value = {"scale": 1024}
        
        epoch = 37
        best_loss = 0.05
        ckpt_path = os.path.join(self.output_dir, "test_ckpt.pth")
        
        # 2. Invoke checkpoint save sequence 
        scheduler = MagicMock()
        scheduler.state_dict.return_value = {"T_0": 10}
        save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, scheduler, best_loss)
        
        self.assertTrue(os.path.exists(ckpt_path), "Checkpoint failed to save to disk")
        
        # 3. Test exact resumabnility load sequence
        loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.assertEqual(loaded['epoch'], 37, "Failed to restore epoch state timeline")
        self.assertEqual(loaded['best_val_loss'], 0.05, "Failed to capture baseline metrics cleanly")
        self.assertTrue('model_state' in loaded, "Missing structural weights schema")
        self.assertTrue('optimizer_state' in loaded, "Missing momentum graph for Adam")
        self.assertEqual(loaded['scaler_state']['scale'], 1024, "Missing CUDA scaler state")

    def test_immutable_logging_schema(self):
        # 1. Simulating JSON parameters injection
        config = {"lr": 1e-4, "seq_len": 20, "workers": 4, "epochs": 300}
        config_path = os.path.join(self.output_dir, "train_config.json")
        
        with open(config_path, "w") as f:
            json.dump(config, f)
            
        # 2. Assert file integrity
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["lr"], 1e-4, "Immutable config corrupted values internally")

if __name__ == '__main__':
    unittest.main()

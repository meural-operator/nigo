import torch
import numpy as np
from turbo_nigo.models import GlobalTurboNIGO

class Evaluator:
    """
    Provides robust inference and rollouts using a trained model.
    """
    def __init__(self, model: GlobalTurboNIGO, dt: float, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.dt = dt
        self.device = device

    def chained_block_rollout(self, initial_frame: torch.Tensor, cond: torch.Tensor, 
                              total_steps: int, block_size: int, 
                              g_min: float, g_max: float) -> np.ndarray:
        """
        Rolls out the model predictions autoregressively in blocks.
        Args:
            initial_frame: (1, 2, H, W)
            cond: (4,) un-batched
        Returns:
            full_pred: (total_steps+1, 2, H, W) Un-normalized numpy array
        """
        curr_state = (initial_frame - g_min) / (g_max - g_min + 1e-8)
        curr_state = curr_state.to(self.device)
        # Add batch dim if not present
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond = cond.to(self.device)
        
        block_time_steps = torch.arange(1, block_size + 1).float().to(self.device) * self.dt
        predictions = [curr_state.cpu()] 
        
        num_blocks = int(np.ceil(total_steps / block_size))
        
        with torch.no_grad():
            for _ in range(num_blocks):
                u_block, _, _, _, _, _ = self.model(curr_state, block_time_steps, cond)
                # u_block: (1, S, C, H, W)
                for t in range(u_block.shape[1]):
                    predictions.append(u_block[:, t].cpu())
                # Next input is the last predicted frame
                curr_state = u_block[:, -1] 
                
        # Stack to (len, B, C, H, W) -> (len, C, H, W)
        full_pred = torch.cat(predictions, dim=0).numpy()
        # Un-normalize
        full_pred = full_pred * (g_max - g_min + 1e-8) + g_min
        return full_pred[:total_steps+1]

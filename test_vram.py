import torch
import time
from turbo_nigo.models.turbo_nigo import GlobalTurboNIGO

def check_memory():
    print("Checking VRAM limits on High-Res GlobalTurboNIGO...")
    device = torch.device('cuda')
    model = GlobalTurboNIGO(
        latent_dim=64, num_bases=8, cond_dim=4, 
        width=64, spatial_size=256, in_channels=2,
        num_layers=3, use_residual=True, norm_type='group'
    ).to(device)
    
    b, s = 2, 20
    print(f"Testing Micro-Batch Size: {b} (sequences of length {s})")
    
    x = torch.randn(b, 2, 256, 256, device=device)
    cond = torch.randn(b, 4, device=device)
    time_steps = torch.linspace(0, 1, s, device=device)
    
    criterion = torch.nn.MSELoss()
    y = torch.randn(b, s, 2, 256, 256, device=device)
    
    start = time.time()
    try:
        with torch.amp.autocast('cuda', enabled=True):
            preds, _, _, _, _, _ = model(x, time_steps, cond)
            loss = criterion(preds, y)
        loss.backward()
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        mem_res = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"Success! Time: {time.time()-start:.2f}s")
        print(f"Peak VRAM: {mem_alloc:.2f} GB (Allocated), {mem_res:.2f} GB (Reserved)")
    except RuntimeError as e:
        print(f"OOM Error: {e}")

if __name__ == '__main__':
    check_memory()

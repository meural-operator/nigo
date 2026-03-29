import torch
import time
from turbo_nigo.models.generator import HyperTurbulentGenerator

def test_generator():
    print("Testing HyperTurbulentGenerator...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    latent_dim = 64
    num_bases = 8
    seq_len = 20
    batch_size = 16
    
    gen = HyperTurbulentGenerator(latent_dim, num_bases).to(device)
    
    z0 = torch.randn(batch_size, latent_dim, dtype=torch.complex64, device=device)
    time_steps = torch.linspace(0, 1.0, seq_len, device=device)
    
    k_coeffs = torch.randn(batch_size, num_bases, device=device)
    r_coeffs = torch.randn(batch_size, num_bases, device=device)
    alpha = torch.ones(batch_size, 1, device=device)
    beta = torch.ones(batch_size, 1, device=device)
    
    print("Running forward pass...")
    start = time.time()
    try:
        out = gen(z0, time_steps, k_coeffs, r_coeffs, alpha, beta)
        torch.cuda.synchronize()
        print(f"Forward pass completed in {time.time() - start:.4f}s")
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == '__main__':
    test_generator()

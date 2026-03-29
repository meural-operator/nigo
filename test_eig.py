import torch
import time

def test_eig():
    print("Testing Spectral Solver (Eigen) Approach...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    latent_dim = 64
    seq_len = 20
    batch_size = 16
    S = seq_len
    
    # Warmup
    A = torch.randn(batch_size, latent_dim, latent_dim, device=device)
    dt = 0.05
    z0 = torch.randn(batch_size, latent_dim, dtype=torch.complex64, device=device)
    
    A_f64 = (A * dt).to(torch.complex128)
    L, V = torch.linalg.eig(A_f64)
    V_inv = torch.linalg.inv(V)
    t_seq = torch.arange(1, S + 1, device=device, dtype=torch.complex128)
    L_evolved = torch.exp(L.unsqueeze(1) * t_seq.unsqueeze(0).unsqueeze(-1))
    z_c = z0.to(torch.complex128).unsqueeze(-1)
    z_in_v = torch.bmm(V_inv, z_c).transpose(1, 2)
    z_spectral = L_evolved * z_in_v
    z_evolved_c128 = torch.matmul(V.unsqueeze(1), z_spectral.unsqueeze(-1)).squeeze(-1)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):  # Run 10 times to get average
        A_f64 = (A * dt).to(torch.complex128)
        L, V = torch.linalg.eig(A_f64)
        V_inv = torch.linalg.inv(V)
        t_seq = torch.arange(1, S + 1, device=device, dtype=torch.complex128)
        L_evolved = torch.exp(L.unsqueeze(1) * t_seq.unsqueeze(0).unsqueeze(-1))
        z_c = z0.to(torch.complex128).unsqueeze(-1)
        z_in_v = torch.bmm(V_inv, z_c).transpose(1, 2)
        z_spectral = L_evolved * z_in_v
        z_evolved_c128 = torch.matmul(V.unsqueeze(1), z_spectral.unsqueeze(-1)).squeeze(-1)
    torch.cuda.synchronize()
    print(f"Spectral Solver took {(time.time() - start)/10:.4f}s per batch")

if __name__ == '__main__':
    test_eig()

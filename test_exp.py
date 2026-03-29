import torch
import time

def test_exp():
    print("Testing Matrix Exp Approach...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    latent_dim = 64
    seq_len = 20
    batch_size = 16
    
    # Warmup
    A = torch.randn(batch_size, latent_dim, latent_dim, device=device)
    dt = 0.05
    
    A_f64 = (A * dt).double()
    eps = 1e-6 * torch.eye(latent_dim, device=device, dtype=torch.float64)
    M = torch.linalg.matrix_exp(A_f64 + eps.unsqueeze(0))
    M_c64 = M.to(torch.complex64)
    props_list = [M_c64]
    for _ in range(1, seq_len):
        props_list.append(props_list[-1] @ M_c64)
    props = torch.stack(props_list, dim=1)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):  # Run 10 times to get average
        A_f64 = (A * dt).double()
        M = torch.linalg.matrix_exp(A_f64 + eps.unsqueeze(0))
        M_c64 = M.to(torch.complex64)
        props_list = [M_c64]
        for _ in range(1, seq_len):
            props_list.append(props_list[-1] @ M_c64)
        props = torch.stack(props_list, dim=1)
    torch.cuda.synchronize()
    print(f"Matrix Exp + Matmul Rollout took {(time.time() - start)/10:.4f}s per batch")

if __name__ == '__main__':
    test_exp()

import os
import sys
import h5py
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Research-Grade LaTeX compliant plotting configurations
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from turbo_nigo.models.turbo_nigo import GlobalTurboNIGO

def load_eval_data(h5_path, target_res=256, max_steps=400):
    """
    Loads exclusively the validation sample (last index) for full continuous rollout.
    """
    g_min, g_max = -3.0, 3.0
    print(f"[*] Extracting Validation Ground Truth from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        N = f['velocity'].shape[0]
        # Always use the very last sample for pure validation
        raw_vel = torch.from_numpy(f['velocity'][N-1]).float().permute(0, 3, 1, 2)
        
        # We only need up to max_steps
        raw_vel = raw_vel[:max_steps + 1]
        
        # Spatial Downsampling
        chunks = [F.interpolate(raw_vel[t0:t0+100], size=(target_res, target_res), mode="bilinear") for t0 in range(0, raw_vel.shape[0], 100)]
        down = torch.cat(chunks, dim=0)
        
        # Normalization (identical to training)
        u_true = (down - g_min) / (g_max - g_min + 1e-8)
        
        # Extract Forcing conditions
        force_raw = f['force'][N-1]
        f_u, f_v = force_raw[:, :, 0], force_raw[:, :, 1]
        cond_vec = torch.tensor([np.mean(f_u), np.std(f_u), np.mean(f_v), np.std(f_v)]) / 2.0
        
    return u_true.unsqueeze(0), cond_vec.unsqueeze(0) # (1, T, C, H, W) and (1, 4)

def plot_training_curves(run_dir, output_dir):
    csv_path = os.path.join(run_dir, 'history.csv')
    if not os.path.exists(csv_path):
        print("[!] No history.csv found. Skipping Loss Curve plot.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss (Rel L2)', lw=2, color='navy')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss (Rel L2)', lw=2, color='darkorange')
    
    # Highlight the Curriculum Switch
    if len(df) > 20:
        plt.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='Physics Constraints Activated')
        
    plt.yscale('log')
    plt.title('TurboNIGO Training Dynamics', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Composite Loss (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_training_curves.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '1_training_curves.pdf'), bbox_inches='tight')
    plt.close()

def compute_kinetic_energy(field):
    """ E = 0.5 * sum(u^2 + v^2) per spatial domain """
    u2 = field[..., 0, :, :] ** 2
    v2 = field[..., 1, :, :] ** 2
    return 0.5 * (u2 + v2).mean(dim=(-1, -2)) # Mean over H, W

def run_autoregressive_rollout(model, u_true, cond, device, seq_len=20, rollouts=10):
    """
    Chains predictions iteratively.
    u_true: (1, T, 2, 256, 256)
    """
    model.eval()
    
    u_pred_list = [u_true[:, 0]] # Start with ground truth initial condition
    u_curr = u_true[:, 0].to(device)
    cond = cond.to(device)
    
    time_steps = torch.linspace(0, 1.0, seq_len, device=device)
    
    print(f"[*] Starting Chained Rollout ({rollouts} blocks of {seq_len} steps = {seq_len*rollouts} total steps)...")
    with torch.no_grad():
        for b in range(rollouts):
            with torch.amp.autocast('cuda', enabled=False): # Full precision eval
                pred_block, _, _, _, _, _ = model(u_curr, time_steps, cond)
            
            # Extract sequence (1, seq_len, 2, 256, 256)
            for t in range(seq_len):
                u_pred_list.append(pred_block[:, t].cpu())
                
            # Autoregressive feed-forward: the last step becomes the new IC
            u_curr = pred_block[:, -1]
            print(f"    Rollout Block {b+1}/{rollouts} Completed.")
            
    # Stack into matching shape (1, T_eval, 2, 256, 256)
    u_pred_full = torch.stack(u_pred_list, dim=1)
    
    # Truncate true sequence to match prediction length
    T_eval = u_pred_full.shape[1]
    u_true_trunc = u_true[:, :T_eval]
    
    # Compute Frame-wise Relative L2 Error
    diff = u_pred_full - u_true_trunc
    l2_err = diff.norm(p=2, dim=(-1, -2, -3)) / (u_true_trunc.norm(p=2, dim=(-1, -2, -3)) + 1e-8)
    l2_err = l2_err.squeeze(0).numpy() # (T_eval,)
    
    # Compute Energies
    e_true = compute_kinetic_energy(u_true_trunc).squeeze(0).numpy()
    e_pred = compute_kinetic_energy(u_pred_full).squeeze(0).numpy()
    
    return u_true_trunc, u_pred_full, l2_err, e_true, e_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Path to RUN_XXXX directory')
    parser.add_argument('--data_path', type=str, default='./datasets/ns_incom_inhom_2d_512-0.h5')
    parser.add_argument('--rollouts', type=int, default=10, help='Number of 20-step blocks to chain')
    args = parser.parse_args()

    eval_dir = os.path.join(args.run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(args.run_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path} - check run_dir.")

    # 1. Plot Training History
    print("[1/4] Generating Training History Curves...")
    plot_training_curves(args.run_dir, eval_dir)

    # 2. Load Model & Data
    print("[2/4] Loading High-Resolution Deep Residual Checkpoint...")
    model = GlobalTurboNIGO(latent_dim=64, width=64, spatial_size=256, num_layers=3, use_residual=True, norm_type='group')
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    
    u_true, cond = load_eval_data(args.data_path, target_res=256, max_steps=args.rollouts * 20)
    
    # 3. Execute Autoregressive Cascade
    print("[3/4] Generating Long-Term Surrogates & Physical Traces...")
    u_true, u_pred, l2_err, e_true, e_pred = run_autoregressive_rollout(model, u_true, cond, device, rollouts=args.rollouts)
    
    # 4. Generate Research Vector Graphics
    print("[4/4] Serializing Output Graphics for LaTeX...")
    
    # A. Relative L2 Curve
    plt.figure(figsize=(10, 5))
    times = np.arange(len(l2_err))
    plt.plot(times, l2_err * 100, color='crimson', lw=2)
    plt.title('Autoregressive Relative $L_2$ Error Growth', fontweight='bold')
    plt.xlabel('Prediction Step (t)')
    plt.ylabel('Relative Error (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, '2_relative_l2_error.png'), dpi=300)
    plt.savefig(os.path.join(eval_dir, '2_relative_l2_error.pdf'), bbox_inches='tight')
    plt.close()
    
    # B. Kinetic Energy Trace
    plt.figure(figsize=(10, 5))
    plt.plot(times, e_true, 'k-', label='Ground Truth (FVM Simulator)', lw=2)
    plt.plot(times, e_pred, 'b--', label='TurboNIGO (Neural Surrogate)', lw=2)
    plt.title('Macroscopic Physical Consistency: Kinetic Energy Dissipation', fontweight='bold')
    plt.xlabel('Prediction Step (t)')
    plt.ylabel('Kinetic Energy $\mathcal{K}$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, '3_energy_trace.png'), dpi=300)
    plt.savefig(os.path.join(eval_dir, '3_energy_trace.pdf'), bbox_inches='tight')
    plt.close()
    
    # C. Absolute Spatial Error Contour (Midpoint and Endpoint)
    mid_t = len(times) // 2
    end_t = len(times) - 1
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, t_step in enumerate([mid_t, end_t]):
        gt_vel = u_true[0, t_step].numpy()
        pr_vel = u_pred[0, t_step].numpy()
        
        # Magnitude
        gt_mag = np.sqrt(gt_vel[0]**2 + gt_vel[1]**2)
        pr_mag = np.sqrt(pr_vel[0]**2 + pr_vel[1]**2)
        err_mag = np.abs(gt_mag - pr_mag)
        
        vmax = max(gt_mag.max(), pr_mag.max())
        
        im0 = axes[idx, 0].imshow(gt_mag, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
        axes[idx, 0].set_title(f"Ground Truth (t={t_step})")
        axes[idx, 0].axis('off')
        
        im1 = axes[idx, 1].imshow(pr_mag, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
        axes[idx, 1].set_title(f"TurboNIGO (t={t_step})")
        axes[idx, 1].axis('off')
        
        im2 = axes[idx, 2].imshow(err_mag, cmap='inferno', origin='lower')
        axes[idx, 2].set_title(f"Absolute Point-wise Error")
        axes[idx, 2].axis('off')
        fig.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)

    plt.suptitle('Long-Term Spatial Rollout Deviations', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, '4_longterm_spatial_error.png'), dpi=300)
    plt.savefig(os.path.join(eval_dir, '4_longterm_spatial_error.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\n[+] Success! Evaluation suite correctly compiled 4 publication-ready PNG/PDFs inside: \n    {eval_dir}")

if __name__ == '__main__':
    main()

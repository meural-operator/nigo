# TurboNIGO: Structure-Preserving Neural Operators via Lyapunov-Stable Latent Dynamics

> **Anonymous Submission — ICML 2026**
> This repository accompanies the paper *"TurboNIGO: Structure-Preserving Neural Operators via Lyapunov-Stable Latent Dynamics"* (under double-blind review). NIGO stands for **Neural Infinitesimal Generator Operator**.

---

## Abstract

We propose **TurboNIGO** (Turbulent **Neural Infinitesimal Generator Operator**), a structure-preserving neural operator framework that models temporal evolution as a continuous-time dynamical system on a latent manifold. The model parameterizes the infinitesimal generator as $A = \alpha(K - K^\top) - \beta R^\top R$, a sum of skew-symmetric (energy-conserving) and dissipative components, guaranteeing **Lyapunov stability** by construction. A physics-informed inference network maps initial conditions and conditioning parameters into local basis coefficients $\{k_b, r_b\}$ and adaptive scaling parameters $\alpha, \beta$, which are composed via matrix exponential evolution $z(t) = \exp(At) \cdot z_0$ to produce future states. A multi-scale temporal refiner module further enhances long-horizon fidelity.

**Key contributions:**
- Structurally-guaranteed Lyapunov stability via the decomposition $A = \alpha(K - K^\top) - \beta R^\top R$
- Physics-conditioned operator learning with zero-shot generalization across Reynolds numbers
- Comprehensive ablation study isolating the contribution of each architectural component

---

## Repository Structure

```
.
├── turbo_nigo/                     # Core framework
│   ├── configs/                    # YAML configuration files
│   │   ├── default_config.yaml     # Bluff-body cylinder flow config
│   │   └── ks_config.yaml          # Kuramoto-Sivashinsky config
│   ├── core/
│   │   └── trainer.py              # Research-grade training loop with full resumability
│   ├── data/
│   │   ├── base_dataset.py         # Abstract dataset interface
│   │   ├── flow_dataset.py         # In-memory 2D flow dataset (cylinder)
│   │   ├── h5_dataset.py           # HDF5 high-resolution dataset loader
│   │   └── analyzer.py             # Dataset visualization & statistics
│   ├── models/
│   │   ├── encoder.py              # Convolutional encoder
│   │   ├── decoder.py              # Transposed-conv decoder
│   │   ├── physics_net.py          # Physics inference network (basis coefficients)
│   │   ├── generator.py            # Hyper-turbulent Lie-algebraic generator
│   │   ├── refiner.py              # Multi-scale temporal refinement
│   │   ├── turbo_nigo.py           # Composite model (GlobalTurboNIGO)
│   │   ├── ablations/              # Ablation-specific subclasses
│   │   │   ├── generator_ablations.py
│   │   │   └── model_ablations.py
│   │   └── extensions/             # Advanced extensions (attention-based physics)
│   └── utils/
│       ├── logger.py               # Multi-backend experiment logger
│       ├── misc.py                 # Seeding, path utilities
│       └── registry.py             # Dataset/model registry
├── scripts/
│   ├── train.py                    # Standard training entry point
│   ├── evaluate.py                 # Checkpoint evaluation
│   ├── run_ablations.py            # Component ablation suite (cylinder flow)
│   ├── run_ablations_ks.py         # Component ablation suite (Kuramoto-Sivashinsky)
│   ├── run_sensitivity.py          # Dataset scale × horizon sensitivity analysis
│   ├── visualize_bc_dataset.py     # Publication figures for cylinder flow
│   └── visualize_ks_dataset.py     # Publication figures for KS equation
├── tests/                          # Unit & integration tests
│   ├── test_math_properties.py     # Skew-symmetry, energy boundedness, Lyapunov
│   ├── test_ablations.py           # Ablation variant correctness
│   ├── test_models.py              # Forward pass shape validation
│   ├── test_extensions.py          # Attention physics & dynamic resolution
│   ├── test_integration.py         # End-to-end pipeline tests
│   └── ...
└── .gitignore
```

---

## Requirements

```
Python >= 3.10
PyTorch >= 2.0
NumPy
SciPy
h5py
matplotlib
PyYAML
tqdm
tensorboard (optional)
```

**Setup:**
```bash
conda create -n cfd python=3.13
conda activate cfd
pip install torch torchvision numpy scipy h5py matplotlib pyyaml tqdm tensorboard pytest
```

---

## Datasets

This framework is evaluated on two benchmark systems of increasing complexity:

| Dataset | Type | Spatial | Temporal | Samples | Size |
|---------|------|---------|----------|---------|------|
| **Bluff-body Cylinder Flow** | 2D Navier-Stokes | 64×64, 2 channels ($u$, $v$) | 1000 steps | 50 cases | ~500 MB |
| **Kuramoto-Sivashinsky** | 1D chaotic PDE | 512 points | 768 steps | 40,000 trajectories | ~88 GB |

Place datasets in `./datasets/` following the structure:
```
datasets/
├── bc/                          # Cylinder flow
│   ├── case000/
│   │   ├── u.npy                # (T, H, W) velocity x-component
│   │   ├── v.npy                # (T, H, W) velocity y-component
│   │   └── meta.json            # {Re, radius, inlet_velocity, bc_type}
│   └── ...
└── KS_dataset/
    ├── KS_ML_DATASET.h5         # train: (40000, 768, 512), test: (10000, 768, 512)
    └── KS_GROUNDTRUTH.h5        # 9 long reference trajectories
```

---

## Training

**Standard training** (cylinder flow):
```bash
python scripts/train.py --config turbo_nigo/configs/default_config.yaml
```

**Resuming** from a checkpoint:
```bash
python scripts/train.py --config turbo_nigo/configs/default_config.yaml --resume_from results/checkpoints/ep050.pth
```

Key configuration flags in `default_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 300 | Training epochs |
| `batch_size` | 64 | Mini-batch size |
| `learning_rate` | 2e-4 | AdamW learning rate |
| `latent_dim` | 64 | Latent space dimension |
| `num_bases` | 8 | Number of Lie-algebra basis matrices $K_b$ |
| `scheduler` | cosine | LR schedule (cosine/plateau/step) |
| `tf32` | true | TensorFloat-32 acceleration |
| `use_amp` | true | Automatic mixed precision |

---

## Ablation Study

The ablation suite quantifies the contribution of each architectural component independently, without modifying the core framework code.

### Component Ablations

| Ablation | What Changes | Mathematical Effect |
|----------|-------------|-------------------|
| **Baseline** | Full TurboNIGO | $A = \alpha(K-K^\top) - \beta R^\top R$ |
| **No Skew (Abl. 1)** | Remove $K-K^\top$ | $A = -\beta R^\top R$ (purely dissipative) |
| **No Dissipative (Abl. 2)** | Remove $R^\top R$ | $A = \alpha(K-K^\top)$ (energy-conserving only) |
| **Dense Generator (Abl. 3)** | Bypass structured factorization | $A = \text{MLP}(z_0)$ (unconstrained) |
| **No Refiner (Abl. 4)** | Skip temporal refinement | $z_{\text{refined}} = z_{\text{base}}$ |
| **Unscaled Generator (Abl. 5)** | Force $\alpha=1, \beta=1$ | $A = (K-K^\top) - R^\top R$ (fixed scale) |

**Run ablations:**
```bash
# Cylinder flow
python scripts/run_ablations.py --config turbo_nigo/configs/default_config.yaml

# Kuramoto-Sivashinsky
python scripts/run_ablations_ks.py --config turbo_nigo/configs/ks_config.yaml
```

### Sensitivity Analysis

Grid search over dataset scale and prediction horizon:
```bash
python scripts/run_sensitivity.py --config turbo_nigo/configs/default_config.yaml
```

| Factor | Values Tested |
|--------|--------------|
| Dataset Scale $N$ | 10, 25, 50 cases |
| Temporal Horizon $T$ | 10, 20, 40 steps |

---

## Model Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Input u₀    │────▶│   Encoder E(·)   │────▶│   Latent z₀         │
│  (C, H, W)   │     │  Conv layers     │     │   (latent_dim,)     │
└──────────────┘     └──────────────────┘     └────────┬────────────┘
                                                       │
                           ┌───────────────────────────┘
                           ▼
               ┌───────────────────────┐
               │  PhysicsInferenceNet  │◀── Conditions c (Re, radius, ...)
               │  MLP → basis coeffs  │
               └───────────┬───────────┘
                           │ {c_b}
                           ▼
               ┌───────────────────────┐
               │  HyperTurbulentGen.   │    A = α(K-Kᵀ) - βRᵀR
               │  Infinitesimal Gen.   │    (skew-symm. + dissipative)
               │  z(t) = expm(At)·z₀  │    Lyapunov stable ✓
               └───────────┬───────────┘
                           │ z(t₁), ..., z(tₜ)
                           ▼
               ┌───────────────────────┐
               │  TemporalRefiner      │
               │  Multi-scale residual │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐     ┌──────────────┐
               │   Decoder D(·)        │────▶│  Output û(t) │
               │  Transposed conv      │     │  (T, C, H, W)│
               └───────────────────────┘     └──────────────┘
```

**Model size:** ~3.27M parameters (12.5 MB in FP32)

| Component | Parameters | Share |
|-----------|-----------|-------|
| Encoder | 1,290,400 | 39.5% |
| Decoder | 1,297,186 | 39.7% |
| Temporal Refiner | 590,720 | 18.1% |
| Generator | 65,536 | 2.0% |
| Conditioning Net | 26,576 | 0.8% |

---

## Visualization

Generate publication-quality figures (PNG + PDF):

```bash
# Cylinder flow: vorticity fields, energy spectra, KE evolution, etc.
python scripts/visualize_bc_dataset.py --output_dir ./figures/bc

# Kuramoto-Sivashinsky: x-t diagrams, spatial spectra, autocorrelation, etc.
python scripts/visualize_ks_dataset.py --output_dir ./figures/ks
```

---

## Testing

Run the full test suite to verify mathematical properties and pipeline integrity:

```bash
pytest tests/ -v
```

Key test categories:
- **`test_math_properties.py`** — Skew-symmetry of $L_s$, negative semi-definiteness of $L_d$, energy boundedness
- **`test_ablations.py`** — Forward pass correctness for all 5 ablation variants
- **`test_models.py`** — Output shape validation across all sub-modules
- **`test_integration.py`** — End-to-end training pipeline on synthetic data

---

## Compute Optimizations

The framework includes several optimizations for high-throughput training on modern GPUs:

| Optimization | Config Flag | Effect |
|-------------|-------------|--------|
| TensorFloat-32 | `tf32: true` | ~2× matmul speedup on Ampere/Ada |
| cuDNN Benchmark | `cudnn_benchmark: true` | Auto-tuned conv kernels |
| Mixed Precision | `use_amp: true` | FP16 forward, FP32 gradients |
| Persistent Workers | Built-in | Zero inter-epoch data-loader overhead |
| `torch.compile()` | `compile: true` | Graph-mode optimization (Linux/WSL) |

---

## Citation

```
Under double-blind review. Citation will be provided upon acceptance.
```

---

## License

This code is provided for **anonymous peer review only**. Redistribution is not permitted during the review period.

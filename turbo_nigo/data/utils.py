import os
import glob
import json
import numpy as np
from tqdm import tqdm
from typing import Tuple

def read_meta(case_dir: str) -> dict:
    """Reads metadata associated with a specific case/simulation."""
    mpath = os.path.join(case_dir, "case.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, 'r') as f: return json.load(f)
        except: return {}
    mpath_alt = os.path.join(case_dir, "meta.json")
    if os.path.exists(mpath_alt):
         try:
            with open(mpath_alt, 'r') as f: return json.load(f)
         except: return {}
    return {}

def compute_global_stats_and_cond_stats(root_dir: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Scans the dataset to compute global min/max for the fields (u, v) and
    mean/std for the conditioning variables.
    """
    case_dirs = sorted(glob.glob(os.path.join(root_dir, "case*")))
    if not case_dirs:
        raise FileNotFoundError(f"No cases found in {root_dir}")
    
    g_min, g_max = float('inf'), float('-inf')
    conds = []
    
    # We only scan a subset for speed if dataset is too large, 
    # but here we follow original robust implementation computing on all viable
    for c in tqdm(case_dirs, desc="Scanning Dataset"):
        try:
            u_path = os.path.join(c, "u.npy")
            v_path = os.path.join(c, "v.npy")
            if not (os.path.exists(u_path) and os.path.exists(v_path)):
                continue

            u = np.load(u_path)
            v = np.load(v_path)
            
            curr_min = min(float(np.min(u)), float(np.min(v)))
            curr_max = max(float(np.max(u)), float(np.max(v)))
            
            g_min = min(g_min, curr_min)
            g_max = max(g_max, curr_max)
            
            meta = read_meta(c)
            conds.append([
                float(meta.get("Re", 0.0)), 
                float(meta.get("radius", 0.0)), 
                float(meta.get("inlet_velocity", 0.0)), 
                float(meta.get("bc_type", 0.0))
            ])
        except Exception:
            pass

    conds = np.array(conds) if len(conds) > 0 else np.zeros((0,4))
    
    if conds.shape[0] == 0: 
        cond_mean = np.zeros(4, dtype=np.float32)
        cond_std = np.ones(4, dtype=np.float32)
    else: 
        cond_mean = conds.mean(axis=0).astype(np.float32)
        cond_std = (conds.std(axis=0) + 1e-8).astype(np.float32)
        
    return g_min, g_max, cond_mean, cond_std

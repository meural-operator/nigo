"""
CLI Tool for Universal Flexible 3D Data Queries ($O(1)$ memory).

Leverages the `Base3DDataset` query interface to extract single 2D slices or 
3D volumetric snapshots directly from multi-gigabyte data lakes instantly.
"""
import argparse
import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

# Insert root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from turbo_nigo.utils.registry import Registry
from turbo_nigo.utils.visualization import Visualizer3D

def main():
    parser = argparse.ArgumentParser(description="TurboNIGO 3D Universal Query CLI")
    
    # Dataset Core targeting
    parser.add_argument("--dataset", type=str, required=True, help="Registered dataset name (e.g. 'ns3d')")
    parser.add_argument("--path", type=str, required=True, help="Root path to the dataset folder")
    
    # Query Paradigm
    parser.add_argument("--query", type=str, choices=['slice', 'volume'], required=True, 
                        help="The strict lazy query structure you want to execute.")
    
    # Query Coordinates
    parser.add_argument("--sample", type=int, default=0, help="Sample index within the massive trajectory stack.")
    parser.add_argument("--time", type=int, default=0, help="Exact temporal snapshot.")
    parser.add_argument("--channel", type=str, required=True, help="Specific physical channel string (e.g., 'pressure')")
    parser.add_argument("--plane", type=str, default='z', choices=['x', 'y', 'z'], help="Orthogonal projection plane (For 'slice')")
    parser.add_argument("--slice_idx", type=int, default=32, help="Depth array cutoff (For 'slice')")
    
    # Output Control
    parser.add_argument("--output", type=str, default="query_output", help="Base filename generated in results dir.")
    parser.add_argument("--verbose_plot", action="store_true", help="Enable to generate titles, colorbars, grids, and axis labels. If omitted, pure geometry is generated.")

    args = parser.parse_args()
    
    # Retrieve dataset natively via the Central Registry
    if args.dataset not in Registry.list_datasets():
        print(f"[!] Target dataset '{args.dataset}' is entirely unregistered.")
        sys.exit(1)
        
    ds_class = Registry.get_dataset(args.dataset)
    print(f"[*] Booting {args.dataset} interface over path {args.path}...")
    dataset = ds_class(root_dir=args.path, seq_len=1, mode='test')  # seq_len=1 minimal overhead setup
    
    # Hook our highly structural visualizer component specifically onto the query-interface
    vis = Visualizer3D(dataset)
    
    # Automatically generate a secure identifying signature based on query parameters
    if args.query == 'slice':
        query_sig = f"slice_S{args.sample}_T{args.time}_C_{args.channel}_{args.plane.upper()}{args.slice_idx}_V{int(args.verbose_plot)}"
    else:
        query_sig = f"volume_S{args.sample}_T{args.time}_C_{args.channel}_V{int(args.verbose_plot)}"
        
    # Ensure robust dataset-isolated storage directories to avoid cross-contamination
    # Group the three structural formats natively using the strict query parameter signature
    out_dir = Path(f"results/dynamic_queries/{args.dataset}/{query_sig}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / query_sig)

    # Check if the primary output already exists (Caching mechanism)
    if Path(out_path + ".png").exists():
        print(f"[*] Cache Hit: Requested query already parsed and available at {out_path}.png. Skipping computation.")
        sys.exit(0)

    try:
        if args.query == 'slice':
            print(f"[*] Firing strict 2D planar query over Plane {args.plane.upper()}[{args.slice_idx}]...")
            vis.render_slice_query(sample_idx=args.sample, channel=args.channel, 
                                   time_step=args.time, plane=args.plane, 
                                   slice_idx=args.slice_idx, save_path=out_path,
                                   verbose_plot=args.verbose_plot)
            
        elif args.query == 'volume':
            print(f"[*] Loading isolated global 3D topology map...")
            vis.render_volume_query(sample_idx=args.sample, channel=args.channel, 
                                    time_step=args.time, save_path=out_path,
                                    verbose_plot=args.verbose_plot)
                                    
        print(f"\n[+] Query Execution Success. Delivered directly at {out_path} [.png|.pdf|.tex]")
        
    except Exception as e:
        print(f"\n[!] Critical Query Execution Runtime Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

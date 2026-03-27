import yaml
import os
import argparse

def load_config(config_path: str = None) -> dict:
    """Loads configuration from a YAML file. 
    If no path is provided, attempts to load default_config.yaml."""
    
    if config_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "default_config.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

def get_args_and_config() -> dict:
    """Parses command line arguments and merges with YAML config."""
    parser = argparse.ArgumentParser(description="TurboNIGO Training Framework")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    # Allow overriding critical parameters
    parser.add_argument("--data_root", type=str, default=None, help="Path to datasets folder")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to results folder")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load base config
    config = load_config(args.config)
    
    # Override with command line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key != "config":
            config[key] = value
            
    return config

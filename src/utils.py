# src/utils.py

import yaml
import random
import numpy as np
import torch
import logging
import os
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The configuration as a dictionary.
        
    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file: {config_path}\n{e}")
        raise

def set_seeds(seed: Optional[int] = None):
    """
    Sets random seeds for reproducibility.

    Args:
        seed (Optional[int]): The seed value. If None, seeds are not set.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # Potentially add for deterministic algorithm behavior (can impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        logging.info(f"Random seeds set to: {seed}")
    else:
        logging.info("Random seeds not set (seed is None).")

def setup_logging(
    log_level: int = logging.INFO, 
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> None:
    """
    Configures the Python logging module.

    Args:
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (Optional[str]): Path to a file to save logs. If None, logs are not saved to a file.
        log_to_console (bool): Whether to output logs to the console.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = []
    if log_to_console:
        handlers.append(logging.StreamHandler())
    
    if log_file:
        ensure_dir(os.path.dirname(log_file)) # Ensure log directory exists
        handlers.append(logging.FileHandler(log_file))
        
    if not handlers: # Ensure at least one handler if both are false, default to console
        handlers.append(logging.StreamHandler())
        
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format, handlers=handlers)
    logging.info("Logging configured.")
    if log_file:
        logging.info(f"Logging to file: {log_file}")

def get_device(device_config: str = "auto") -> torch.device:
    """
    Determines and returns the torch.device based on the configuration.

    Args:
        device_config (str): Device configuration string ('auto', 'cuda', 'cuda:0', 'cpu').

    Returns:
        torch.device: The selected torch device.
    """
    if device_config.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif "cuda" in device_config.lower():
        if not torch.cuda.is_available():
            logging.warning("CUDA specified but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(device_config)
    else:
        device = torch.device("cpu")
    
    logging.info(f"Using device: {device}")
    return device

def ensure_dir(dir_path: str) -> None:
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        dir_path (str): The path to the directory.
    """
    if dir_path: # Ensure dir_path is not empty or None
        os.makedirs(dir_path, exist_ok=True)

# --- How these will be used (conceptual, in main.py or elsewhere) ---
if __name__ == "__main__":
    # This is just for demonstration.
    # In practice, these calls would be in your main training script.

    # 1. Setup basic logging (before loading config, to catch errors during loading)
    setup_logging() # Call this as early as possible

    # 2. Load configuration
    # Assume you have a config file path, perhaps from argparse
    config_file_path = "../configs/colored_mnist_default.yaml" # Adjust path as needed for testing
    if not os.path.exists(config_file_path):
         # Create a dummy config for demonstration if it doesn't exist
        dummy_config_content = """
project:
  name: "DemoProject"
  experiment_name: "demo_run"
  seed: 123
training:
  device: "auto"
logging:
  output_dir: "./results_demo"
  use_experiment_subfolder: true
  log_filename: "experiment.log" # Example: add a log filename to config
"""
        ensure_dir(os.path.dirname(config_file_path))
        with open(config_file_path, 'w') as f:
            f.write(dummy_config_content)
        logging.info(f"Created dummy config for demo: {config_file_path}")
        
    config = load_config(config_file_path)

    # 3. Re-setup logging if config has specific file settings
    # Construct log file path based on config
    log_file_path = None
    if config.get('logging') and config['logging'].get('output_dir'):
        output_dir = config['logging']['output_dir']
        if config['logging'].get('use_experiment_subfolder') and config.get('project', {}).get('experiment_name'):
            output_dir = os.path.join(output_dir, config['project']['experiment_name'])
        ensure_dir(output_dir) # Ensure the experiment specific output directory exists
        
        log_filename = config['logging'].get('log_filename', 'run.log') # Get log filename from config or use default
        log_file_path = os.path.join(output_dir, log_filename)
    
    setup_logging(log_file=log_file_path) # Re-init with file handler if specified

    # 4. Set seeds
    set_seeds(config.get('project', {}).get('seed'))

    # 5. Get device
    device = get_device(config.get('training', {}).get('device', 'auto'))
    
    logging.info("utils.py demonstration finished.")
    logging.info(f"Selected device in demo: {device}")
    if log_file_path:
        logging.info(f"Check logs at: {log_file_path}")
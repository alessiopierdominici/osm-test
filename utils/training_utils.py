"""
Shared training utilities for CLIP training scripts.
Contains common functions used across different training configurations.
"""

import torch
import yaml
import logging
from torch.optim import Adafactor


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_level):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def setup_optimizer(model, config):
    """Initialize optimizer based on configuration."""
    optimizer_name = config['training']['optimizer']
    lr = float(config['training']['learning_rate'])
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adafactor":
        if weight_decay is not None:
            return Adafactor(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return Adafactor(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")


def setup_device(config):
    """Setup computation device."""
    cuda_device = config['device']['cuda_device']
    if cuda_device == -1:
        return torch.device("cpu")
    return torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")


def calculate_effective_batch_size(config):
    """Calculate effective batch size considering gradient accumulation."""
    batch_size = config['training']['batch_size']
    if config['gradient_accumulation']['enabled']:
        return batch_size * config['gradient_accumulation']['steps']
    return batch_size


def validate_common_config(config):
    """Validate common configuration parameters."""
    # Check if required sections exist
    required_sections = ['model', 'training', 'device', 'logging', 'checkpoints']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate optimizer choice
    valid_optimizers = ["Adam", "AdamW", "Adafactor"]
    if config['training']['optimizer'] not in valid_optimizers:
        raise ValueError(f"Invalid optimizer. Choose from: {valid_optimizers}")
    
    # Validate device configuration
    if not isinstance(config['device']['cuda_device'], int):
        raise ValueError("cuda_device must be an integer (-1 for CPU, >= 0 for GPU)")


def log_training_config(config, script_name):
    """Log training configuration in a clean format."""
    effective_batch_size = calculate_effective_batch_size(config)
    
    logging.info("=" * 60)
    logging.info(f"{script_name.upper()} CONFIGURATION")
    logging.info("=" * 60)
    logging.info(f"Model: {config['model'].get('name', 'Custom')}")
    logging.info(f"Learning Rate: {config['training']['learning_rate']}")
    logging.info(f"Optimizer: {config['training']['optimizer']}")
    logging.info(f"Epochs: {config['training']['epochs']}")
    logging.info(f"Batch Size: {config['training']['batch_size']}")
    if config['gradient_accumulation']['enabled']:
        logging.info(f"Gradient Accumulation: {config['gradient_accumulation']['steps']} steps")
        logging.info(f"Effective Batch Size: {effective_batch_size}")
    logging.info(f"Device: CUDA {config['device']['cuda_device']}" if config['device']['cuda_device'] >= 0 else "CPU")
    logging.info(f"Seed: {config['training']['seed']}")
    logging.info("=" * 60)
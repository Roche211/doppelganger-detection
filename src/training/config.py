"""
Training Configuration

SIMPLE CONFIGURATION - NO OVERCOMPLICATION
Keep hyperparameters simple and well-documented
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Simple training configuration
    
    SENIOR ML ENGINEER APPROACH:
    - Start with reasonable defaults
    - Document the reasoning behind each choice
    - Make it easy to modify for experimentation
    """
    
    # Model Configuration
    embedding_dim: int = 512
    freeze_backbone: bool = True  # Start conservative, unfreeze later
    dropout_rate: float = 0.1
    
    # Training Configuration  
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss Configuration
    triplet_margin: float = 0.5
    loss_type: str = 'triplet'  # 'triplet' or 'contrastive'
    
    # Data Configuration
    train_ratio: float = 0.8
    num_workers: int = 4
    
    # Optimization Configuration
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'none'
    warmup_epochs: int = 5
    
    # Validation Configuration
    val_frequency: int = 1  # Validate every N epochs
    save_frequency: int = 5  # Save checkpoint every N epochs
    
    # Paths
    data_dir: str = "/workspace/data/processed"
    output_dir: str = "/workspace/models/checkpoints"
    pretrained_dir: str = "/workspace/models/pretrained"
    
    # Logging
    log_frequency: int = 50  # Log every N batches
    save_best_only: bool = True
    
    # Device
    device: Optional[str] = None  # Auto-detect if None
    mixed_precision: bool = True  # Use automatic mixed precision
    
    # Progressive Training
    unfreeze_epoch: int = 20  # Epoch to unfreeze backbone
    
    def __post_init__(self):
        """Validate configuration and set derived values"""
        
        # Auto-detect device
        if self.device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.pretrained_dir, exist_ok=True)
        
        # Validate paths
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for saving"""
        return {
            'model': {
                'embedding_dim': self.embedding_dim,
                'freeze_backbone': self.freeze_backbone,
                'dropout_rate': self.dropout_rate
            },
            'training': {
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
                'warmup_epochs': self.warmup_epochs
            },
            'loss': {
                'triplet_margin': self.triplet_margin,
                'loss_type': self.loss_type
            },
            'data': {
                'train_ratio': self.train_ratio,
                'num_workers': self.num_workers
            },
            'paths': {
                'data_dir': self.data_dir,
                'output_dir': self.output_dir,
                'pretrained_dir': self.pretrained_dir
            },
            'device': self.device,
            'mixed_precision': self.mixed_precision
        }


# Default configuration
DEFAULT_CONFIG = TrainingConfig()


def get_config(config_overrides: Optional[dict] = None) -> TrainingConfig:
    """
    Get training configuration with optional overrides
    
    Args:
        config_overrides: Dictionary of config values to override
    
    Returns:
        Training configuration
    """
    config = TrainingConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown config key: {key}")
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Default configuration:")
    print(config.to_dict())
    
    # Test with overrides
    overrides = {
        'batch_size': 64,
        'learning_rate': 2e-4,
        'num_epochs': 100
    }
    
    custom_config = get_config(overrides)
    print("\nCustom configuration:")
    print(f"Batch size: {custom_config.batch_size}")
    print(f"Learning rate: {custom_config.learning_rate}")
    print(f"Epochs: {custom_config.num_epochs}")

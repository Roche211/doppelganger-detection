"""
Main Training Script - Simple and Clear

CRITICAL IMPLEMENTATION:
- Simple training loop following exact technical plan
- Pre-trained ArcFace ResNet-100 fine-tuning
- Progressive training (frozen -> unfrozen backbone)
- Proper validation and checkpointing
- Clear logging and monitoring

NO OVERCOMPLICATION - JUST GET IT WORKING
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Import our modules
import sys
sys.path.append('/workspace/src')

from models.similarity_arcface import setup_similarity_model
from models.losses import TripletLoss, compute_triplet_accuracy
from data.dataset import get_data_loaders
from training.config import TrainingConfig, get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimilarityTrainer:
    """
    Simple trainer for similarity learning - NO OVERENGINEERING
    
    FOCUS: Just get the basic similarity learning working
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"🚀 Initializing Similarity Trainer")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Mixed precision: {config.mixed_precision}")
        
        # Initialize model
        self.model = setup_similarity_model(
            device=str(self.device),
            embedding_dim=config.embedding_dim,
            freeze_backbone=config.freeze_backbone
        )
        
        # Initialize loss function
        self.criterion = TripletLoss(margin=config.triplet_margin)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epochs': []
        }
        
        logger.info("✅ Trainer initialized successfully")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on config"""
        
        if self.config.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        logger.info(f"📊 Optimizer: {self.config.optimizer} (lr={self.config.learning_rate})")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        
        if self.config.scheduler.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
        
        if scheduler:
            logger.info(f"📊 Scheduler: {self.config.scheduler}")
        
        return scheduler
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            # We will ignore provided random negative and mine in-batch semi-hard negatives
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            
            self.optimizer.zero_grad()

            # Encode anchors and positives
            if self.scaler is not None:
                with autocast():
                    anchor_emb = self.model(anchor)
                    positive_emb = self.model(positive)
                    
                    # Build candidate negatives: all embeddings in batch except own positive and anchor
                    # Concatenate all embeddings to form pool
                    pool_emb = torch.cat([anchor_emb, positive_emb], dim=0)  # (2B, D)
                    B = anchor_emb.shape[0]
                    
                    # For each i in [0..B-1], compute semi-hard negative from pool indices [0..2B-1] \ {i, i+B}
                    # distances
                    d_pos = F.pairwise_distance(anchor_emb, positive_emb, p=2)  # (B,)
                    # distances from each anchor to all in pool
                    d_pool = torch.cdist(anchor_emb, pool_emb, p=2)  # (B, 2B)
                    
                    # mask out self and mate
                    mask = torch.ones_like(d_pool, dtype=torch.bool)
                    ar = torch.arange(B, device=d_pool.device)
                    mask[ar, ar] = False
                    mask[ar, ar + B] = False
                    d_masked = d_pool.masked_fill(~mask, float('inf'))
                    
                    # semi-hard selection: choose j with d_pos[i] < d_pool[i,j] < d_pos[i]+margin; else hardest
                    margin = getattr(self.criterion, 'margin', 0.5)
                    semi_mask = (d_masked > d_pos.unsqueeze(1)) & (d_masked < (d_pos + margin).unsqueeze(1))
                    # default to minimal d among masked (hardest negative greater than positive if no semi-hard)
                    # pick indices
                    neg_idx = []
                    for i in range(B):
                        if semi_mask[i].any():
                            # pick the max within semi-hard band (hardest semi-hard)
                            vals = d_masked[i]
                            chosen = torch.argmax(vals.masked_fill(~semi_mask[i], -float('inf'))).item()
                        else:
                            # pick minimal valid distance > positive distance
                            vals = d_masked[i]
                            chosen = torch.argmin(vals).item()
                        neg_idx.append(chosen)
                    neg_idx = torch.tensor(neg_idx, device=pool_emb.device)
                    negative_emb = pool_emb[neg_idx]
                    
                    loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                pool_emb = torch.cat([anchor_emb, positive_emb], dim=0)
                B = anchor_emb.shape[0]
                d_pos = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                d_pool = torch.cdist(anchor_emb, pool_emb, p=2)
                mask = torch.ones_like(d_pool, dtype=torch.bool)
                ar = torch.arange(B, device=d_pool.device)
                mask[ar, ar] = False
                mask[ar, ar + B] = False
                d_masked = d_pool.masked_fill(~mask, float('inf'))
                margin = getattr(self.criterion, 'margin', 0.5)
                semi_mask = (d_masked > d_pos.unsqueeze(1)) & (d_masked < (d_pos + margin).unsqueeze(1))
                neg_idx = []
                for i in range(B):
                    if semi_mask[i].any():
                        vals = d_masked[i]
                        chosen = torch.argmax(vals.masked_fill(~semi_mask[i], -float('inf'))).item()
                    else:
                        vals = d_masked[i]
                        chosen = torch.argmin(vals).item()
                    neg_idx.append(chosen)
                neg_idx = torch.tensor(neg_idx, device=pool_emb.device)
                negative_emb = pool_emb[neg_idx]
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                loss.backward()
                self.optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                accuracy = compute_triplet_accuracy(anchor_emb, positive_emb, negative_emb)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            
            # Log progress
            if batch_idx % self.config.log_frequency == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs}, "
                    f"Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Acc: {accuracy:.4f}"
                )
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                pool_emb = torch.cat([anchor_emb, positive_emb], dim=0)
                B = anchor_emb.shape[0]
                d_pos = F.pairwise_distance(anchor_emb, positive_emb, p=2)
                d_pool = torch.cdist(anchor_emb, pool_emb, p=2)
                mask = torch.ones_like(d_pool, dtype=torch.bool)
                ar = torch.arange(B, device=d_pool.device)
                mask[ar, ar] = False
                mask[ar, ar + B] = False
                d_masked = d_pool.masked_fill(~mask, float('inf'))
                margin = getattr(self.criterion, 'margin', 0.5)
                semi_mask = (d_masked > d_pos.unsqueeze(1)) & (d_masked < (d_pos + margin).unsqueeze(1))
                neg_idx = []
                for i in range(B):
                    if semi_mask[i].any():
                        vals = d_masked[i]
                        chosen = torch.argmax(vals.masked_fill(~semi_mask[i], -float('inf'))).item()
                    else:
                        vals = d_masked[i]
                        chosen = torch.argmin(vals).item()
                    neg_idx.append(chosen)
                neg_idx = torch.tensor(neg_idx, device=pool_emb.device)
                negative_emb = pool_emb[neg_idx]
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                accuracy = compute_triplet_accuracy(anchor_emb, positive_emb, negative_emb)
                
                total_loss += loss.item()
                total_accuracy += accuracy
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.output_dir) / "best_similarity_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 New best model saved: {best_path}")
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"🔄 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        logger.info(f"✅ Checkpoint loaded. Resuming from epoch {self.start_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info(f"🚀 Starting training for {self.config.num_epochs} epochs...")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Progressive training: unfreeze backbone after specified epoch
            if epoch == self.config.unfreeze_epoch and self.model.backbone_frozen:
                logger.info(f"🔓 Unfreezing backbone at epoch {epoch + 1}")
                self.model.unfreeze_backbone()
                
                # Optionally reduce learning rate when unfreezing
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                logger.info(f"📉 Reduced learning rate to {param_group['lr']}")
            
            # Training phase
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            if epoch % self.config.val_frequency == 0:
                val_loss, val_accuracy = self.validate_epoch(val_loader)
            else:
                val_loss, val_accuracy = 0.0, 0.0
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update history
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logger.info(f"✅ Epoch {epoch+1}/{self.config.num_epochs} complete ({epoch_time:.1f}s):")
            logger.info(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            if val_loss > 0:
                logger.info(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss if val_loss > 0 else False
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config.save_frequency == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        total_time = time.time() - start_time
        logger.info(f"🎯 Training complete! Total time: {total_time/3600:.1f}h")
        logger.info(f"   Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(self.config.num_epochs - 1, is_best=False)
        
        # Save training history
        history_path = Path(self.config.output_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"📊 Training history saved: {history_path}")


def main(config_overrides: Optional[Dict] = None):
    """
    Main training function
    
    Args:
        config_overrides: Optional config overrides
    """
    # Load configuration
    config = get_config(config_overrides)
    logger.info("📋 Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"   {key}: {value}")
    
    # Create data loaders
    logger.info("📚 Creating data loaders...")
    train_loader, val_loader = get_data_loaders(
        processed_data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_ratio=config.train_ratio
    )
    
    # Initialize trainer
    trainer = SimilarityTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    # Example usage
    logger.info("🧪 Starting similarity learning training...")
    
    # You can override config here
    config_overrides = {
        'batch_size': 16,  # Smaller batch size for testing
        'num_epochs': 10,   # Fewer epochs for testing
        'log_frequency': 10
    }
    
    main(config_overrides)

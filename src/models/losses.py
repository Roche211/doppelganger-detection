"""
Loss Functions for Similarity Learning

CRITICAL IMPLEMENTATION:
- Simple triplet loss for similarity learning
- Contrastive loss as alternative
- Proper validation and logging
- Handle edge cases gracefully

OBJECTIVE: 
- Minimize distance between similar faces (anchor, positive)
- Maximize distance between dissimilar faces (anchor, negative)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, 
                margin: float = 0.5, reduction: str = 'mean') -> torch.Tensor:
    """
    Triplet loss for similarity learning
    
    OBJECTIVE: 
    - Minimize distance between similar faces (anchor, positive)
    - Maximize distance between dissimilar faces (anchor, negative)
    
    SENIOR ML ENGINEER CONSIDERATIONS:
    1. Use appropriate distance metric (L2 for normalized embeddings)
    2. Handle edge cases (perfect matches, identical negatives)
    3. Add numerical stability
    4. Log loss components for debugging
    5. Consider hard negative mining
    
    Args:
        anchor: Anchor embeddings (N, embedding_dim)
        positive: Positive embeddings (N, embedding_dim) - similar to anchor
        negative: Negative embeddings (N, embedding_dim) - dissimilar to anchor
        margin: Margin for triplet loss
        reduction: Loss reduction ('mean', 'sum', 'none')
    
    Returns:
        Triplet loss value
    """
    # VALIDATION: Check input shapes match
    if not (anchor.shape == positive.shape == negative.shape):
        raise ValueError(f"Shape mismatch: anchor {anchor.shape}, positive {positive.shape}, negative {negative.shape}")
    
    # Compute distances (using L2 distance for normalized embeddings)
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    
    # ENGINEERING NOTE: Since embeddings are L2-normalized, 
    # L2 distance is related to cosine similarity: d = sqrt(2 - 2*cos_sim)
    
    # Triplet loss: we want pos_dist < neg_dist
    # Loss = max(0, pos_dist - neg_dist + margin)
    loss_per_sample = F.relu(pos_dist - neg_dist + margin)
    
    # Calculate statistics for monitoring
    with torch.no_grad():
        num_active = (loss_per_sample > 0).sum().item()
        avg_pos_dist = pos_dist.mean().item()
        avg_neg_dist = neg_dist.mean().item()
        
        # DEBUGGING INFO: Log loss components
        if num_active > 0:
            logger.debug(f"Triplet loss stats:")
            logger.debug(f"  Active triplets: {num_active}/{len(loss_per_sample)}")
            logger.debug(f"  Avg positive distance: {avg_pos_dist:.4f}")
            logger.debug(f"  Avg negative distance: {avg_neg_dist:.4f}")
            logger.debug(f"  Margin: {margin}")
    
    # Apply reduction
    if reduction == 'mean':
        return loss_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum()
    elif reduction == 'none':
        return loss_per_sample
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def contrastive_loss(anchor: torch.Tensor, other: torch.Tensor, labels: torch.Tensor,
                    margin: float = 0.5, reduction: str = 'mean') -> torch.Tensor:
    """
    Contrastive loss formulation
    
    ENGINEERING CHOICE: Sometimes contrastive loss is more stable
    This pulls similar pairs together and pushes dissimilar pairs apart
    
    Args:
        anchor: Anchor embeddings (N, embedding_dim)
        other: Other embeddings (N, embedding_dim)
        labels: Binary labels (N,) - 1 for similar, 0 for dissimilar
        margin: Margin for dissimilar pairs
        reduction: Loss reduction ('mean', 'sum', 'none')
    
    Returns:
        Contrastive loss value
    """
    # VALIDATION: Check input shapes
    if anchor.shape != other.shape:
        raise ValueError(f"Shape mismatch: anchor {anchor.shape}, other {other.shape}")
    
    if labels.shape[0] != anchor.shape[0]:
        raise ValueError(f"Labels shape {labels.shape} doesn't match embeddings {anchor.shape[0]}")
    
    # Compute distances
    distances = F.pairwise_distance(anchor, other, p=2)
    
    # Contrastive loss
    # For similar pairs (label=1): minimize distance
    # For dissimilar pairs (label=0): maximize distance up to margin
    pos_loss = labels.float() * distances.pow(2)
    neg_loss = (1 - labels.float()) * F.relu(margin - distances).pow(2)
    
    loss_per_sample = pos_loss + neg_loss
    
    # Apply reduction
    if reduction == 'mean':
        return loss_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum()
    elif reduction == 'none':
        return loss_per_sample
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class TripletLoss(nn.Module):
    """
    Triplet Loss Module with additional features
    """
    
    def __init__(self, margin: float = 0.5, hard_mining: bool = False):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for triplet loss
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
        
        Returns:
            Loss value
        """
        if self.hard_mining:
            return self._hard_triplet_loss(anchor, positive, negative)
        else:
            return triplet_loss(anchor, positive, negative, margin=self.margin)
    
    def _hard_triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, 
                          negative: torch.Tensor) -> torch.Tensor:
        """
        Hard negative mining triplet loss
        
        This is more advanced - for now, fall back to standard triplet loss
        """
        logger.warning("Hard negative mining not implemented yet, using standard triplet loss")
        return triplet_loss(anchor, positive, negative, margin=self.margin)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss Module
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, other: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for contrastive loss
        
        Args:
            anchor: Anchor embeddings
            other: Other embeddings
            labels: Binary similarity labels
        
        Returns:
            Loss value
        """
        return contrastive_loss(anchor, other, labels, margin=self.margin)


def compute_triplet_accuracy(anchor: torch.Tensor, positive: torch.Tensor, 
                           negative: torch.Tensor) -> float:
    """
    Compute accuracy for triplet loss (fraction where pos_dist < neg_dist)
    
    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings  
        negative: Negative embeddings
    
    Returns:
        Accuracy as fraction
    """
    with torch.no_grad():
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        correct = (pos_dist < neg_dist).float()
        accuracy = correct.mean().item()
        
        return accuracy


def compute_similarity_metrics(embeddings1: torch.Tensor, embeddings2: torch.Tensor, 
                             labels: torch.Tensor) -> dict:
    """
    Compute various similarity metrics
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        labels: Ground truth similarity labels (1 for similar, 0 for dissimilar)
    
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Compute cosine similarities (since embeddings are normalized)
        similarities = F.cosine_similarity(embeddings1, embeddings2, dim=1)
        
        # Compute distances
        distances = F.pairwise_distance(embeddings1, embeddings2, p=2)
        
        # Basic statistics
        metrics = {
            'mean_similarity': similarities.mean().item(),
            'std_similarity': similarities.std().item(),
            'mean_distance': distances.mean().item(),
            'std_distance': distances.std().item()
        }
        
        # Compute metrics by label if available
        if labels is not None:
            pos_mask = labels.bool()
            neg_mask = ~pos_mask
            
            if pos_mask.any():
                metrics['pos_mean_similarity'] = similarities[pos_mask].mean().item()
                metrics['pos_mean_distance'] = distances[pos_mask].mean().item()
            
            if neg_mask.any():
                metrics['neg_mean_similarity'] = similarities[neg_mask].mean().item()
                metrics['neg_mean_distance'] = distances[neg_mask].mean().item()
        
        return metrics


if __name__ == "__main__":
    # Test loss functions
    logger.info("🧪 Testing loss functions...")
    
    # Create dummy embeddings
    batch_size = 32
    embedding_dim = 512
    
    anchor = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    positive = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    negative = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    
    # Test triplet loss
    try:
        loss = triplet_loss(anchor, positive, negative, margin=0.5)
        logger.info(f"✅ Triplet loss test passed: {loss.item():.4f}")
        
        # Test accuracy
        accuracy = compute_triplet_accuracy(anchor, positive, negative)
        logger.info(f"✅ Triplet accuracy test passed: {accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Triplet loss test failed: {e}")
    
    # Test contrastive loss
    try:
        labels = torch.randint(0, 2, (batch_size,))
        loss = contrastive_loss(anchor, positive, labels, margin=0.5)
        logger.info(f"✅ Contrastive loss test passed: {loss.item():.4f}")
        
    except Exception as e:
        logger.error(f"❌ Contrastive loss test failed: {e}")
    
    # Test similarity metrics
    try:
        metrics = compute_similarity_metrics(anchor, positive, labels)
        logger.info(f"✅ Similarity metrics test passed: {metrics}")
        
    except Exception as e:
        logger.error(f"❌ Similarity metrics test failed: {e}")

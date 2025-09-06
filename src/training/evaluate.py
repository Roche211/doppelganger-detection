"""
Evaluation Metrics for Similarity Learning

SIMPLE EVALUATION - NO COMPLEX METRICS
- Recall@K on validation pairs
- Visual inspection of results  
- Basic similarity statistics
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('/workspace/src')

from models.similarity_arcface import SimilarityArcFace, find_top_k_similar
from data.dataset import SimpleTripletDataset

logger = logging.getLogger(__name__)


def evaluate_recall_at_k(model: SimilarityArcFace, val_dataset: SimpleTripletDataset, 
                        k_values: List[int] = [1, 5, 10], device: str = 'cuda') -> Dict[str, float]:
    """
    Simple evaluation: Recall@K on validation pairs
    
    NO COMPLEX METRICS - JUST THE BASICS
    
    Args:
        model: Trained similarity model
        val_dataset: Validation dataset
        k_values: List of k values to evaluate
        device: Device to use
    
    Returns:
        Dictionary of recall@k scores
    """
    logger.info(f"📊 Evaluating Recall@{k_values}...")
    
    model.eval()
    model = model.to(device)
    
    # Collect all embeddings and create database
    logger.info("🔄 Creating embedding database...")
    
    all_embeddings = []
    all_paths = []
    query_positive_pairs = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            try:
                anchor, positive, negative = val_dataset[i]
                
                # Convert to batch format
                anchor_batch = anchor.unsqueeze(0).to(device)
                positive_batch = positive.unsqueeze(0).to(device)
                
                # Get embeddings
                anchor_emb = model(anchor_batch)
                positive_emb = model(positive_batch)
                
                # Store for database
                all_embeddings.extend([anchor_emb.cpu(), positive_emb.cpu()])
                all_paths.extend([f"anchor_{i}", f"positive_{i}"])
                
                # Store query-positive pairs
                query_positive_pairs.append((len(all_embeddings) - 2, len(all_embeddings) - 1))
                
                if (i + 1) % 100 == 0:
                    logger.info(f"   Processed {i + 1}/{len(val_dataset)} pairs...")
                    
            except Exception as e:
                logger.warning(f"Error processing pair {i}: {e}")
                continue
    
    if len(all_embeddings) == 0:
        logger.error("No embeddings collected!")
        return {}
    
    # Stack all embeddings
    database_embeddings = torch.stack(all_embeddings)
    logger.info(f"✅ Database created: {database_embeddings.shape}")
    
    # Evaluate recall@k
    recall_scores = {}
    
    for k in k_values:
        correct_retrievals = 0
        total_queries = len(query_positive_pairs)
        
        logger.info(f"🔍 Evaluating Recall@{k}...")
        
        for query_idx, positive_idx in query_positive_pairs:
            query_emb = database_embeddings[query_idx:query_idx+1]  # (1, embedding_dim)
            
            # Find top-k similar
            top_similarities, top_indices = find_top_k_similar(
                query_emb, database_embeddings, k=min(k+1, len(database_embeddings))
            )
            
            # Check if positive is in top-k (excluding self)
            top_indices_list = top_indices.tolist()
            if query_idx in top_indices_list:
                top_indices_list.remove(query_idx)  # Remove self
            
            if positive_idx in top_indices_list[:k]:
                correct_retrievals += 1
        
        recall_k = correct_retrievals / total_queries if total_queries > 0 else 0.0
        recall_scores[f'recall@{k}'] = recall_k
        
        logger.info(f"✅ Recall@{k}: {recall_k:.4f} ({correct_retrievals}/{total_queries})")
    
    return recall_scores


def evaluate_similarity_distribution(model: SimilarityArcFace, val_dataset: SimpleTripletDataset,
                                   device: str = 'cuda', num_samples: int = 1000) -> Dict[str, float]:
    """
    Evaluate similarity score distributions
    
    Args:
        model: Trained similarity model
        val_dataset: Validation dataset
        device: Device to use
        num_samples: Number of samples to evaluate
    
    Returns:
        Dictionary of similarity statistics
    """
    logger.info(f"📊 Evaluating similarity distributions on {num_samples} samples...")
    
    model.eval()
    model = model.to(device)
    
    positive_similarities = []
    negative_similarities = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset))):
            try:
                anchor, positive, negative = val_dataset[i]
                
                # Convert to batch format
                anchor_batch = anchor.unsqueeze(0).to(device)
                positive_batch = positive.unsqueeze(0).to(device)
                negative_batch = negative.unsqueeze(0).to(device)
                
                # Get embeddings
                anchor_emb = model(anchor_batch)
                positive_emb = model(positive_batch)
                negative_emb = model(negative_batch)
                
                # Compute similarities (cosine similarity for normalized embeddings)
                pos_sim = F.cosine_similarity(anchor_emb, positive_emb).item()
                neg_sim = F.cosine_similarity(anchor_emb, negative_emb).item()
                
                positive_similarities.append(pos_sim)
                negative_similarities.append(neg_sim)
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
    
    if not positive_similarities or not negative_similarities:
        logger.error("No similarities computed!")
        return {}
    
    # Compute statistics
    stats = {
        'positive_mean': np.mean(positive_similarities),
        'positive_std': np.std(positive_similarities),
        'negative_mean': np.mean(negative_similarities),
        'negative_std': np.std(negative_similarities),
        'separation': np.mean(positive_similarities) - np.mean(negative_similarities),
        'num_samples': len(positive_similarities)
    }
    
    logger.info(f"✅ Similarity statistics:")
    logger.info(f"   Positive pairs: {stats['positive_mean']:.3f} ± {stats['positive_std']:.3f}")
    logger.info(f"   Negative pairs: {stats['negative_mean']:.3f} ± {stats['negative_std']:.3f}")
    logger.info(f"   Separation: {stats['separation']:.3f}")
    
    return stats


def simple_visual_evaluation(model: SimilarityArcFace, val_dataset: SimpleTripletDataset,
                           device: str = 'cuda', num_examples: int = 5,
                           save_path: Optional[str] = None) -> None:
    """
    Simple visual evaluation - show query and top similar results
    
    Args:
        model: Trained similarity model
        val_dataset: Validation dataset
        device: Device to use
        num_examples: Number of examples to show
        save_path: Optional path to save visualization
    """
    logger.info(f"👁️  Creating visual evaluation with {num_examples} examples...")
    
    model.eval()
    model = model.to(device)
    
    # For simplicity, just show some triplet examples
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(min(num_examples, len(val_dataset))):
            try:
                anchor, positive, negative = val_dataset[i]
                
                # Convert to batch format
                anchor_batch = anchor.unsqueeze(0).to(device)
                positive_batch = positive.unsqueeze(0).to(device)
                negative_batch = negative.unsqueeze(0).to(device)
                
                # Get embeddings
                anchor_emb = model(anchor_batch)
                positive_emb = model(positive_batch)
                negative_emb = model(negative_batch)
                
                # Compute similarities
                pos_sim = F.cosine_similarity(anchor_emb, positive_emb).item()
                neg_sim = F.cosine_similarity(anchor_emb, negative_emb).item()
                
                # Convert tensors back to images for display
                def tensor_to_image(tensor):
                    # Denormalize from [-1, 1] to [0, 1]
                    img = (tensor + 1) / 2
                    img = img.permute(1, 2, 0).clamp(0, 1)
                    return img.numpy()
                
                # Display images
                axes[i, 0].imshow(tensor_to_image(anchor))
                axes[i, 0].set_title(f"Anchor")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(tensor_to_image(positive))
                axes[i, 1].set_title(f"Similar (sim={pos_sim:.3f})")
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(tensor_to_image(negative))
                axes[i, 2].set_title(f"Dissimilar (sim={neg_sim:.3f})")
                axes[i, 2].axis('off')
                
            except Exception as e:
                logger.warning(f"Error creating visual for example {i}: {e}")
                continue
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"💾 Visual evaluation saved: {save_path}")
    
    plt.show()


def comprehensive_evaluation(model_path: str, val_dataset_path: str, 
                           output_dir: str, device: str = 'cuda') -> Dict:
    """
    Run comprehensive evaluation on trained model
    
    Args:
        model_path: Path to trained model checkpoint
        val_dataset_path: Path to validation dataset
        output_dir: Directory to save evaluation results
        device: Device to use
    
    Returns:
        Dictionary of all evaluation results
    """
    logger.info(f"🎯 Running comprehensive evaluation...")
    
    # Load model
    logger.info(f"🔄 Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # This is a simplified version - you'd need to properly reconstruct the model
    # For now, just return placeholder results
    
    evaluation_results = {
        'model_path': model_path,
        'device': device,
        'recall_scores': {},
        'similarity_stats': {},
        'evaluation_time': 0.0
    }
    
    logger.info("⚠️  Comprehensive evaluation not fully implemented yet")
    logger.info("   Use individual evaluation functions for now")
    
    return evaluation_results


if __name__ == "__main__":
    # Test evaluation functions
    logger.info("🧪 Testing evaluation functions...")
    
    # This would require a trained model and dataset
    logger.info("⚠️  Evaluation tests require trained model and processed dataset")
    logger.info("   Run after training is complete")

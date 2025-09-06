"""
Data Organization Utilities - Create Organized Pair Structure

CRITICAL IMPLEMENTATION:
- Parse SLLFW pairs and create organized folder structure
- Process HDA doppelgänger pairs with same structure  
- Handle CelebA negatives sampling
- Create metadata for training pipeline
- Maintain pair relationships for easy training

OUTPUT STRUCTURE:
processed/pairs/sllfw_positive_pairs/
├── pair_0001/
│   ├── image1_112x112.jpg  # Person A
│   └── image2_112x112.jpg  # Person B (different but similar)
├── pair_0002/
│   ├── image1_112x112.jpg
│   └── image2_112x112.jpg
└── ...
"""

import os
import json
import random
import shutil
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

from .sllfw_parser import parse_sllfw_pairs
from .preprocessing import setup_face_analyzer, preprocess_face

logger = logging.getLogger(__name__)


def create_organized_sllfw_pairs(pair_file_path: str, lfw_dir: str, output_dir: str) -> Dict:
    """
    Parse SLLFW pairs and create organized folder structure for easy training
    
    ENGINEERING APPROACH:
    1. Parse pair_SLLFW.txt to get mismatched pairs (our positives)
    2. Create numbered pair folders for easy tracking
    3. Copy and rename aligned images to maintain pair relationships
    4. Save metadata for training pipeline
    
    Args:
        pair_file_path: Path to pair_SLLFW.txt
        lfw_dir: Directory containing LFW images
        output_dir: Output directory for processed data
    
    Returns:
        Dictionary with processing statistics and metadata
    """
    logger.info("🏗️  Creating organized SLLFW pair structure...")
    
    # Setup directories
    pairs_dir = Path(output_dir) / "pairs" / "sllfw_positive_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse SLLFW file (using our validated parsing logic)
    positive_pairs = parse_sllfw_pairs(pair_file_path)
    logger.info(f"✅ Parsed {len(positive_pairs)} positive pairs from SLLFW")
    
    # Setup face processor
    face_analyzer = setup_face_analyzer()
    
    pair_metadata = []
    successful_pairs = 0
    processing_stats = {
        'total_pairs': len(positive_pairs),
        'successful_pairs': 0,
        'failed_pairs': 0,
        'failure_reasons': {
            'image1_not_found': 0,
            'image2_not_found': 0,
            'image1_no_face': 0,
            'image2_no_face': 0,
            'both_no_face': 0
        }
    }
    
    for pair_idx, (img1_path, img2_path) in enumerate(positive_pairs):
        pair_num = f"pair_{pair_idx + 1:04d}"
        pair_dir = pairs_dir / pair_num
        pair_dir.mkdir(exist_ok=True)
        
        # Full paths to original images
        full_img1_path = Path(lfw_dir) / img1_path
        full_img2_path = Path(lfw_dir) / img2_path
        
        # Check if both images exist
        if not full_img1_path.exists():
            processing_stats['failure_reasons']['image1_not_found'] += 1
            logger.warning(f"Image 1 not found: {full_img1_path}")
            continue
            
        if not full_img2_path.exists():
            processing_stats['failure_reasons']['image2_not_found'] += 1
            logger.warning(f"Image 2 not found: {full_img2_path}")
            continue
        
        # Process both images
        output_img1 = pair_dir / "image1_112x112.jpg"
        output_img2 = pair_dir / "image2_112x112.jpg"
        
        aligned_img1 = preprocess_face(str(full_img1_path), face_analyzer, save_path=str(output_img1))
        aligned_img2 = preprocess_face(str(full_img2_path), face_analyzer, save_path=str(output_img2))
        
        # Check processing results
        if aligned_img1 is None and aligned_img2 is None:
            processing_stats['failure_reasons']['both_no_face'] += 1
            logger.warning(f"No faces detected in both images for pair {pair_num}")
            continue
        elif aligned_img1 is None:
            processing_stats['failure_reasons']['image1_no_face'] += 1
            logger.warning(f"No face detected in image1 for pair {pair_num}")
            continue
        elif aligned_img2 is None:
            processing_stats['failure_reasons']['image2_no_face'] += 1
            logger.warning(f"No face detected in image2 for pair {pair_num}")
            continue
        
        # Both images processed successfully
        pair_info = {
            'pair_id': pair_num,
            'original_paths': [img1_path, img2_path],
            'processed_paths': [str(output_img1), str(output_img2)],
            'person1': img1_path.split('/')[0],
            'person2': img2_path.split('/')[0]
        }
        pair_metadata.append(pair_info)
        successful_pairs += 1
        
        if (pair_idx + 1) % 100 == 0:
            logger.info(f"📊 Processed {pair_idx + 1}/{len(positive_pairs)} pairs ({successful_pairs} successful)...")
    
    # Update final statistics
    processing_stats['successful_pairs'] = successful_pairs
    processing_stats['failed_pairs'] = len(positive_pairs) - successful_pairs
    
    # Save metadata for training
    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / "sllfw_pairs_list.json"
    
    metadata_output = {
        'processing_stats': processing_stats,
        'pairs': pair_metadata
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_output, f, indent=2)
    
    logger.info(f"✅ SLLFW pair organization complete:")
    logger.info(f"   Successful: {successful_pairs}/{len(positive_pairs)} pairs ({100*successful_pairs/len(positive_pairs):.1f}%)")
    logger.info(f"   Structure: {pairs_dir}")
    logger.info(f"   Metadata: {metadata_file}")
    logger.info(f"   Failure breakdown: {processing_stats['failure_reasons']}")
    
    return metadata_output


def create_organized_hda_pairs(hda_dir: str, output_dir: str) -> Dict:
    """
    Process HDA doppelgänger dataset with same organized structure as SLLFW
    
    CORRECT HDA STRUCTURE:
    - Each person has Original and Lookalike versions
    - Format: {Gender}_{ID}_Original.{ext} and {Gender}_{ID}_Lookalike.{ext}
    - These form perfect positive pairs for similarity learning
    
    Args:
        hda_dir: Directory containing HDA doppelgänger images
        output_dir: Output directory for processed data
    
    Returns:
        Dictionary with processing statistics and metadata
    """
    logger.info("🏗️  Creating organized HDA doppelgänger pair structure...")
    
    # Setup directories
    pairs_dir = Path(output_dir) / "pairs" / "hda_positive_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    
    hda_path = Path(hda_dir)
    face_analyzer = setup_face_analyzer()
    pair_metadata = []
    successful_pairs = 0
    pair_idx = 0
    
    # Process by gender
    for gender_dir in ['Male', 'Female']:
        gender_path = hda_path / gender_dir
        if not gender_path.exists():
            continue
            
        logger.info(f"Processing {gender_dir} HDA images...")
        
        # Find all images in this gender directory
        all_files = list(gender_path.glob('*'))
        
        # Group by ID to find Original-Lookalike pairs
        id_groups = {}
        for file_path in all_files:
            if file_path.is_file():
                filename = file_path.stem  # filename without extension
                # Parse format: {Gender}_{ID}_{Type}
                parts = filename.split('_')
                if len(parts) >= 3:
                    gender_prefix = parts[0]  # M or F
                    person_id = parts[1]      # ID number
                    image_type = parts[2]     # Original or Lookalike
                    
                    key = f"{gender_prefix}_{person_id}"
                    if key not in id_groups:
                        id_groups[key] = {}
                    id_groups[key][image_type] = file_path
        
        logger.info(f"Found {len(id_groups)} person IDs in {gender_dir}")
        
        # Create pairs from Original-Lookalike matches
        for person_key, images in id_groups.items():
            if 'Original' in images and 'Lookalike' in images:
                original_path = images['Original']
                lookalike_path = images['Lookalike']
                
                pair_idx += 1
                pair_num = f"pair_{pair_idx:04d}"
                pair_dir = pairs_dir / pair_num
                pair_dir.mkdir(exist_ok=True)
                
                # Process both images
                output_original = pair_dir / "image1_112x112.jpg"  # Original person
                output_lookalike = pair_dir / "image2_112x112.jpg"  # Their lookalike
                
                aligned_original = preprocess_face(str(original_path), face_analyzer, save_path=str(output_original))
                aligned_lookalike = preprocess_face(str(lookalike_path), face_analyzer, save_path=str(output_lookalike))
                
                if aligned_original is not None and aligned_lookalike is not None:
                    pair_info = {
                        'pair_id': pair_num,
                        'original_paths': [str(original_path), str(lookalike_path)],
                        'processed_paths': [str(output_original), str(output_lookalike)],
                        'person_id': person_key,
                        'gender': gender_dir.lower(),
                        'source': 'hda_doppelganger',
                        'pair_type': 'original_lookalike'
                    }
                    pair_metadata.append(pair_info)
                    successful_pairs += 1
                    logger.debug(f"✅ Created pair {pair_num}: {person_key}")
                else:
                    logger.warning(f"⚠️  Failed to process pair for {person_key}")
            else:
                logger.debug(f"⚠️  Incomplete pair for {person_key}: {list(images.keys())}")
    
    # Save metadata
    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / "hda_pairs_list.json"
    
    processing_stats = {
        'total_person_ids': len(id_groups) if 'id_groups' in locals() else 0,
        'successful_pairs': successful_pairs,
        'failed_pairs': pair_idx - successful_pairs,
        'success_rate': 100 * successful_pairs / pair_idx if pair_idx > 0 else 0
    }
    
    metadata_output = {
        'processing_stats': processing_stats,
        'pairs': pair_metadata
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_output, f, indent=2)
    
    logger.info(f"✅ HDA pair organization complete:")
    logger.info(f"   Successful pairs: {successful_pairs}")
    logger.info(f"   Success rate: {processing_stats['success_rate']:.1f}%")
    logger.info(f"   Structure: {pairs_dir}")
    logger.info(f"   Metadata: {metadata_file}")
    
    return metadata_output


def sample_celeba_negatives(celeba_dir: str, output_dir: str, num_samples: int = 10000) -> Dict:
    """
    Sample random CelebA images as negatives and preprocess to 112x112
    
    Args:
        celeba_dir: Directory containing CelebA images
        output_dir: Output directory for processed negatives
        num_samples: Number of negative samples to create
    
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"🎲 Sampling {num_samples} CelebA negatives...")
    
    # Setup directories
    negatives_dir = Path(output_dir) / "pairs" / "celeba_negatives_112x112"
    negatives_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CelebA images
    celeba_path = Path(celeba_dir)
    celeba_images = []
    
    # Look for images in the CelebA directory structure
    for img_file in celeba_path.rglob('*.jpg'):
        celeba_images.append(str(img_file))
    
    logger.info(f"📊 Found {len(celeba_images)} CelebA images")
    
    if len(celeba_images) == 0:
        logger.error("No CelebA images found!")
        return {'processing_stats': {'successful_samples': 0, 'failed_samples': 0}}
    
    # Randomly sample images
    num_samples = min(num_samples, len(celeba_images))
    sampled_images = random.sample(celeba_images, num_samples)
    
    face_analyzer = setup_face_analyzer()
    successful_samples = 0
    negative_metadata = []
    
    for i, img_path in enumerate(sampled_images):
        if (i + 1) % 1000 == 0:
            logger.info(f"📊 Processing negatives: {i + 1}/{num_samples}")
        
        # Create output filename
        img_name = Path(img_path).stem
        output_path = negatives_dir / f"{img_name}_112x112.jpg"
        
        # Process image
        aligned_face = preprocess_face(img_path, face_analyzer, save_path=str(output_path))
        
        if aligned_face is not None:
            successful_samples += 1
            negative_metadata.append({
                'original_path': img_path,
                'processed_path': str(output_path),
                'sample_id': f"negative_{i + 1:05d}"
            })
    
    # Save metadata
    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / "celeba_negatives_list.json"
    
    processing_stats = {
        'total_sampled': num_samples,
        'successful_samples': successful_samples,
        'failed_samples': num_samples - successful_samples,
        'success_rate': 100 * successful_samples / num_samples if num_samples > 0 else 0
    }
    
    metadata_output = {
        'processing_stats': processing_stats,
        'negatives': negative_metadata
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_output, f, indent=2)
    
    logger.info(f"✅ CelebA negative sampling complete:")
    logger.info(f"   Successful: {successful_samples}/{num_samples} ({processing_stats['success_rate']:.1f}%)")
    logger.info(f"   Structure: {negatives_dir}")
    logger.info(f"   Metadata: {metadata_file}")
    
    return metadata_output


def create_train_val_splits(sllfw_metadata: Dict, hda_metadata: Dict, 
                           train_ratio: float = 0.8) -> Tuple[Dict, Dict]:
    """
    Create train/validation splits from positive pairs
    
    Args:
        sllfw_metadata: SLLFW pairs metadata
        hda_metadata: HDA pairs metadata
        train_ratio: Ratio of data for training
    
    Returns:
        Tuple of (train_metadata, val_metadata)
    """
    logger.info(f"📊 Creating train/val splits (train ratio: {train_ratio})")
    
    # Combine all positive pairs
    all_pairs = []
    
    if 'pairs' in sllfw_metadata:
        for pair in sllfw_metadata['pairs']:
            pair['source'] = 'sllfw'
            all_pairs.append(pair)
    
    if 'pairs' in hda_metadata:
        for pair in hda_metadata['pairs']:
            pair['source'] = 'hda'
            all_pairs.append(pair)
    
    logger.info(f"Total pairs: {len(all_pairs)}")
    
    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * train_ratio)
    
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    train_metadata = {
        'split': 'train',
        'total_pairs': len(train_pairs),
        'pairs': train_pairs
    }
    
    val_metadata = {
        'split': 'validation',
        'total_pairs': len(val_pairs),
        'pairs': val_pairs
    }
    
    logger.info(f"✅ Train/val split complete:")
    logger.info(f"   Train: {len(train_pairs)} pairs")
    logger.info(f"   Validation: {len(val_pairs)} pairs")
    
    return train_metadata, val_metadata


if __name__ == "__main__":
    # Test data organization
    logger.info("🧪 Testing data organization...")
    
    # Test SLLFW organization
    sllfw_result = create_organized_sllfw_pairs(
        pair_file_path="/workspace/pair_SLLFW.txt",
        lfw_dir="/workspace/lfw-deepfunneled",
        output_dir="/workspace/data/processed"
    )
    
    logger.info(f"SLLFW organization result: {sllfw_result['processing_stats']}")

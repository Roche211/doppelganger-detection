"""
SLLFW Parser - Extract Similar-Looking Different People Pairs

CRITICAL UNDERSTANDING:
- 12,000 total lines in pair_SLLFW.txt
- 10 folds × 1,200 lines each
- Each pair = 2 consecutive lines
- Lines 1-600 per fold: matched pairs (SAME PERSON) - IGNORE
- Lines 601-1200 per fold: mismatched pairs (DIFFERENT PEOPLE WHO LOOK SIMILAR) - USE AS POSITIVES

This extracts exactly 3,000 positive pairs for similarity learning.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_sllfw_pairs(pair_file_path: str) -> List[Tuple[str, str]]:
    """
    Extract ONLY the mismatched pairs (our positives) from SLLFW
    
    VALIDATION APPROACH:
    1. File has 12,000 lines total (confirmed by wc -l)
    2. 10 folds × 1,200 lines each
    3. Each fold: 600 matched + 600 mismatched pairs
    4. Each pair = 2 consecutive lines
    5. We want mismatched pairs (different people who look similar)
    
    Returns:
        List of (img1_path, img2_path) tuples for similar-looking different people
    """
    logger.info("🔍 PARSING SLLFW PAIRS - VALIDATION CHECKS:")
    
    # VALIDATION: Check file exists and has expected line count
    if not os.path.exists(pair_file_path):
        raise FileNotFoundError(f"SLLFW pair file not found: {pair_file_path}")
    
    with open(pair_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # VALIDATION: Confirm expected file structure
    expected_lines = 12000
    if len(lines) != expected_lines:
        raise ValueError(f"Expected {expected_lines} lines, got {len(lines)}. File format may be incorrect.")
    
    logger.info(f"✅ File validation passed: {len(lines)} lines")
    
    positive_pairs = []
    
    # Process each of 10 folds
    for fold in range(10):
        fold_start = fold * 1200  # Each fold has 1200 lines (600 pairs × 2 lines each)
        
        logger.info(f"📁 Processing fold {fold + 1}/10 (lines {fold_start}-{fold_start + 1199})")
        
        # Skip matched pairs (lines 0-599 of each fold) - SAME PERSON
        # Extract mismatched pairs (lines 600-1199 of each fold) - DIFFERENT PEOPLE, SIMILAR FACES
        mismatched_start = fold_start + 600
        mismatched_end = fold_start + 1200
        
        fold_pairs = 0
        
        # Process pairs (every 2 consecutive lines = 1 pair)
        for i in range(mismatched_start, mismatched_end, 2):
            if i + 1 < len(lines):
                img1_path = lines[i].strip()
                img2_path = lines[i + 1].strip()
                
                # VALIDATION: Ensure these are different people (different folder names)
                person1 = img1_path.split('/')[0]
                person2 = img2_path.split('/')[0]
                
                if person1 == person2:
                    raise ValueError(f"Expected different people, got same person: {person1}")
                
                positive_pairs.append((img1_path, img2_path))
                fold_pairs += 1
        
        # VALIDATION: Each fold should have exactly 300 mismatched pairs
        if fold_pairs != 300:
            raise ValueError(f"Fold {fold} has {fold_pairs} pairs, expected 300")
        
        logger.info(f"✅ Fold {fold + 1}: extracted {fold_pairs} similar-looking different people pairs")
    
    # FINAL VALIDATION: Should have exactly 3,000 pairs total
    expected_pairs = 3000
    if len(positive_pairs) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} pairs total, got {len(positive_pairs)}")
    
    logger.info(f"🎯 SUCCESS: Extracted {len(positive_pairs)} similar-looking different people pairs")
    logger.info(f"📊 Sample pairs:")
    for i in range(min(3, len(positive_pairs))):
        p1, p2 = positive_pairs[i]
        logger.info(f"   {p1} <-> {p2}")
    
    return positive_pairs


def validate_sllfw_structure(pair_file_path: str) -> dict:
    """
    Validate the SLLFW file structure and return statistics
    
    Returns:
        Dictionary with validation statistics
    """
    logger.info("🔬 Validating SLLFW file structure...")
    
    with open(pair_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    stats = {
        'total_lines': len(lines),
        'expected_lines': 12000,
        'folds': 10,
        'lines_per_fold': 1200,
        'matched_pairs_per_fold': 300,
        'mismatched_pairs_per_fold': 300,
        'total_matched_pairs': 3000,
        'total_mismatched_pairs': 3000,
        'validation_passed': False
    }
    
    # Check basic structure
    if len(lines) == 12000:
        stats['validation_passed'] = True
        logger.info("✅ Basic structure validation passed")
    else:
        logger.error(f"❌ Expected 12000 lines, got {len(lines)}")
    
    # Sample validation: Check first few pairs from each section
    sample_validation = []
    for fold in range(3):  # Check first 3 folds
        fold_start = fold * 1200
        
        # Check matched pair (same person)
        matched_idx = fold_start
        if matched_idx + 1 < len(lines):
            img1 = lines[matched_idx].strip()
            img2 = lines[matched_idx + 1].strip()
            person1 = img1.split('/')[0] if '/' in img1 else ''
            person2 = img2.split('/')[0] if '/' in img2 else ''
            
            sample_validation.append({
                'fold': fold + 1,
                'type': 'matched',
                'same_person': person1 == person2,
                'pair': (img1, img2)
            })
        
        # Check mismatched pair (different people)
        mismatched_idx = fold_start + 600
        if mismatched_idx + 1 < len(lines):
            img1 = lines[mismatched_idx].strip()
            img2 = lines[mismatched_idx + 1].strip()
            person1 = img1.split('/')[0] if '/' in img1 else ''
            person2 = img2.split('/')[0] if '/' in img2 else ''
            
            sample_validation.append({
                'fold': fold + 1,
                'type': 'mismatched',
                'same_person': person1 == person2,
                'pair': (img1, img2)
            })
    
    stats['sample_validation'] = sample_validation
    
    logger.info("📊 Sample validation results:")
    for sample in sample_validation:
        status = "✅" if (sample['type'] == 'matched' and sample['same_person']) or \
                       (sample['type'] == 'mismatched' and not sample['same_person']) else "❌"
        logger.info(f"   {status} Fold {sample['fold']} {sample['type']}: same_person={sample['same_person']}")
    
    return stats


if __name__ == "__main__":
    # Test the parser
    pair_file = "/workspace/pair_SLLFW.txt"
    
    # Validate structure first
    stats = validate_sllfw_structure(pair_file)
    
    # Parse positive pairs
    if stats['validation_passed']:
        positive_pairs = parse_sllfw_pairs(pair_file)
        logger.info(f"🎯 Extracted {len(positive_pairs)} positive pairs for similarity learning")
    else:
        logger.error("❌ File structure validation failed")

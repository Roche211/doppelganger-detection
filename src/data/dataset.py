"""
Dataset Classes for Similarity Learning

CRITICAL IMPLEMENTATION:
- SimpleTripletDataset for triplet training
- Load positive pairs from organized structure
- Sample random negatives from CelebA  
- Create triplets: (anchor, positive_similar, negative_dissimilar)

KEEP IT SIMPLE - NO OVERENGINEERING
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SimpleTripletDataset(Dataset):
    """
    Simple dataset for triplet training - NO OVERENGINEERING
    
    APPROACH:
    1. Load positive pairs from organized structure
    2. Sample random negatives from CelebA  
    3. Create triplets: (anchor, positive_similar, negative_dissimilar)
    
    KEEP IT SIMPLE - JUST WHAT WE NEED
    """
    
    def __init__(self, positive_pairs_dir: str, celeba_negatives_dir: str, 
                 transform: Optional[transforms.Compose] = None, split: str = 'train'):
        """
        Initialize dataset
        
        Args:
            positive_pairs_dir: Directory containing organized positive pairs
            celeba_negatives_dir: Directory containing CelebA negative samples
            transform: Image transforms
            split: Dataset split ('train' or 'val')
        """
        logger.info(f"📚 Loading {split} triplet dataset...")
        
        self.split = split
        self.positive_pairs_dir = Path(positive_pairs_dir)
        self.celeba_negatives_dir = Path(celeba_negatives_dir)
        
        # Load positive pairs (SLLFW + HDA)
        self.positive_pairs = []
        
        # SLLFW positive pairs
        sllfw_dir = self.positive_pairs_dir / "sllfw_positive_pairs"
        if sllfw_dir.exists():
            sllfw_pairs = self._load_pairs_from_directory(sllfw_dir, source='sllfw')
            self.positive_pairs.extend(sllfw_pairs)
            logger.info(f"   Loaded {len(sllfw_pairs)} SLLFW pairs")
        
        # HDA positive pairs (same structure)
        hda_dir = self.positive_pairs_dir / "hda_positive_pairs"
        if hda_dir.exists():
            hda_pairs = self._load_pairs_from_directory(hda_dir, source='hda')
            self.positive_pairs.extend(hda_pairs)
            logger.info(f"   Loaded {len(hda_pairs)} HDA pairs")
        
        # Load CelebA negatives
        self.negative_images = []
        if self.celeba_negatives_dir.exists():
            self.negative_images = [str(p) for p in self.celeba_negatives_dir.glob("*.jpg")]
            logger.info(f"   Loaded {len(self.negative_images)} CelebA negatives")
        
        if len(self.positive_pairs) == 0:
            raise ValueError(f"No positive pairs found in {positive_pairs_dir}")
        
        if len(self.negative_images) == 0:
            raise ValueError(f"No negative images found in {celeba_negatives_dir}")
        
        logger.info(f"✅ {split.capitalize()} dataset loaded:")
        logger.info(f"   Positive pairs: {len(self.positive_pairs)}")
        logger.info(f"   Negative pool: {len(self.negative_images)}")
        
        # Simple transform: just convert to tensor and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),  # Ensure correct size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
            ])
        else:
            self.transform = transform


    @staticmethod
    def from_splits(pairs_root: str, celeba_negatives_dir: str, sllfw_split_file: Optional[str],
                    hda_split_file: Optional[str], split: str = 'train',
                    transform: Optional[transforms.Compose] = None) -> 'SimpleTripletDataset':
        """Build a dataset from deterministic split JSONs.

        Args:
            pairs_root: path to processed/pairs
            celeba_negatives_dir: path to celeba_negatives_112x112
            sllfw_split_file: JSON with {train/val/test: [pair_folder_names]}
            hda_split_file: JSON with {train/val/test: [pair_folder_names]}
            split: one of 'train' | 'val' | 'test'
        """
        root = Path(pairs_root)
        ds = SimpleTripletDataset(positive_pairs_dir=pairs_root,
                                  celeba_negatives_dir=celeba_negatives_dir,
                                  transform=transform, split=split)
        # Override positive_pairs with split-controlled lists
        selected_pairs: List[Tuple[str, str]] = []

        # Helper to add pairs from listed folder names
        def add_pairs_from_list(base_dir: Path, folder_names: List[str]):
            for name in folder_names:
                pair_dir = base_dir / name
                img1 = pair_dir / "image1_112x112.jpg"
                img2 = pair_dir / "image2_112x112.jpg"
                if img1.exists() and img2.exists():
                    selected_pairs.append((str(img1), str(img2)))

        if sllfw_split_file and Path(sllfw_split_file).exists():
            with open(sllfw_split_file, 'r') as f:
                sllfw_splits = json.load(f)
            sllfw_names = sllfw_splits.get(split, [])
            add_pairs_from_list(root / "sllfw_positive_pairs", sllfw_names)

        if hda_split_file and Path(hda_split_file).exists():
            with open(hda_split_file, 'r') as f:
                hda_splits = json.load(f)
            hda_names = hda_splits.get(split, [])
            add_pairs_from_list(root / "hda_positive_pairs", hda_names)

        if selected_pairs:
            ds.positive_pairs = selected_pairs
            logger.info(f"🔒 Deterministic split '{split}': {len(ds.positive_pairs)} positive pairs")
        else:
            logger.warning("No split files found or empty; falling back to directory scan")
        return ds
    
    def _load_pairs_from_directory(self, pairs_dir: Path, source: str) -> List[Tuple[str, str]]:
        """
        Load pairs from organized directory structure
        
        Args:
            pairs_dir: Directory containing pair folders
            source: Source name for logging
        
        Returns:
            List of (image1_path, image2_path) tuples
        """
        pairs = []
        
        for pair_folder in sorted(pairs_dir.iterdir()):
            if pair_folder.is_dir():
                img1 = pair_folder / "image1_112x112.jpg"
                img2 = pair_folder / "image2_112x112.jpg"
                
                if img1.exists() and img2.exists():
                    pairs.append((str(img1), str(img2)))
                else:
                    logger.warning(f"Missing images in {pair_folder}")
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.positive_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get triplet: (anchor, positive, negative)
        
        Args:
            idx: Index
        
        Returns:
            Tuple of (anchor, positive, negative) tensors
        """
        # Get positive pair (similar faces, different people)
        anchor_path, positive_path = self.positive_pairs[idx]
        
        # Sample random negative (dissimilar face)
        negative_path = random.choice(self.negative_images)
        
        try:
            # Load images (already 112x112 aligned)
            anchor = Image.open(anchor_path).convert('RGB')
            positive = Image.open(positive_path).convert('RGB')
            negative = Image.open(negative_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)
            
            return anchor, positive, negative
            
        except Exception as e:
            logger.warning(f"Error loading triplet at index {idx}: {e}")
            # Return a different random triplet
            return self.__getitem__(random.randint(0, len(self) - 1))


class PairDataset(Dataset):
    """
    Simple dataset for pair-based training (contrastive loss)
    """
    
    def __init__(self, positive_pairs_dir: str, celeba_negatives_dir: str,
                 transform: Optional[transforms.Compose] = None, split: str = 'train'):
        """
        Initialize pair dataset
        
        Args:
            positive_pairs_dir: Directory containing positive pairs
            celeba_negatives_dir: Directory containing negative samples
            transform: Image transforms
            split: Dataset split
        """
        logger.info(f"📚 Loading {split} pair dataset...")
        
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load positive pairs
        self.pairs = []
        self.labels = []
        
        # Add positive pairs (label = 1)
        positive_pairs_path = Path(positive_pairs_dir)
        for pairs_subdir in ['sllfw_positive_pairs', 'hda_positive_pairs']:
            pairs_dir = positive_pairs_path / pairs_subdir
            if pairs_dir.exists():
                for pair_folder in sorted(pairs_dir.iterdir()):
                    if pair_folder.is_dir():
                        img1 = pair_folder / "image1_112x112.jpg"
                        img2 = pair_folder / "image2_112x112.jpg"
                        
                        if img1.exists() and img2.exists():
                            self.pairs.append((str(img1), str(img2)))
                            self.labels.append(1)  # Similar pair
        
        # Add negative pairs (label = 0)
        negatives_dir = Path(celeba_negatives_dir)
        if negatives_dir.exists():
            negative_images = list(negatives_dir.glob("*.jpg"))
            
            # Create random negative pairs
            num_negatives = len(self.pairs)  # Match number of positive pairs
            for _ in range(num_negatives):
                img1, img2 = random.sample(negative_images, 2)
                self.pairs.append((str(img1), str(img2)))
                self.labels.append(0)  # Dissimilar pair
        
        logger.info(f"✅ {split.capitalize()} pair dataset loaded:")
        logger.info(f"   Total pairs: {len(self.pairs)}")
        logger.info(f"   Positive pairs: {sum(self.labels)}")
        logger.info(f"   Negative pairs: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get pair: (image1, image2, label)
        
        Returns:
            Tuple of (image1, image2, similarity_label)
        """
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        try:
            # Load images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            return img1, img2, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            logger.warning(f"Error loading pair at index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


def create_data_splits(processed_data_dir: str, train_ratio: float = 0.8) -> Dict:
    """
    Create train/validation splits from processed data
    
    Args:
        processed_data_dir: Directory containing processed data
        train_ratio: Ratio for training split
    
    Returns:
        Dictionary with split information
    """
    logger.info(f"📊 Creating data splits (train ratio: {train_ratio})")
    
    processed_path = Path(processed_data_dir)
    pairs_dir = processed_path / "pairs"
    
    # Count available pairs
    total_pairs = 0
    pair_sources = {}
    
    for source_dir in ['sllfw_positive_pairs', 'hda_positive_pairs']:
        source_path = pairs_dir / source_dir
        if source_path.exists():
            source_pairs = len([d for d in source_path.iterdir() if d.is_dir()])
            pair_sources[source_dir] = source_pairs
            total_pairs += source_pairs
    
    # Create splits
    train_pairs = int(total_pairs * train_ratio)
    val_pairs = total_pairs - train_pairs
    
    splits_info = {
        'total_pairs': total_pairs,
        'train_pairs': train_pairs,
        'val_pairs': val_pairs,
        'train_ratio': train_ratio,
        'pair_sources': pair_sources,
        'pairs_dir': str(pairs_dir),
        'celeba_negatives_dir': str(pairs_dir / "celeba_negatives_112x112")
    }
    
    logger.info(f"✅ Data splits created:")
    logger.info(f"   Total pairs: {total_pairs}")
    logger.info(f"   Train pairs: {train_pairs}")
    logger.info(f"   Val pairs: {val_pairs}")
    logger.info(f"   Sources: {pair_sources}")
    
    return splits_info


def get_data_loaders(processed_data_dir: str, batch_size: int = 32, num_workers: int = 4,
                    train_ratio: float = 0.8) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        processed_data_dir: Directory containing processed data
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        train_ratio: Ratio for training split
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info(f"🔄 Creating data loaders (batch_size: {batch_size})")
    
    processed_path = Path(processed_data_dir)
    pairs_dir = processed_path / "pairs"
    celeba_dir = pairs_dir / "celeba_negatives_112x112"
    
    # Create datasets
    train_dataset = SimpleTripletDataset(
        positive_pairs_dir=str(pairs_dir),
        celeba_negatives_dir=str(celeba_dir),
        split='train'
    )
    
    val_dataset = SimpleTripletDataset(
        positive_pairs_dir=str(pairs_dir),
        celeba_negatives_dir=str(celeba_dir),
        split='val'
    )
    
    # Note: prefer using get_data_loaders_from_splits for proper deterministic splits
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    logger.info(f"✅ Data loaders created:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def get_data_loaders_from_splits(processed_data_dir: str, batch_size: int = 32, num_workers: int = 4,
                                 sllfw_split_file: Optional[str] = None, hda_split_file: Optional[str] = None
                                 ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test loaders from split JSONs.

    Args:
        processed_data_dir: root processed dir
        sllfw_split_file: path to sllfw_splits.json
        hda_split_file: path to hda_splits.json
    """
    processed_path = Path(processed_data_dir)
    pairs_dir = processed_path / "pairs"
    celeba_dir = pairs_dir / "celeba_negatives_112x112"

    common_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_ds = SimpleTripletDataset.from_splits(str(pairs_dir), str(celeba_dir), sllfw_split_file, hda_split_file, split='train', transform=common_transform)
    val_ds = SimpleTripletDataset.from_splits(str(pairs_dir), str(celeba_dir), sllfw_split_file, hda_split_file, split='val', transform=common_transform)
    test_ds = SimpleTripletDataset.from_splits(str(pairs_dir), str(celeba_dir), sllfw_split_file, hda_split_file, split='test', transform=common_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False
    )

    logger.info(f"✅ Deterministic loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    logger.info("🧪 Testing dataset loading...")
    
    # Test with dummy data
    processed_dir = "/workspace/data/processed"
    
    if os.path.exists(processed_dir):
        try:
            splits_info = create_data_splits(processed_dir)
            logger.info(f"✅ Data splits test passed: {splits_info}")
            
        except Exception as e:
            logger.error(f"❌ Dataset test failed: {e}")
    else:
        logger.warning("Processed data directory not found for testing")

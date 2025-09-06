"""
Deterministic train/val/test split creation using existing pair folders.

Rules:
- SLLFW: split at pair level, deterministically by pair_id (folder name), with a fixed seed.
- HDA: split by person_id (from metadata), keep both images of each person_id in the same split.

Outputs:
- JSON files listing pair folder names for each split.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

SPLIT_NAMES = ("train", "val", "test")


def _stable_hash(s: str, seed: str = "s3ed") -> int:
    h = hashlib.sha256((seed + s).encode("utf-8")).hexdigest()
    return int(h, 16)


def create_sllfw_splits(pairs_root: str, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: str = "s3ed") -> Dict[str, List[str]]:
    """Create deterministic splits for SLLFW pair folders.

    Args:
        pairs_root: path to `pairs/sllfw_positive_pairs`
        ratios: train/val/test ratios (sum to 1)
        seed: deterministic seed string

    Returns:
        dict with split name -> list of pair folder names
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    train_r, val_r, test_r = ratios

    root = Path(pairs_root)
    pair_dirs = sorted([d.name for d in root.iterdir() if d.is_dir()])
    n = len(pair_dirs)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    # deterministic order by stable hash
    pair_dirs_sorted = sorted(pair_dirs, key=lambda x: _stable_hash(x, seed))

    splits = {
        "train": pair_dirs_sorted[:n_train],
        "val": pair_dirs_sorted[n_train:n_train + n_val],
        "test": pair_dirs_sorted[n_train + n_val:]
    }
    return splits


def create_hda_splits(hda_metadata_file: str, pairs_root: str, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: str = "s3ed") -> Dict[str, List[str]]:
    """Create deterministic splits for HDA by person_id.

    Args:
        hda_metadata_file: JSON produced earlier with person_id and pair_id
        pairs_root: path to `pairs/hda_positive_pairs`
        ratios: train/val/test ratios
        seed: deterministic seed string

    Returns:
        dict with split name -> list of pair folder names
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    train_r, val_r, test_r = ratios

    with open(hda_metadata_file, "r") as f:
        meta = json.load(f)
    pairs = meta.get("pairs", [])

    # Group pair_ids by person_id
    pid_to_pairs: Dict[str, List[str]] = {}
    for p in pairs:
        pid = p.get("person_id", "")
        pair_id = p.get("pair_id", "")
        if not pid or not pair_id:
            continue
        pid_to_pairs.setdefault(pid, []).append(pair_id)

    person_ids = sorted(pid_to_pairs.keys(), key=lambda x: _stable_hash(x, seed))
    n = len(person_ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    split_pids = {
        "train": person_ids[:n_train],
        "val": person_ids[n_train:n_train + n_val],
        "test": person_ids[n_train + n_val:]
    }

    splits = {"train": [], "val": [], "test": []}
    for split, pids in split_pids.items():
        for pid in pids:
            splits[split].extend(pid_to_pairs.get(pid, []))

    return splits


def save_splits(splits: Dict[str, List[str]], output_file: str):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(splits, f, indent=2)


if __name__ == "__main__":
    # Example CLI generation (paths may be adjusted by caller scripts)
    base = Path("/workspace/data/processed/pairs")
    sllfw_dir = base / "sllfw_positive_pairs"
    hda_dir = base / "hda_positive_pairs"
    out_dir = Path("/workspace/data/processed/metadata")

    if sllfw_dir.exists():
        sllfw_splits = create_sllfw_splits(str(sllfw_dir))
        save_splits(sllfw_splits, str(out_dir / "sllfw_splits.json"))
        print("Saved:", out_dir / "sllfw_splits.json")

    hda_meta = Path("/workspace/data/processed/metadata/hda_pairs_list.json")
    if hda_dir.exists() and hda_meta.exists():
        hda_splits = create_hda_splits(str(hda_meta), str(hda_dir))
        save_splits(hda_splits, str(out_dir / "hda_splits.json"))
        print("Saved:", out_dir / "hda_splits.json")



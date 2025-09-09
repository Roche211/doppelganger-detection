#!/usr/bin/env python3
"""
Exact Duplicate Removal Only

This script finds and removes ONLY exact file duplicates (same bytes)
using MD5 hashing. No perceptual/similarity detection.
"""

import os
import sys
import hashlib
import shutil
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

def calculate_file_hash(filepath, hash_type='md5'):
    """Calculate MD5 hash of file contents."""
    hash_func = hashlib.md5()

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def find_exact_duplicates(image_files, gallery_dir):
    """Find exact duplicates using file hashing."""

    print("🔍 Finding exact duplicates by MD5 hash...")
    hash_groups = defaultdict(list)

    for filename in tqdm(image_files):
        filepath = os.path.join(gallery_dir, filename)
        file_hash = calculate_file_hash(filepath)
        hash_groups[file_hash].append(filename)

    # Separate unique files from duplicates
    unique_files = []
    duplicate_groups = []

    for hash_value, files in hash_groups.items():
        if len(files) > 1:
            # This is a duplicate group
            duplicate_groups.append({
                'hash': hash_value,
                'files': files,
                'count': len(files)
            })
            # Keep only the first file from each group
            unique_files.append(files[0])
        else:
            # Single file, definitely unique
            unique_files.append(files[0])

    return duplicate_groups, unique_files

def copy_unique_images(unique_files, source_dir, dest_dir):
    """Copy unique images to destination directory."""

    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    failed = 0

    print(f"📋 Copying {len(unique_files)} unique images...")

    for filename in tqdm(unique_files):
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(dest_dir, filename)

        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            print(f"❌ Failed to copy {filename}: {e}")
            failed += 1

    return copied, failed

def generate_report(duplicate_groups, total_files, unique_files, output_file):
    """Generate detailed deduplication report."""

    total_duplicates = sum(len(group['files']) - 1 for group in duplicate_groups)

    report = {
        'summary': {
            'total_images': total_files,
            'unique_images': len(unique_files),
            'duplicate_groups': len(duplicate_groups),
            'total_duplicates_removed': total_duplicates,
            'duplicate_percentage': round(total_duplicates / total_files * 100, 2) if total_files > 0 else 0
        },
        'duplicate_groups': duplicate_groups,
        'unique_files': unique_files
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    return report

def print_summary_report(report):
    """Print formatted summary of deduplication results."""

    print("\n" + "="*60)
    print("📊 EXACT DUPLICATE REMOVAL REPORT")
    print("="*60)

    s = report['summary']
    print(f"Total Images: {s['total_images']}")
    print(f"Unique Images: {s['unique_images']}")
    print(f"Duplicates Removed: {s['total_duplicates_removed']}")
    print(f"Duplicate Groups: {s['duplicate_groups']}")
    print(f"Duplicate Percentage: {s['duplicate_percentage']}%")

    if s['duplicate_groups'] > 0:
        print(f"\n📁 Duplicate Groups Found:")
        # Show largest duplicate groups first
        sorted_groups = sorted(report['duplicate_groups'], key=lambda x: x['count'], reverse=True)

        for i, group in enumerate(sorted_groups[:10], 1):  # Show top 10
            print(f"  Group {i}: {group['count']} identical images")
            print(f"    Hash: {group['hash'][:16]}...")
            print(f"    Files: {group['files'][:3]}{'...' if len(group['files']) > 3 else ''}")

        if s['duplicate_groups'] > 10:
            print(f"  ... and {s['duplicate_groups'] - 10} more groups")

    print("\n" + "="*60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove only exact file duplicates")
    parser.add_argument('--gallery_dir', default='/workspace/processed_gallery_112_lightning',
                       help='Directory containing images')
    parser.add_argument('--output_dir', default='/workspace/processed_gallery_exact_unique',
                       help='Directory to save unique images')
    parser.add_argument('--report_file', default='/workspace/exact_deduplication_report.json',
                       help='Output file for deduplication report')

    args = parser.parse_args()

    # Check directories
    if not os.path.exists(args.gallery_dir):
        print(f"❌ Gallery directory not found: {args.gallery_dir}")
        sys.exit(1)

    if os.path.abspath(args.gallery_dir) == os.path.abspath(args.output_dir):
        print("❌ Output directory cannot be the same as input directory")
        sys.exit(1)

    # Get image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        f for f in os.listdir(args.gallery_dir)
        if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print("❌ No image files found in gallery directory")
        sys.exit(1)

    print("🎯 Exact Duplicate Removal Only")
    print(f"Source: {args.gallery_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Images: {len(image_files)}")
    print("-" * 50)
    print("⚠️  This will ONLY remove exact file duplicates (same bytes)")
    print("⚠️  Visually similar but different files will be kept")

    # Find exact duplicates
    duplicate_groups, unique_files = find_exact_duplicates(image_files, args.gallery_dir)

    # Copy unique images
    copied, failed = copy_unique_images(unique_files, args.gallery_dir, args.output_dir)

    # Generate report
    report = generate_report(duplicate_groups, len(image_files), unique_files, args.report_file)

    # Print summary
    print_summary_report(report)

    print(f"\n✅ Exact deduplication complete!")
    print(f"📂 Unique images saved to: {args.output_dir}")
    print(f"📊 Full report saved to: {args.report_file}")

    if failed > 0:
        print(f"⚠️  {failed} files failed to copy")

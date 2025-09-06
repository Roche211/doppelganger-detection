"""
Face Preprocessing Pipeline - InsightFace 112x112 Alignment

CRITICAL REQUIREMENTS:
- Use InsightFace for face detection and alignment
- Output exactly 112x112 RGB images (ArcFace standard)
- Handle missing faces gracefully
- Log statistics for debugging
- Validate output dimensions and format

SENIOR ML ENGINEER APPROACH:
- Fail fast and fail loud with proper validation
- Log everything for debugging
- Handle edge cases (no faces, multiple faces)
- Track success/failure rates
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
    logger.info("✅ InsightFace available")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.error("❌ InsightFace not available. Install with: pip install insightface")


def setup_face_analyzer(ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640), det_thresh: Optional[float] = None):
    """
    Initialize InsightFace with proper error handling
    
    Args:
        ctx_id: GPU context ID (0 for first GPU, -1 for CPU)
        det_size: Detection input size
    
    Returns:
        InsightFace app instance
    """
    if not INSIGHTFACE_AVAILABLE:
        raise ImportError("InsightFace not available. Install with: pip install insightface")
    
    try:
        logger.info("🔄 Initializing InsightFace face analyzer...")
        app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        # Optionally set detection threshold if supported by this version
        if det_thresh is not None and hasattr(app, 'det_thresh'):
            try:
                app.det_thresh = float(det_thresh)
                logger.info(f"   Detection threshold set to: {app.det_thresh}")
            except Exception:
                pass
        logger.info("✅ InsightFace initialized successfully")
        logger.info(f"   Context: {'GPU' if ctx_id >= 0 else 'CPU'}")
        logger.info(f"   Detection size: {det_size}")
        return app
    except Exception as e:
        logger.error(f"❌ Failed to initialize InsightFace: {e}")
        raise


def preprocess_face(image_path: str, face_analyzer, save_path: Optional[str] = None, 
                   target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
    """
    Convert any image to ArcFace-ready 112x112 aligned face
    
    ENGINEERING CONSIDERATIONS:
    - Handle missing files gracefully
    - Validate face detection results
    - Log failure cases for debugging
    - Return None for failed cases (don't crash)
    - Optionally save processed faces for inspection
    
    Args:
        image_path: Path to input image
        face_analyzer: InsightFace app instance
        save_path: Optional path to save aligned face
        target_size: Output size (default 112x112 for ArcFace)
    
    Returns:
        Aligned face as numpy array (H, W, C) or None if failed
    """
    try:
        # VALIDATION: Check if file exists
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return None
        
        # Load image (BGR)
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not load image: {image_path}")
            return None
        
        def detect_and_align(bgr_img: np.ndarray) -> Optional[np.ndarray]:
            faces_local = face_analyzer.get(bgr_img)
            if faces_local is None or len(faces_local) == 0:
                return None
            face_local = max(
                faces_local,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            if not hasattr(face_local, 'kps') or face_local.kps is None:
                return None
            try:
                import insightface.utils.face_align
                return insightface.utils.face_align.norm_crop(
                    bgr_img, landmark=face_local.kps, image_size=target_size[0]
                )
            except Exception:
                return None

        # Progressive retries: original → upscaled small images → downscaled huge images
        H, W = img.shape[:2]
        attempts = [img]
        max_side = max(H, W)
        # If very small faces likely, upscale to ~640 on long side
        if max_side < 400:
            scale = 640.0 / max(1.0, float(max_side))
            up = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_CUBIC)
            attempts.append(up)
        # If extremely large, also try downscale to ~640 to stabilize detector
        if max_side > 1600:
            scale = 640.0 / float(max_side)
            down = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
            attempts.append(down)

        aligned_face = None
        for cand in attempts:
            aligned_face = detect_and_align(cand)
            if aligned_face is not None:
                break
        if aligned_face is None:
            logger.warning(f"No faces detected in: {image_path}")
            return None
        
        # VALIDATION: Check output format
        if aligned_face is None:
            logger.warning(f"Face alignment failed for: {image_path}")
            return None
        
        # Ensure correct dimensions
        if len(aligned_face.shape) == 3 and aligned_face.shape[:2] == target_size:
            # Save directly (aligned_face is BGR when using cv2 pipeline)
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                success = cv2.imwrite(save_path, aligned_face)
                if not success:
                    logger.warning(f"Failed to save aligned face to: {save_path}")
            
            return aligned_face
        else:
            logger.error(f"Unexpected aligned face shape: {aligned_face.shape} for {image_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def batch_preprocess_dataset(image_paths: List[str], output_dir: str, dataset_name: str,
                           face_analyzer=None, save_images: bool = True) -> Tuple[int, List[str], Dict]:
    """
    Process entire dataset with progress tracking and statistics
    
    SENIOR ML ENGINEER APPROACH:
    - Track success/failure rates
    - Save statistics for analysis
    - Create organized output structure
    - Handle errors gracefully
    - Provide detailed progress reporting
    
    Args:
        image_paths: List of input image paths
        output_dir: Directory to save processed images
        dataset_name: Name for logging
        face_analyzer: InsightFace app (will create if None)
        save_images: Whether to save processed images
    
    Returns:
        Tuple of (success_count, failed_paths, statistics)
    """
    logger.info(f"🔄 Processing {dataset_name} dataset: {len(image_paths)} images")
    
    if face_analyzer is None:
        face_analyzer = setup_face_analyzer()
    
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_paths = []
    processing_stats = {
        'dataset_name': dataset_name,
        'total_images': len(image_paths),
        'success_count': 0,
        'failed_count': 0,
        'success_rate': 0.0,
        'failure_reasons': {},
        'sample_successes': [],
        'sample_failures': []
    }
    
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            logger.info(f"📊 Progress: {i + 1}/{len(image_paths)} ({100*(i+1)/len(image_paths):.1f}%)")
        
        # Create output path maintaining structure
        if save_images:
            rel_path = os.path.relpath(img_path, start=os.path.dirname(img_path))
            output_filename = f"{os.path.splitext(os.path.basename(rel_path))[0]}_112x112.jpg"
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_path = None
        
        # Process face
        aligned_face = preprocess_face(img_path, face_analyzer, save_path=output_path)
        
        if aligned_face is not None:
            success_count += 1
            if len(processing_stats['sample_successes']) < 5:
                processing_stats['sample_successes'].append(img_path)
        else:
            failed_paths.append(img_path)
            if len(processing_stats['sample_failures']) < 5:
                processing_stats['sample_failures'].append(img_path)
    
    # Update statistics
    processing_stats['success_count'] = success_count
    processing_stats['failed_count'] = len(failed_paths)
    processing_stats['success_rate'] = 100 * success_count / len(image_paths) if image_paths else 0
    
    # REPORT STATISTICS
    logger.info(f"✅ {dataset_name} preprocessing complete:")
    logger.info(f"   Success: {success_count}/{len(image_paths)} ({processing_stats['success_rate']:.1f}%)")
    logger.info(f"   Failed: {len(failed_paths)} images")
    
    if failed_paths:
        failed_log = os.path.join(output_dir, f'{dataset_name}_failed_preprocessing.txt')
        with open(failed_log, 'w') as f:
            for path in failed_paths:
                f.write(f"{path}\n")
        logger.info(f"📝 Failed image paths saved to: {failed_log}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, f'{dataset_name}_preprocessing_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(processing_stats, f, indent=2)
    
    # VALIDATION: Ensure minimum success rate
    if processing_stats['success_rate'] < 90:
        logger.warning(f"⚠️  WARNING: Low success rate ({processing_stats['success_rate']:.1f}%). Check data quality.")
    
    return success_count, failed_paths, processing_stats


def validate_aligned_faces(aligned_faces_dir: str, expected_size: Tuple[int, int] = (112, 112)) -> Dict:
    """
    Validate a directory of aligned faces
    
    Args:
        aligned_faces_dir: Directory containing aligned face images
        expected_size: Expected image dimensions
    
    Returns:
        Validation statistics
    """
    logger.info(f"🔬 Validating aligned faces in: {aligned_faces_dir}")
    
    validation_stats = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'size_mismatches': 0,
        'load_failures': 0,
        'sample_valid': [],
        'sample_invalid': []
    }
    
    if not os.path.exists(aligned_faces_dir):
        logger.error(f"Directory not found: {aligned_faces_dir}")
        return validation_stats
    
    image_files = [f for f in os.listdir(aligned_faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    validation_stats['total_files'] = len(image_files)
    
    for img_file in image_files:
        img_path = os.path.join(aligned_faces_dir, img_file)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                validation_stats['load_failures'] += 1
                validation_stats['invalid_files'] += 1
                if len(validation_stats['sample_invalid']) < 3:
                    validation_stats['sample_invalid'].append(f"{img_file}: Load failure")
                continue
            
            if img.shape[:2] != expected_size:
                validation_stats['size_mismatches'] += 1
                validation_stats['invalid_files'] += 1
                if len(validation_stats['sample_invalid']) < 3:
                    validation_stats['sample_invalid'].append(f"{img_file}: Size {img.shape[:2]} != {expected_size}")
                continue
            
            validation_stats['valid_files'] += 1
            if len(validation_stats['sample_valid']) < 3:
                validation_stats['sample_valid'].append(f"{img_file}: {img.shape}")
                
        except Exception as e:
            validation_stats['invalid_files'] += 1
            if len(validation_stats['sample_invalid']) < 3:
                validation_stats['sample_invalid'].append(f"{img_file}: {str(e)}")
    
    success_rate = 100 * validation_stats['valid_files'] / validation_stats['total_files'] if validation_stats['total_files'] > 0 else 0
    
    logger.info(f"✅ Validation complete:")
    logger.info(f"   Valid: {validation_stats['valid_files']}/{validation_stats['total_files']} ({success_rate:.1f}%)")
    logger.info(f"   Size mismatches: {validation_stats['size_mismatches']}")
    logger.info(f"   Load failures: {validation_stats['load_failures']}")
    
    return validation_stats


if __name__ == "__main__":
    # Test preprocessing pipeline
    logger.info("🧪 Testing preprocessing pipeline...")
    
    # Test with a few sample images
    sample_images = [
        "/workspace/lfw-deepfunneled/Abel_Pacheco/Abel_Pacheco_0001.jpg",
        "/workspace/lfw-deepfunneled/Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg"
    ]
    
    # Test single image processing
    if os.path.exists(sample_images[0]):
        face_analyzer = setup_face_analyzer()
        result = preprocess_face(sample_images[0], face_analyzer, save_path="/tmp/test_aligned.jpg")
        if result is not None:
            logger.info(f"✅ Single image test passed: {result.shape}")
        else:
            logger.error("❌ Single image test failed")
    else:
        logger.warning("Sample images not found for testing")

"""
Optimized Face Preprocessor - Speed improvements for RTX 5090
Key optimizations:
1. Smaller detection size for faster processing
2. Optional landmark alignment (can be disabled)
3. Better batch processing
4. Reduced I/O operations
5. Memory optimizations
"""

import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import warnings
from typing import Optional, List, Tuple
from PIL import Image, ImageOps
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure project src is on path
sys.path.append('/workspace/src')

# Try to import InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    print("InsightFace not available, falling back to MTCNN only")
    INSIGHTFACE_AVAILABLE = False

# Try to import MTCNN
try:
    from facenet_pytorch import MTCNN
    import torch
    MTCNN_AVAILABLE = True
except ImportError:
    print("MTCNN not available")
    MTCNN_AVAILABLE = False
    torch = None


class OptimizedFacePreprocessor:
    def __init__(self, device='cuda', fast_mode=True, enable_alignment=False):
        self.device = device
        self.fast_mode = fast_mode
        self.enable_alignment = enable_alignment

        # Initialize MTCNN if available (with faster settings)
        self.mtcnn = None
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN(
                    device=device,
                    min_face_size=40,  # Increased for faster processing
                    thresholds=[0.7, 0.8, 0.8],  # Higher thresholds = faster
                    post_process=False,
                    keep_all=False,
                    selection_method='largest'  # Take largest face only
                )
                print("MTCNN initialized with fast settings")
            except Exception as e:
                print(f"Error initializing MTCNN: {e}")
                self.mtcnn = None

        # Initialize InsightFace with optimized settings
        self.insightface = None
        if INSIGHTFACE_AVAILABLE:
            try:
                # Only load detection module if alignment is disabled
                modules = ['detection']
                if self.enable_alignment:
                    modules.append('landmark_2d_106')
                
                self.insightface = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                    allowed_modules=modules
                )
                
                # Use smaller detection size for faster processing
                det_size = (320, 320) if fast_mode else (480, 480)
                self.insightface.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=det_size)
                print(f"InsightFace initialized with detection size: {det_size}")
            except Exception as e:
                print(f"Error initializing InsightFace: {e}")
                self.insightface = None

    def preprocess_image_fast(self, img_path: str, margin_percent: float = 0.35, 
                             target_size: int = 112) -> Optional[Image.Image]:
        """
        Fast preprocessing with minimal operations
        """
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Try InsightFace first
            if self.insightface is not None:
                try:
                    faces = self.insightface.get(img_array)
                    
                    if faces and len(faces) > 0:
                        # Take the largest face
                        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        bbox = face.bbox.astype(int)
                        
                        # Quick crop with margin
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        margin_x = int(width * margin_percent)
                        margin_y = int(height * margin_percent)
                        
                        img_width, img_height = img.size
                        x1 = max(0, bbox[0] - margin_x)
                        y1 = max(0, bbox[1] - margin_y)
                        x2 = min(img_width, bbox[2] + margin_x)
                        y2 = min(img_height, bbox[3] + margin_y)
                        
                        # Crop and resize in one step
                        cropped_img = img.crop((x1, y1, x2, y2))
                        
                        # Optional alignment (disabled by default for speed)
                        if self.enable_alignment and hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                            cropped_img = self._apply_alignment(cropped_img, face.landmark_2d_106, x1, y1)
                        
                        # Final resize
                        return cropped_img.resize((target_size, target_size), Image.LANCZOS)
                        
                except Exception as e:
                    pass  # Fall through to MTCNN or full image
            
            # Fallback to MTCNN
            if self.mtcnn is not None:
                try:
                    boxes, probs = self.mtcnn.detect([img])
                    
                    if boxes[0] is not None and len(boxes[0]) > 0:
                        box = boxes[0][0]  # Take first detection
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        
                        # Apply margin and crop
                        width = x2 - x1
                        height = y2 - y1
                        margin_x = int(width * margin_percent)
                        margin_y = int(height * margin_percent)
                        
                        img_width, img_height = img.size
                        x1 = max(0, x1 - margin_x)
                        y1 = max(0, y1 - margin_y)
                        x2 = min(img_width, x2 + margin_x)
                        y2 = min(img_height, y2 + margin_y)
                        
                        cropped_img = img.crop((x1, y1, x2, y2))
                        return cropped_img.resize((target_size, target_size), Image.LANCZOS)
                        
                except Exception as e:
                    pass
            
            # No face detected - resize full image
            return img.resize((target_size, target_size), Image.LANCZOS)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

    def _apply_alignment(self, cropped_img: Image.Image, landmarks: np.ndarray, 
                        x_offset: int, y_offset: int) -> Image.Image:
        """
        Simple eye-based alignment (optional for speed)
        """
        try:
            # Convert landmarks to cropped image coordinates
            landmarks = landmarks - np.array([x_offset, y_offset])
            
            # Get eye coordinates (simplified)
            left_eye = np.mean(landmarks[33:42], axis=0)
            right_eye = np.mean(landmarks[89:96], axis=0)
            
            # Calculate rotation angle
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Only apply small rotations to avoid artifacts
            if abs(angle) < 30:
                img_array = np.array(cropped_img)
                h, w = img_array.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img_array, M, (w, h))
                return Image.fromarray(rotated)
            
        except Exception:
            pass
        
        return cropped_img

    def batch_process_optimized(self, input_dir: str, output_dir: str, 
                               batch_size: int = 64, margin_percent: float = 0.35,
                               target_size: int = 112, start_from: int = 0, 
                               end_at: Optional[int] = None) -> int:
        """
        Optimized batch processing with better memory management
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Apply limits
        if end_at is not None:
            image_files = image_files[start_from:end_at]
        else:
            image_files = image_files[start_from:]
        
        total_images = len(image_files)
        print(f"Processing {total_images} images with optimized settings")
        
        success_count = 0
        failed_count = 0
        start_time = time.time()
        
        # Process with progress bar
        with tqdm(total=total_images, desc="Processing faces", 
                 unit="img", smoothing=0.1) as pbar:
            
            for i in range(0, total_images, batch_size):
                batch_files = image_files[i:i + batch_size]
                
                # Process batch
                for filename in batch_files:
                    try:
                        input_path = os.path.join(input_dir, filename)
                        output_filename = f"{os.path.splitext(filename)[0]}_112x112.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Skip if already exists
                        if os.path.exists(output_path):
                            success_count += 1
                            pbar.update(1)
                            continue
                        
                        # Process image
                        processed_img = self.preprocess_image_fast(
                            input_path, margin_percent, target_size
                        )
                        
                        if processed_img is not None:
                            processed_img.save(output_path, format='JPEG', quality=90)
                            success_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        failed_count += 1
                    
                    pbar.update(1)
                
                # Clear GPU cache periodically
                if torch is not None and torch.cuda.is_available() and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                
                # Update speed in progress bar
                elapsed = time.time() - start_time
                speed = (success_count + failed_count) / elapsed if elapsed > 0 else 0
                pbar.set_postfix({"Speed": f"{speed:.1f} img/s", "Failed": failed_count})
        
        elapsed_time = time.time() - start_time
        avg_speed = total_images / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nProcessing complete:")
        print(f"  Success: {success_count}/{total_images} ({100*success_count/total_images:.1f}%)")
        print(f"  Failed: {failed_count}")
        print(f"  Average speed: {avg_speed:.1f} images/second")
        print(f"  Total time: {elapsed_time:.1f} seconds")
        
        return success_count


def process_gallery_optimized(input_dir: str, output_dir: str, 
                             batch_size: int = 64, fast_mode: bool = True,
                             enable_alignment: bool = False):
    """
    Process gallery with optimized settings for speed
    """
    device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
    
    processor = OptimizedFacePreprocessor(
        device=device, 
        fast_mode=fast_mode,
        enable_alignment=enable_alignment
    )
    
    success_count = processor.batch_process_optimized(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        margin_percent=0.35,
        target_size=112
    )
    
    return success_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized face preprocessing for speed")
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed images')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--fast_mode', action='store_true', default=True, 
                       help='Use faster detection settings')
    parser.add_argument('--enable_alignment', action='store_true', default=False,
                       help='Enable face alignment (slower but better quality)')
    parser.add_argument('--start_from', type=int, default=0, help='Start from this index')
    parser.add_argument('--end_at', type=int, default=None, help='End at this index')

    args = parser.parse_args()

    device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
    processor = OptimizedFacePreprocessor(
        device=device,
        fast_mode=args.fast_mode,
        enable_alignment=args.enable_alignment
    )
    
    processed_count = processor.batch_process_optimized(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        start_from=args.start_from,
        end_at=args.end_at
    )

    print(f"Successfully processed {processed_count} images")
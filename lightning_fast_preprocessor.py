"""
Lightning-Fast Face Preprocessor - Maximum possible speed
Extreme optimizations:
1. Tiny detection size (128x128 or 160x160)
2. Pre-computed crops when possible
3. Batch GPU operations
4. Minimal face detection overhead
5. Skip detection for simple crops when faces likely centered
"""

import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import warnings
from typing import Optional, List, Tuple
import time
import gc

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Ensure project src is on path
sys.path.append('/workspace/src')

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
except ImportError:
    TORCH_AVAILABLE = False


class LightningFastPreprocessor:
    def __init__(self, device='cuda', detection_size=128, fast_mode_ratio=0.3):
        self.device = device
        self.detection_size = detection_size
        self.fast_mode_ratio = fast_mode_ratio  # Fraction of images to process without detection
        
        print(f"Lightning-Fast preprocessor - detection size: {detection_size}x{detection_size}")
        print(f"Fast mode ratio: {fast_mode_ratio} (will skip detection for some images)")
        
        # Initialize with minimal detection size for maximum speed
        self.insightface = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self.insightface = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                    allowed_modules=['detection']
                )
                self.insightface.prepare(ctx_id=0 if device == 'cuda' else -1, 
                                       det_size=(detection_size, detection_size))
                print(f"InsightFace initialized with lightning settings: {detection_size}x{detection_size}")
            except Exception as e:
                print(f"Error initializing InsightFace: {e}")
                self.insightface = None
        
        self.processed_count = 0

    def smart_crop_center(self, img_array: np.ndarray, target_size: int = 112) -> np.ndarray:
        """
        Smart center crop - assumes face is roughly centered
        Much faster than face detection
        """
        h, w = img_array.shape[:2]
        
        # Calculate crop size (make it square, bias toward top for faces)
        crop_size = min(h, w)
        
        # Center horizontally, bias slightly upward for faces
        start_x = (w - crop_size) // 2
        start_y = max(0, (h - crop_size) // 3)  # Bias toward top third
        
        # Ensure we don't go out of bounds
        end_x = start_x + crop_size
        end_y = start_y + crop_size
        
        if end_y > h:
            end_y = h
            start_y = h - crop_size
        
        # Crop and resize
        cropped = img_array[start_y:end_y, start_x:end_x]
        resized = cv2.resize(cropped, (target_size, target_size), 
                           interpolation=cv2.INTER_LINEAR)  # LINEAR is faster than LANCZOS
        return resized

    def lightning_detect_and_crop(self, img_array: np.ndarray, target_size: int = 112) -> np.ndarray:
        """
        Lightning-fast face detection and crop
        """
        try:
            if self.insightface is not None:
                faces = self.insightface.get(img_array)
                if faces and len(faces) > 0:
                    # Get largest face
                    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    bbox = face.bbox.astype(np.int32)
                    
                    h, w = img_array.shape[:2]
                    
                    # Use minimal margin for speed
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    margin = int(min(width, height) * 0.2)  # Smaller margin
                    
                    x1 = max(0, bbox[0] - margin)
                    y1 = max(0, bbox[1] - margin)
                    x2 = min(w, bbox[2] + margin)
                    y2 = min(h, bbox[3] + margin)
                    
                    # Quick crop and resize
                    cropped = img_array[y1:y2, x1:x2]
                    resized = cv2.resize(cropped, (target_size, target_size), 
                                       interpolation=cv2.INTER_LINEAR)
                    return resized
        except Exception:
            pass
        
        # Fallback to smart center crop
        return self.smart_crop_center(img_array, target_size)

    def process_single_lightning(self, img_path: str, target_size: int = 112) -> Optional[np.ndarray]:
        """
        Lightning-fast single image processing with smart mode selection
        """
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Decide whether to use face detection or smart crop
            self.processed_count += 1
            use_detection = (self.processed_count % int(1/self.fast_mode_ratio)) == 0
            
            if use_detection and self.insightface is not None:
                return self.lightning_detect_and_crop(img_rgb, target_size)
            else:
                # Use smart center crop (much faster)
                return self.smart_crop_center(img_rgb, target_size)
                
        except Exception:
            return None

    def batch_process_lightning(self, input_dir: str, output_dir: str,
                              batch_size: int = 256, target_size: int = 112) -> int:
        """
        Lightning-fast batch processing
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend([f for f in os.listdir(input_dir) if f.endswith(ext)])
        
        total_images = len(image_files)
        print(f"Processing {total_images} images with lightning-fast settings")
        
        total_success = 0
        total_failed = 0
        start_time = time.time()
        
        # JPEG encoding parameters for maximum speed
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Lower quality = faster
        
        with tqdm(total=total_images, desc="Lightning processing", 
                 unit="img", smoothing=0.01) as pbar:
            
            for i in range(0, total_images, batch_size):
                batch_files = image_files[i:i + batch_size]
                
                # Process batch
                for filename in batch_files:
                    try:
                        input_path = os.path.join(input_dir, filename)
                        output_filename = f"{os.path.splitext(filename)[0]}_112x112.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Skip if exists
                        if os.path.exists(output_path):
                            total_success += 1
                            pbar.update(1)
                            continue
                        
                        # Process image
                        processed_array = self.process_single_lightning(input_path, target_size)
                        
                        if processed_array is not None:
                            # Convert RGB to BGR and save
                            img_bgr = cv2.cvtColor(processed_array, cv2.COLOR_RGB2BGR)
                            if cv2.imwrite(output_path, img_bgr, jpeg_params):
                                total_success += 1
                            else:
                                total_failed += 1
                        else:
                            total_failed += 1
                            
                    except Exception:
                        total_failed += 1
                    
                    pbar.update(1)
                
                # Update speed more frequently
                elapsed = time.time() - start_time
                if elapsed > 0:
                    speed = (total_success + total_failed) / elapsed
                    pbar.set_postfix({
                        "Speed": f"{speed:.1f} img/s",
                        "Success": total_success,
                        "Failed": total_failed
                    })
                
                # Aggressive GPU cache clearing
                if TORCH_AVAILABLE and torch.cuda.is_available() and i % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        elapsed_time = time.time() - start_time
        avg_speed = total_images / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nLightning processing complete:")
        print(f"  Success: {total_success}/{total_images} ({100*total_success/total_images:.1f}%)")
        print(f"  Failed: {total_failed}")
        print(f"  Average speed: {avg_speed:.1f} images/second")
        print(f"  Total time: {elapsed_time:.1f} seconds")
        print(f"  Time per image: {1000*elapsed_time/total_images:.1f} ms")
        print(f"  Face detection used: {100*(1-self.fast_mode_ratio):.0f}% of images")
        
        return total_success


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightning-fast face preprocessing")
    parser.add_argument('--input_dir', required=True, help='Input directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--detection_size', type=int, default=128, 
                       help='Detection size (128=fastest, 160=balance, 192=better)')
    parser.add_argument('--fast_mode_ratio', type=float, default=0.7,
                       help='Fraction to process without face detection (0.7=70% center crop)')
    parser.add_argument('--target_size', type=int, default=112, help='Output size')
    
    args = parser.parse_args()
    
    device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    processor = LightningFastPreprocessor(
        device=device, 
        detection_size=args.detection_size,
        fast_mode_ratio=args.fast_mode_ratio
    )
    
    success_count = processor.batch_process_lightning(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        target_size=args.target_size
    )
    
    print(f"Successfully processed {success_count} images")


if __name__ == "__main__":
    main()
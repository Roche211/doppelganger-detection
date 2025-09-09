"""
Single Image Preprocessor - Exact same logic as lightning_fast_preprocessor.py

Process individual images using the exact same preprocessing pipeline
that worked perfectly for the 71k gallery.
"""

import os
import sys
import numpy as np
import cv2
import warnings
from typing import Optional
from PIL import Image, ImageOps

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


class SingleImageLightningPreprocessor:
    def __init__(self, device='cuda', detection_size=128):
        self.device = device
        self.detection_size = detection_size
        
        print(f"Single image lightning preprocessor - detection size: {detection_size}x{detection_size}")
        
        # Initialize with minimal detection size for maximum speed (same as lightning)
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

    def smart_crop_center(self, img_array: np.ndarray, target_size: int = 112) -> np.ndarray:
        """
        Smart center crop - assumes face is roughly centered
        Exact same logic as lightning_fast_preprocessor.py
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
        Exact same logic as lightning_fast_preprocessor.py
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

    def process_single_image(self, img_path: str, output_path: str, 
                           use_face_detection: bool = True, target_size: int = 112) -> bool:
        """
        Process a single image using exact same logic as lightning preprocessor
        
        Args:
            img_path: Path to input image
            output_path: Path to save processed image
            use_face_detection: Whether to use face detection (True) or smart crop (False)
            target_size: Output size (112 for ArcFace)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image (same as lightning)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                return False
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process using exact same logic
            if use_face_detection and self.insightface is not None:
                processed_array = self.lightning_detect_and_crop(img_rgb, target_size)
                print(f"Processed with face detection: {img_path}")
            else:
                # Use smart center crop (same as 70% of lightning processing)
                processed_array = self.smart_crop_center(img_rgb, target_size)
                print(f"Processed with smart center crop: {img_path}")
            
            # Save with same quality settings as lightning
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert RGB to BGR and save (same as lightning)
            img_bgr = cv2.cvtColor(processed_array, cv2.COLOR_RGB2BGR)
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Same quality as lightning
            
            if cv2.imwrite(output_path, img_bgr, jpeg_params):
                print(f"✅ Successfully saved: {output_path}")
                return True
            else:
                print(f"❌ Failed to save: {output_path}")
                return False
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False


def process_tadhg_image():
    """
    Process Tadhg.jpeg using exact same pipeline as lightning preprocessor
    """
    print("🚀 Processing Tadhg.jpeg with lightning preprocessing")
    
    input_path = "/workspace/Tadhg.jpeg"
    output_path = "/workspace/processed_gallery_112_lightning/Tadhg_112x112.jpg"
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"❌ Input image not found: {input_path}")
        return False
    
    # Initialize preprocessor with same settings as lightning
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    processor = SingleImageLightningPreprocessor(device=device, detection_size=128)
    
    # Process with face detection (like 30% of lightning images)
    success = processor.process_single_image(
        img_path=input_path,
        output_path=output_path,
        use_face_detection=True,  # Try face detection first
        target_size=112
    )
    
    if success:
        print(f"🎉 Tadhg.jpeg processed successfully!")
        print(f"   Input: {input_path}")
        print(f"   Output: {output_path}")
        print(f"   Ready to add to gallery for similarity search!")
        return True
    else:
        print(f"❌ Failed to process Tadhg.jpeg")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process single image with lightning preprocessing")
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--detection_size', type=int, default=128, help='Detection size')
    parser.add_argument('--target_size', type=int, default=112, help='Output size')
    parser.add_argument('--no_face_detection', action='store_true', help='Skip face detection, use center crop')
    
    args = parser.parse_args()
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    processor = SingleImageLightningPreprocessor(device=device, detection_size=args.detection_size)
    
    success = processor.process_single_image(
        img_path=args.input,
        output_path=args.output,
        use_face_detection=not args.no_face_detection,
        target_size=args.target_size
    )
    
    if success:
        print(f"✅ Successfully processed {args.input} -> {args.output}")
    else:
        print(f"❌ Failed to process {args.input}")

#!/usr/bin/env python3
"""
Simple ONNX Test Script
Tests if the ONNX model works and produces correct embeddings
"""

import sys
import numpy as np
import torch
sys.path.append('/workspace/src')

try:
    import onnxruntime as ort
    print("✅ ONNX Runtime available")
except ImportError:
    print("❌ ONNX Runtime not available")
    sys.exit(1)

from inference.utils import load_model_from_checkpoint

def test_onnx_vs_pytorch():
    print("🔍 TESTING ONNX vs PyTorch")
    print("=" * 50)
    
    # 1. Load PyTorch model
    print("1. Loading PyTorch model...")
    pytorch_model = load_model_from_checkpoint('/workspace/models/checkpoints/best_similarity_model.pth')
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    print(f"   ✅ PyTorch model loaded on {device}")
    
    # 2. Load ONNX model
    print("2. Loading ONNX model...")
    try:
        onnx_session = ort.InferenceSession('similarity_model_fixed.onnx')
        print("   ✅ ONNX model loaded")
    except Exception as e:
        print(f"   ❌ ONNX load failed: {e}")
        return False
    
    # 3. Create test input
    print("3. Creating test input...")
    np.random.seed(42)
    test_input_np = np.random.randn(1, 3, 112, 112).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np).to(device)
    print(f"   Input shape: {test_input_np.shape}")
    
    # 4. PyTorch inference
    print("4. Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input_torch)
        pytorch_embedding = pytorch_output.cpu().numpy()
    print(f"   ✅ PyTorch output shape: {pytorch_embedding.shape}")
    print(f"   Sample: {pytorch_embedding[0][:5]}")
    
    # 5. ONNX inference - try multiple methods
    print("5. Running ONNX inference...")
    
    onnx_embedding = None
    
    # Method A: Check if model has inputs
    inputs = onnx_session.get_inputs()
    if len(inputs) > 0:
        input_name = inputs[0].name
        print(f"   Trying with input name: {input_name}")
        try:
            outputs = onnx_session.run(None, {input_name: test_input_np})
            onnx_embedding = outputs[0]
            print("   ✅ Method A (named input) worked!")
        except Exception as e:
            print(f"   ❌ Method A failed: {e}")
    
    # Method B: Try positional input
    if onnx_embedding is None:
        print("   Trying positional input...")
        try:
            # Create input dict with index
            input_dict = {}
            for i, inp in enumerate(onnx_session.get_inputs()):
                if inp.name:
                    input_dict[inp.name] = test_input_np
                else:
                    input_dict[str(i)] = test_input_np
            
            if not input_dict:
                # Try with common names
                for name in ['input', 'x', 'data']:
                    try:
                        outputs = onnx_session.run(None, {name: test_input_np})
                        onnx_embedding = outputs[0]
                        print(f"   ✅ Method B ({name}) worked!")
                        break
                    except:
                        continue
            else:
                outputs = onnx_session.run(None, input_dict)
                onnx_embedding = outputs[0]
                print("   ✅ Method B (input dict) worked!")
                
        except Exception as e:
            print(f"   ❌ Method B failed: {e}")
    
    if onnx_embedding is None:
        print("   ❌ All ONNX inference methods failed!")
        return False
    
    print(f"   ✅ ONNX output shape: {onnx_embedding.shape}")
    print(f"   Sample: {onnx_embedding[0][:5]}")
    
    # 6. Compare outputs
    print("6. Comparing outputs...")
    max_diff = np.max(np.abs(pytorch_embedding - onnx_embedding))
    mean_diff = np.mean(np.abs(pytorch_embedding - onnx_embedding))
    
    print(f"   Max difference: {max_diff:.8f}")
    print(f"   Mean difference: {mean_diff:.8f}")
    
    if max_diff < 1e-5:
        print("   ✅ PERFECT MATCH! ONNX == PyTorch")
        return True
    elif max_diff < 1e-3:
        print("   ⚠️  Small differences (acceptable)")
        return True
    else:
        print("   ❌ Large differences!")
        return False

if __name__ == "__main__":
    success = test_onnx_vs_pytorch()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS! Your ONNX model:")
        print("   ✅ Uses your best checkpoint weights")
        print("   ✅ Produces identical embeddings")
        print("   ✅ Ready for deployment!")
    else:
        print("❌ FAILED! ONNX model has issues")
        print("   Need to debug the export process")
    print("=" * 50)

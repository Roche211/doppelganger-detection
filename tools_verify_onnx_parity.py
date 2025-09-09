#!/usr/bin/env python3
"""
Verify ONNX parity w.r.t. PyTorch on a single 112x112 image.
Ensures cosine similarity ~ 1.0.

Usage:
  python tools_verify_onnx_parity.py \
    --image /workspace/processed_gallery_112_lightning/TEST_112x112.jpg \
    --checkpoint /workspace/models/checkpoints/best_similarity_model.pth \
    --onnx /workspace/similarity_full.onnx
"""
import os, sys, argparse, numpy as np, cv2
sys.path.append('/workspace/src')
import torch
from inference.utils import load_model_from_checkpoint, load_image_as_tensor
import onnxruntime as ort

def embed_pytorch(img112: str, checkpoint: str) -> np.ndarray:
    model = load_model_from_checkpoint(checkpoint)
    model.eval()
    t = load_image_as_tensor(img112)   # ToTensor+Normalize(0.5)
    if t is None:
        raise FileNotFoundError(f"Image not found or unreadable: {img112}")
    t = (t + 1) / 2.0                  # back to [0,1]
    device = next(model.parameters()).device
    t = t.to(device)
    with torch.no_grad():
        e = model(t).cpu().numpy()[0]
    e = e / (np.linalg.norm(e)+1e-12)
    return e

def embed_onnx(img112: str, onnx_path: str) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    inp = sess.get_inputs()[0].name
    img_bgr = cv2.imread(img112)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {img112}")

    # Prepare RGB path
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x_rgb = np.transpose((img_rgb - 127.5)/127.5, (2,0,1))[None, ...]
    y_rgb = sess.run(None, {inp: x_rgb})[0][0]
    y_rgb = y_rgb / (np.linalg.norm(y_rgb)+1e-12)

    # Prepare BGR path (no color swap)
    img_bgr_f = img_bgr.astype(np.float32)
    x_bgr = np.transpose((img_bgr_f - 127.5)/127.5, (2,0,1))[None, ...]
    y_bgr = sess.run(None, {inp: x_bgr})[0][0]
    y_bgr = y_bgr / (np.linalg.norm(y_bgr)+1e-12)

    return y_rgb, y_bgr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--onnx', required=True)
    args = ap.parse_args()

    e_pt = embed_pytorch(args.image, args.checkpoint)
    e_rgb, e_bgr = embed_onnx(args.image, args.onnx)

    cos_rgb = float(np.dot(e_pt, e_rgb) / (np.linalg.norm(e_pt)*np.linalg.norm(e_rgb)))
    cos_bgr = float(np.dot(e_pt, e_bgr) / (np.linalg.norm(e_pt)*np.linalg.norm(e_bgr)))
    print(f"Cosine(Pytorch, ONNX RGB) = {cos_rgb:.6f}")
    print(f"Cosine(Pytorch, ONNX BGR) = {cos_bgr:.6f}")
    best = max(cos_rgb, cos_bgr)
    if best > 0.999:
        print("✅ Parity OK")
    else:
        print("❌ Parity LOW — check normalization or projection orientation")

if __name__=='__main__':
    main()

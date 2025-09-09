#!/usr/bin/env python3
"""
Compose final similarity ONNX:
- Default input: BGR, [N,3,112,112] float32, normalized as (x-127.5)/127.5
- Backbone: InsightFace buffalo_l recognition r50 ONNX
- Head: projection 512x512 from best_similarity_model.pth
- Output: [N,512] L2-normalized embeddings

Usage:
  python tools_compose_similarity_onnx.py \
    --backbone /root/.insightface/models/buffalo_l/w600k_r50.onnx \
    --checkpoint /workspace/models/checkpoints/best_similarity_model.pth \
    --output /workspace/similarity_full.onnx \
    [--expects_rgb]  # If set, exported ONNX will accept RGB and internally swap to BGR
"""
import os, sys, argparse, numpy as np
import onnx, onnx.helper as oh, onnx.numpy_helper as onh
from onnx import TensorProto
sys.path.append('/workspace/src')
import torch

def extract_projection(checkpoint: str) -> np.ndarray:
    ckpt = torch.load(checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    for k in ['projection.weight','model.projection.weight','module.projection.weight']:
        if k in state:
            w = state[k].detach().cpu().numpy().astype(np.float32)
            assert w.shape == (512,512), f"Unexpected projection shape {w.shape}"
            return w
    # fallback search
    for k,v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim==2 and tuple(v.shape)==(512,512):
            return v.detach().cpu().numpy().astype(np.float32)
    raise RuntimeError('Projection weights (512x512) not found in checkpoint')

def add_rgb_input_swap(model: onnx.ModelProto) -> None:
    """
    Modify model in-place to accept RGB input and internally swap channels to BGR
    before feeding the original backbone input.
    """
    # Original backbone input
    orig_in = model.graph.input[0].name
    # New external input (RGB)
    rgb_in = 'input_rgb'
    # Replace graph input with RGB input
    model.graph.input.clear()
    model.graph.input.extend([oh.make_tensor_value_info(rgb_in, TensorProto.FLOAT, [None,3,112,112])])
    # Channel indices for RGB->BGR
    idx = onh.from_array(np.array([2,1,0], dtype=np.int64), name='rgb2bgr_idx')
    gather = oh.make_node('Gather', inputs=[rgb_in, 'rgb2bgr_idx'], outputs=[orig_in], axis=1, name='RGB2BGR')
    # Insert initializer and node at the start
    model.graph.initializer.extend([idx])
    model.graph.node.insert(0, gather)

def compose(backbone_path: str, checkpoint: str, output_path: str, expects_rgb: bool = False):
    assert os.path.exists(backbone_path), f"Backbone not found: {backbone_path}"
    assert os.path.exists(checkpoint), f"Checkpoint not found: {checkpoint}"

    model = onnx.load(backbone_path)
    # Optionally adapt the input to accept RGB and convert to BGR internally
    if expects_rgb:
        add_rgb_input_swap(model)

    # Expect single output of size [N,512]
    assert len(model.graph.output)>=1, 'Backbone ONNX has no outputs'
    backbone_out = model.graph.output[0].name

    W = extract_projection(checkpoint)            # (512,512)
    W_t = W.T                                     # Gemm expects KxN shape here
    W_init = onh.from_array(W_t, name='proj_W')
    eps_init = onh.from_array(np.array([1e-12], np.float32), name='eps')

    proj_out = 'proj_out'
    l2 = 'l2'
    l2c = 'l2_clipped'
    out = 'embedding'

    gemm = oh.make_node('Gemm', inputs=[backbone_out,'proj_W'], outputs=[proj_out],
                        alpha=1.0, beta=1.0, transA=0, transB=0, name='Projection')
    reduce = oh.make_node('ReduceL2', inputs=[proj_out], outputs=[l2], axes=[1], keepdims=1, name='ReduceL2')
    clip = oh.make_node('Clip', inputs=[l2,'eps'], outputs=[l2c], name='ClipEps')
    div = oh.make_node('Div', inputs=[proj_out,l2c], outputs=[out], name='L2Normalize')

    model.graph.initializer.extend([W_init, eps_init])
    model.graph.node.extend([gemm, reduce, clip, div])

    # Set final output
    model.graph.output.clear()
    model.graph.output.extend([oh.make_tensor_value_info(out, TensorProto.FLOAT, [None,512])])

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    mode = 'RGB' if expects_rgb else 'BGR'
    print(f"✅ Saved: {output_path} (expects {mode} input normalized to [-1,1], CHW)")

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--expects_rgb', action='store_true', help='Make exported ONNX accept RGB input (internally swaps to BGR)')
    args = ap.parse_args()
    compose(args.backbone, args.checkpoint, args.output, expects_rgb=args.expects_rgb)

import os
import sys
import glob
import logging
from typing import List, Optional

import torch
from PIL import Image

# Ensure project src is on path when running as a script
sys.path.append('/workspace/src')

from data.preprocessing import setup_face_analyzer, preprocess_face
from inference.utils import (
    load_model_from_checkpoint,
    compute_embeddings_for_images,
    save_embeddings,
    cosine_search,
    default_transform,
    load_image_as_tensor,
)

logger = logging.getLogger(__name__)

# Optional MTCNN fallback
try:
    from facenet_pytorch import MTCNN
    _HAS_MTCNN = True
except Exception:
    _HAS_MTCNN = False


def _mtcnn_fallback_crop_112(img_path: str, mtcnn: 'MTCNN', margin: float = 0.3) -> Optional[Image.Image]:
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return None
    # MTCNN.detect can take PIL or tensor; we'll pass PIL as list to keep API parity
    boxes, probs = mtcnn.detect([img])
    if boxes is None or boxes[0] is None or len(boxes[0]) == 0:
        return None
    # pick highest prob
    import numpy as np
    idx = int(np.argmax(probs[0]))
    x1, y1, x2, y2 = boxes[0][idx]
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin
    W, H = img.size
    x1 = max(0, int(round(x1 - mx)))
    y1 = max(0, int(round(y1 - my)))
    x2 = min(W, int(round(x2 + mx)))
    y2 = min(H, int(round(y2 + my)))
    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize((112, 112))
    return crop


def list_gallery_images(gallery_dir: str) -> List[str]:
    exts = ('*.jpg', '*.jpeg', '*.png')
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(gallery_dir, e)))
    files.sort()
    return files


def preprocess_gallery_if_needed(gallery_dir: str, output_dir: str,
                                det_size: int = 640, det_thresh: float = 0.4,
                                fallback_resize: bool = True,
                                enable_mtcnn_fallback: bool = False,
                                mtcnn_margin: float = 0.3) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    # Pass det_thresh directly to setup_face_analyzer
    face_app = setup_face_analyzer(det_size=(det_size, det_size), det_thresh=det_thresh)
    mtcnn = None
    if enable_mtcnn_fallback and _HAS_MTCNN:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mtcnn = MTCNN(device=device, min_face_size=20, thresholds=[0.6, 0.7, 0.7], post_process=False, keep_all=False)

    input_images = list_gallery_images(gallery_dir)
    processed_paths: List[str] = []
    
    print(f"Processing {len(input_images)} images with InsightFace (det_thresh={det_thresh})...")
    
    # Process with progress tracking
    for i, p in enumerate(input_images):
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i + 1}/{len(input_images)} ({100*(i+1)/len(input_images):.1f}%)")
        base = os.path.splitext(os.path.basename(p))[0]
        outp = os.path.join(output_dir, f"{base}_112x112.jpg")
        if not os.path.exists(outp):
            aligned = preprocess_face(p, face_app, save_path=outp)
            if aligned is None and mtcnn is not None:
                try:
                    crop = _mtcnn_fallback_crop_112(p, mtcnn, margin=mtcnn_margin)
                    if crop is not None:
                        crop.save(outp, format='JPEG', quality=95)
                except Exception:
                    pass
            if not os.path.exists(outp) and aligned is None and fallback_resize:
                # simple fallback: resize entire image to 112x112
                try:
                    im = Image.open(p).convert('RGB')
                    im = im.resize((112, 112))
                    im.save(outp, format='JPEG', quality=95)
                except Exception:
                    pass
        if os.path.exists(outp):
            processed_paths.append(outp)
    return processed_paths


def export_gallery_embeddings(gallery_112_dir: str, checkpoint: str, out_pt: str, out_json: str,
                              batch_size: int = 256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_from_checkpoint(checkpoint, device=device)

    imgs = list_gallery_images(gallery_112_dir)
    embs, paths = compute_embeddings_for_images(model, imgs, device=device, batch_size=batch_size)
    save_embeddings(embs, paths, out_pt, out_json)


def preprocess_single_query(image_path: str, save_dir: str,
                            det_size: int = 640, det_thresh: float = 0.4,
                            fallback_resize: bool = True,
                            enable_mtcnn_fallback: bool = False,
                            mtcnn_margin: float = 0.3) -> Optional[str]:
    os.makedirs(save_dir, exist_ok=True)
    # Pass det_thresh directly to setup_face_analyzer
    face_app = setup_face_analyzer(det_size=(det_size, det_size), det_thresh=det_thresh)
    mtcnn = None
    if enable_mtcnn_fallback and _HAS_MTCNN:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mtcnn = MTCNN(device=device, min_face_size=20, thresholds=[0.6, 0.7, 0.7], post_process=False, keep_all=False)
    base = os.path.splitext(os.path.basename(image_path))[0]
    outp = os.path.join(save_dir, f"{base}_112x112.jpg")
    aligned = preprocess_face(image_path, face_app, save_path=outp)
    if aligned is None and mtcnn is not None:
        try:
            crop = _mtcnn_fallback_crop_112(image_path, mtcnn, margin=mtcnn_margin)
            if crop is not None:
                crop.save(outp, format='JPEG', quality=95)
        except Exception:
            pass
    if not os.path.exists(outp) and aligned is None and fallback_resize:
        try:
            im = Image.open(image_path).convert('RGB')
            im = im.resize((112, 112))
            im.save(outp, format='JPEG', quality=95)
        except Exception:
            return None
    return outp if os.path.exists(outp) else None


def search_queries_against_db(queries: List[str], db_pt: str, checkpoint: str, top_k: int = 5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_from_checkpoint(checkpoint, device=device)
    payload = torch.load(db_pt, map_location=device)
    db_embs: torch.Tensor = payload['embeddings']
    db_paths: List[str] = payload['paths']

    for q in queries:
        tensor = load_image_as_tensor(q)
        if tensor is None:
            logger.warning(f"Skipping query (failed load): {q}")
            continue
        tensor = tensor.to(device)
        tensor = (tensor + 1) / 2.0
        with torch.no_grad():
            qemb = model(tensor)
        sims, idxs = cosine_search(qemb, db_embs.to(device), top_k=top_k)
        print(f"\nQuery: {q}")
        for rank, (s, idx) in enumerate(zip(sims.cpu().tolist(), idxs.cpu().tolist()), start=1):
            print(f"  {rank}. {db_paths[idx]}  (cos={s:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery_raw', type=str, default='/workspace/extracted_images')
    parser.add_argument('--gallery_112', type=str, default='/workspace/processed_gallery_112')
    parser.add_argument('--checkpoint', type=str, default='/workspace/models/checkpoints/best_similarity_model.pth')
    parser.add_argument('--db_pt', type=str, default='/workspace/models/checkpoints/gallery_embeddings.pt')
    parser.add_argument('--db_json', type=str, default='/workspace/models/checkpoints/gallery_paths.json')
    parser.add_argument('--prepare_only', action='store_true')
    parser.add_argument('--export_only', action='store_true')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--det_size', type=int, default=640)
    parser.add_argument('--det_thresh', type=float, default=0.35)
    parser.add_argument('--no_fallback_resize', action='store_true')
    parser.add_argument('--enable_mtcnn_fallback', action='store_true')
    parser.add_argument('--mtcnn_margin', type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db_pt), exist_ok=True)

    gallery_paths = preprocess_gallery_if_needed(
        args.gallery_raw,
        args.gallery_112,
        det_size=args.det_size,
        det_thresh=args.det_thresh,
        fallback_resize=not args.no_fallback_resize,
        enable_mtcnn_fallback=args.enable_mtcnn_fallback,
        mtcnn_margin=args.mtcnn_margin,
    )
    print(f"Processed/kept: {len(gallery_paths)} images in {args.gallery_112}")
    if args.prepare_only:
        raise SystemExit(0)

    export_gallery_embeddings(args.gallery_112, args.checkpoint, args.db_pt, args.db_json)
    if args.export_only:
        raise SystemExit(0)

    # Queries: process like gallery
    q1 = preprocess_single_query(
        '/workspace/chris.jpeg', args.gallery_112,
        det_size=args.det_size, det_thresh=args.det_thresh,
        fallback_resize=not args.no_fallback_resize,
        enable_mtcnn_fallback=args.enable_mtcnn_fallback,
        mtcnn_margin=args.mtcnn_margin,
    )
    q2 = preprocess_single_query(
        '/workspace/myself.jpeg', args.gallery_112,
        det_size=args.det_size, det_thresh=args.det_thresh,
        fallback_resize=not args.no_fallback_resize,
        enable_mtcnn_fallback=args.enable_mtcnn_fallback,
        mtcnn_margin=args.mtcnn_margin,
    )
    queries = [p for p in [q1, q2] if p is not None]
    search_queries_against_db(queries, args.db_pt, args.checkpoint, top_k=args.top_k)



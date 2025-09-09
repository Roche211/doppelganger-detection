import os
import json
import logging
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.similarity_arcface import setup_similarity_model

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: Optional[str] = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = setup_similarity_model(device=device, embedding_dim=512, freeze_backbone=True)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded model weights from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}; using current weights")
    model.eval()
    return model


def default_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def load_image_as_tensor(img_path: str, transform=None) -> Optional[torch.Tensor]:
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to open image {img_path}: {e}")
        return None
    transform = transform or default_transform()
    tensor = transform(img).unsqueeze(0)
    return tensor


def compute_embeddings_for_images(model, image_paths: List[str], device: Optional[str] = None,
                                  batch_size: int = 128) -> Tuple[torch.Tensor, List[str]]:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    transform = default_transform()

    valid_paths: List[str] = []
    tensors: List[torch.Tensor] = []
    for p in image_paths:
        t = load_image_as_tensor(p, transform=transform)
        if t is not None:
            valid_paths.append(p)
            tensors.append(t)

    if not tensors:
        return torch.empty(0, 512), []

    images = torch.cat(tensors, dim=0).to(device)

    # Model expects [0,1] or [-1,1], we used Normalize to [-1,1] around 0.5: scale back to [0,1]
    images = (images + 1) / 2.0

    all_embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i+batch_size]
            emb = model(batch)
            all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, valid_paths


def save_embeddings(embeddings: torch.Tensor, paths: List[str], out_pt: str, out_json: str):
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    torch.save({'embeddings': embeddings, 'paths': paths}, out_pt)
    with open(out_json, 'w') as f:
        json.dump({'paths': paths, 'num': len(paths)}, f, indent=2)
    logger.info(f"Saved embeddings: {embeddings.shape} -> {out_pt}")


def cosine_search(query_emb: torch.Tensor, db_embs: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    query = F.normalize(query_emb, p=2, dim=1)
    db = F.normalize(db_embs, p=2, dim=1)
    sims = query @ db.T
    vals, idxs = torch.topk(sims, k=min(top_k, db.shape[0]), dim=1)
    return vals.squeeze(0), idxs.squeeze(0)










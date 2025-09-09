## Face Similarity ONNX Deployment Guide

This guide explains how to use the exported ONNX models for embedding generation and similarity search, and how to preprocess images to the required 112x112 face crops.

### Files
- `similarity_full.onnx` — expects BGR input (OpenCV style), float32 CHW, normalized to [-1, 1].
- `similarity_full_rgb.onnx` — expects RGB input (PIL/most web stacks), float32 CHW, normalized to [-1, 1].
- `single_image_preprocessor.py` — single-image preprocessor that produces 112x112 JPEG face crops using the same "lightning" logic as the gallery.

Pick ONE ONNX that matches your image-loading pipeline:
- Use `similarity_full_rgb.onnx` if you read images as RGB (e.g., PIL, web frameworks).
- Use `similarity_full.onnx` if you read images as BGR (OpenCV `cv2.imread`).

Both ONNX models output 512-D L2-normalized embeddings (cosine similarity can be computed via dot product).

---

### 1) Preprocess images to 112x112 face crops
The model requires 112x112 face crops. Do not pass raw images directly.

Use `single_image_preprocessor.py`:

```bash
python /workspace/single_image_preprocessor.py \
  --input /path/to/input.jpg \
  --output /path/to/output_112x112.jpg \
  --detection_size 128 \
  --target_size 112
```

Notes:
- It attempts fast InsightFace detection; if it fails, it falls back to a smart center crop (same as the lightning pipeline).
- It saves a 112x112 JPEG ready for embedding.
- If you need batch processing at scale, use your gallery preprocessor (same lightning logic) or the enhanced preprocessor.

---

### Quick install (pip)

- GPU:
```bash
pip install -U onnxruntime-gpu numpy pillow opencv-python-headless insightface tqdm
```

- CPU:
```bash
pip install -U onnxruntime numpy pillow opencv-python-headless insightface tqdm
```

- ONNX-only (no detection/preprocessing):
```bash
pip install -U onnxruntime-gpu numpy pillow tqdm
# or use onnxruntime instead of onnxruntime-gpu for CPU-only
```

---

### 2) ONNX Runtime setup (GPU/CPU)

```python
import onnxruntime as ort
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # GPU then CPU fallback
sess = ort.InferenceSession('/path/to/similarity_full_rgb.onnx', providers=providers)
```

- If CUDA is not available, ORT will fall back to CPU.
- For BGR pipelines, load `similarity_full.onnx` instead.

---

### 3) Embedding a single preprocessed image

RGB model (`similarity_full_rgb.onnx`):

```python
import onnxruntime as ort, numpy as np
from PIL import Image

sess = ort.InferenceSession('similarity_full_rgb.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
inp = sess.get_inputs()[0].name

img = Image.open('face_112x112.jpg').convert('RGB')
x = np.asarray(img, dtype=np.float32)
x = ((x - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]  # [1,3,112,112]

emb = sess.run(None, {inp: x})[0]  # (1,512), L2-normalized
```

BGR model (`similarity_full.onnx`):

```python
import onnxruntime as ort, numpy as np, cv2

sess = ort.InferenceSession('similarity_full.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
inp = sess.get_inputs()[0].name

img = cv2.imread('face_112x112.jpg')  # BGR
x = ((img.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]

emb = sess.run(None, {inp: x})[0]  # (1,512), L2-normalized
```

---

### 4) Embedding a gallery (batch loop) and saving

RGB model example:

```python
import os, json, numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

sess = ort.InferenceSession('similarity_full_rgb.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
inp = sess.get_inputs()[0].name

gallery_dir = '/path/to/processed_gallery_112'  # 112x112 JPGs
paths = [os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
embs = np.zeros((len(paths), 512), dtype=np.float32)

for i, p in enumerate(tqdm(paths)):
    img = Image.open(p).convert('RGB')
    x = np.asarray(img, dtype=np.float32)
    x = ((x - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]
    y = sess.run(None, {inp: x})[0][0]
    embs[i] = y / (np.linalg.norm(y) + 1e-12)  # normalization is redundant but safe

np.save('/path/to/gallery_embeddings.npy', embs)
with open('/path/to/gallery_paths.json', 'w') as f:
    json.dump(paths, f)
```

---

### 5) Querying: cosine top-K search

```python
import json, numpy as np
from PIL import Image
import onnxruntime as ort

# Load ONNX session
sess = ort.InferenceSession('similarity_full_rgb.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
inp = sess.get_inputs()[0].name

# Load gallery
db = np.load('/path/to/gallery_embeddings.npy')          # [N, 512]
paths = json.load(open('/path/to/gallery_paths.json'))   # [N]

# Embed a query face (112x112 RGB)
img = Image.open('query_112x112.jpg').convert('RGB')
x = np.asarray(img, dtype=np.float32)
x = ((x - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]
qe = sess.run(None, {inp: x})[0][0]
qe = qe / (np.linalg.norm(qe) + 1e-12)

# Cosine (dot) similarities
sims = db @ qe
idx = np.argsort(-sims)[:10]
for r in idx:
    print(f"{sims[r]:.4f}\t{paths[r]}")
```

For BGR pipelines (`similarity_full.onnx`), replace the image loading with OpenCV and skip RGB conversion.

---

### 6) Input contract (critical)
- Size: 112x112
- Type: float32
- Layout: CHW (channel-first)
- Normalization: `(x - 127.5) / 127.5` into [-1, 1]
- Color:
  - `similarity_full_rgb.onnx`: RGB
  - `similarity_full.onnx`: BGR
- Output: `(1, 512)` L2-normalized embedding

If embeddings are not ~unit-norm, check preprocessing and normalization first.

---

### 7) Troubleshooting
- "No face detected" during preprocessing:
  - Use the provided preprocessor; it falls back to smart center crop if detection fails.
  - Try a slightly larger `--detection_size` (e.g., 256) if your faces are tiny.
- Low parity vs PyTorch:
  - Ensure the correct ONNX model (RGB vs BGR) matches your loader.
  - Verify normalization is `(x - 127.5) / 127.5` before CHW.
- Slow inference:
  - Confirm `CUDAExecutionProvider` is active.
  - Batch inputs by stacking along dimension 0 if your app supports it.

---

### 8) What to share with a friend
- ONNX model: choose `similarity_full_rgb.onnx` (most common), or `similarity_full.onnx` for OpenCV/BGR apps.
- `single_image_preprocessor.py` for generating 112x112 face crops.
- This `ONNX_README.md` for exact usage details.

Optional: If your friend needs to build a gallery, share a tiny script that loops over a folder, embeds images, and saves `gallery_embeddings.npy` + `gallery_paths.json` (see Section 4).

---

### 9) Minimal end-to-end example (RGB model)

```python
# pip install onnxruntime-gpu Pillow tqdm numpy opencv-python-headless
import os, numpy as np, onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession('similarity_full_rgb.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
inp = sess.get_inputs()[0].name

# Embed one image
img = Image.open('face_112x112.jpg').convert('RGB')
x = np.asarray(img, dtype=np.float32)
x = ((x - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]
emb = sess.run(None, {inp: x})[0]  # (1,512)

print('Embedding shape:', emb.shape, 'L2 norm:', np.linalg.norm(emb))
```

You’re ready to deploy. Use RGB ONNX for PIL/web pipelines, BGR ONNX for OpenCV pipelines. Keep preprocessing consistent with 112x112 crops for best results.

# Doppelgänger Detection System

A face similarity system that finds visual doppelgängers by fine-tuning pre-trained ArcFace ResNet-100 for similarity learning instead of identity classification.

## 🎯 Objective

**Input**: Pre-trained ArcFace ResNet-100 (trained for identity separation)  
**Fine-tuning**: Triplet/contrastive loss (retrained for visual similarity)  
**Output**: Model that finds doppelgängers instead of rejecting them

## 📊 Technical Approach

### Data Sources (3 datasets serving 2 purposes):
- **Positive Pairs (similar faces, different people)**: 
  - SLLFW mismatched pairs (3,000 pairs)
  - HDA-Doppelgänger dataset
- **Negative Pairs (dissimilar faces)**: 
  - CelebA random sampling

### Model Architecture:
1. **Pre-trained ArcFace ResNet-100** (not from scratch)
2. **Fine-tuned with triplet loss** for visual similarity  
3. **112x112 preprocessing** using InsightFace
4. **Simple triplet/contrastive loss** (no overcomplication)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Setup
Ensure your workspace has the following structure:
```
workspace/
├── pair_SLLFW.txt           # SLLFW pair definitions
├── lfw-deepfunneled/        # LFW images
├── HDA-Doppelgaenger/       # HDA doppelgänger dataset (optional)
├── img_align_celeba/        # CelebA dataset (optional)
└── main.py                  # Main execution script
```

### 3. Run Complete Pipeline
```bash
python main.py
```

This will:
1. Parse and organize SLLFW pairs (3,000 similar-looking different people)
2. Process HDA doppelgänger pairs if available
3. Sample CelebA negatives if available
4. Train similarity model using pre-trained ArcFace ResNet-100
5. Evaluate performance

## 📁 Generated Structure

After running, the system creates:

```
workspace/
├── data/
│   └── processed/
│       ├── pairs/
│       │   ├── sllfw_positive_pairs/    # Organized SLLFW pairs
│       │   ├── hda_positive_pairs/      # Organized HDA pairs
│       │   └── celeba_negatives_112x112/ # CelebA negatives
│       └── metadata/                    # Processing statistics
├── models/
│   ├── checkpoints/                     # Training checkpoints
│   └── pretrained/                      # Pre-trained models
└── src/                                 # Source code
```

## 🔧 Configuration

Modify training parameters in `main.py`:

```python
config = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'embedding_dim': 512,
    'freeze_backbone': True,
    'unfreeze_epoch': 20
}
```

## 📊 Expected Results

- **Data**: Exactly 3,000 SLLFW pairs + HDA pairs organized correctly
- **Model**: Pre-trained ArcFace ResNet-100 loads and fine-tunes
- **Training**: Loss decreases over epochs
- **Evaluation**: Recall@5 > 70% (realistic target)
- **Output**: Model that finds similar faces instead of rejecting them

## 🧠 Key Implementation Details

### SLLFW Data Format
- 12,000 total lines in `pair_SLLFW.txt`
- 10 folds × 1,200 lines each
- **Each pair = 2 consecutive lines**
- Lines 1-600 per fold: matched pairs (SAME PERSON) - **IGNORED**
- Lines 601-1200 per fold: mismatched pairs (DIFFERENT PEOPLE, SIMILAR FACES) - **USED AS POSITIVES**

### Progressive Training Strategy
1. **Start with frozen backbone** (conservative fine-tuning)
2. **Unfreeze at epoch 20** for end-to-end training
3. **Triplet loss** for similarity learning
4. **L2 normalized embeddings** for cosine similarity

## 📈 Training Process

1. **Load pre-trained ArcFace ResNet-100**
2. **Replace identity classification head** with similarity learning head
3. **Start with frozen backbone** (first 20 epochs)
4. **Progressive unfreezing** for fine-tuning
5. **Triplet loss optimization** (anchor, positive_similar, negative_dissimilar)

## 🎯 Success Criteria

- ✅ **Data**: 3,000+ SLLFW pairs processed successfully
- ✅ **Model**: Pre-trained ArcFace loads and adapts for similarity
- ✅ **Training**: Loss convergence and improving accuracy
- ✅ **Evaluation**: Meaningful similarity retrieval performance
- ✅ **Output**: Production-ready doppelgänger detection system

## 🔍 Usage After Training

```python
from src.models.similarity_arcface import setup_similarity_model
from src.models.similarity_arcface import find_top_k_similar

# Load trained model
model = setup_similarity_model()
model.load_state_dict(torch.load('models/checkpoints/best_similarity_model.pth'))

# Find similar faces
query_embedding = model(query_image)
top_similarities, top_indices = find_top_k_similar(query_embedding, database_embeddings, k=5)
```

## 📝 Architecture Decisions

1. **Pre-trained ArcFace**: Leverages robust facial feature extraction
2. **Triplet Loss**: Simple and effective for similarity learning
3. **Progressive Training**: Conservative approach reducing overfitting risk
4. **112x112 Input**: Standard ArcFace input size
5. **L2 Normalization**: Enables cosine similarity computation
6. **Organized Pairs**: Clean data structure for efficient training

## 🚨 Important Notes

- **Focus**: Faces only - nothing else matters
- **Approach**: Fine-tune pre-trained model, don't train from scratch
- **Data**: Use SLLFW mismatched pairs as positive examples
- **Loss**: Simple triplet loss, no overcomplication
- **Validation**: Recall@K for similarity retrieval performance

## 📊 Monitoring

The system logs comprehensive statistics:
- Data preprocessing success rates
- Training loss and accuracy curves
- Validation performance metrics
- Model checkpointing and best model selection

Check `doppelganger_system.log` for detailed execution logs.

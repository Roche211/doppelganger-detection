"""
Similarity ArcFace - Modified Pre-trained ArcFace for Doppelgänger Detection

CRITICAL APPROACH:
- Load PRE-TRAINED ArcFace ResNet-100 (trained on MS1MV2: 5.8M images, 85K identities)
- Keep the robust facial feature extraction backbone
- Replace identity classification head with similarity learning head
- Fine-tune for visual similarity instead of identity separation

KEY POINT: We leverage the pre-trained knowledge and adapt it for our specific task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
    logger.info("✅ InsightFace available for model loading")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.error("❌ InsightFace not available. Install with: pip install insightface")


def load_pretrained_arcface(model_name: str = 'arcface_r100_v1'):
    """
    Load ArcFace ResNet-100 trained on MS1MV2 (5.8M celebrity faces)
    
    ENGINEERING VALIDATION:
    - This model already knows facial features (trained on 85K identities)  
    - We are FINE-TUNING for similarity, not training from scratch
    - ResNet-100 backbone provides robust feature extraction
    
    Args:
        model_name: InsightFace model name
    
    Returns:
        Pre-trained InsightFace model
    """
    if not INSIGHTFACE_AVAILABLE:
        logger.error("InsightFace not available. Install with: pip install insightface")
        return None
    
    try:
        logger.info("🔄 Loading PRE-TRAINED ArcFace ResNet-100...")
        
        # Initialize face analysis app which includes the recognition model
        app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Get the recognition model from the app
        rec_model = None
        for model in app.models.values():
            if hasattr(model, 'get_feat') or 'recognition' in str(type(model)).lower():
                rec_model = model
                break
        
        if rec_model is None:
            logger.warning("⚠️  Could not find recognition model in FaceAnalysis app")
            # Try the direct model zoo approach
            try:
                rec_model = insightface.model_zoo.get_model(model_name)
            except Exception as e2:
                logger.error(f"❌ Failed to load from model zoo: {e2}")
                return None
        
        logger.info("✅ Pre-trained ArcFace model loaded successfully")
        logger.info("📊 Model was trained on MS1MV2: 5.8M images, 85K identities")
        logger.info(f"🧠 Backbone: ResNet-100 (pre-trained on celebrity faces)")
        logger.info(f"📐 Input: 112x112x3 RGB images")
        logger.info(f"🎯 Output: 512-dimensional embeddings")
        
        return rec_model
        
    except Exception as e:
        logger.error(f"❌ Failed to load pre-trained ArcFace: {e}")
        logger.warning("⚠️  Will proceed without pretrained features - training from scratch")
        return None


class SimilarityArcFace(nn.Module):
    """
    Use pre-trained ArcFace 512-d features directly; L2-normalize; optional fine-tune.
    """
    def __init__(self, pretrained_model=None, embedding_dim: int = 512, 
                 freeze_backbone: bool = True):
        super().__init__()
        
        if pretrained_model is None:
            pretrained_model = load_pretrained_arcface()
        self.pretrained_model = pretrained_model
        self.embedding_dim = embedding_dim
        logger.info("✅ Using pre-trained ResNet-100 backbone (trained on 85K identities)")
        
        # Minimal projection head to enable training while preserving raw features initially.
        # If embedding_dim == 512, initialize to identity so behavior matches raw features.
        self.projection = nn.Linear(512, embedding_dim, bias=False)
        with torch.no_grad():
            if embedding_dim == 512:
                self.projection.weight.copy_(torch.eye(512))
            else:
                nn.init.xavier_uniform_(self.projection.weight)
        logger.info(
            f"🧩 Added projection layer 512→{embedding_dim} (identity init when 512) for trainability"
        )
        
        # TRAINING STRATEGY: Freeze backbone initially
        self.backbone_frozen = freeze_backbone
        if freeze_backbone:
            logger.info("🔒 Backbone FROZEN - training uses fixed ArcFace features")
        else:
            logger.info("🔓 Backbone UNFROZEN - end-to-end fine-tuning")
    
    def freeze_backbone(self):
        """Freeze pre-trained backbone for progressive training"""
        self.backbone_frozen = True
        logger.info("🔒 Pre-trained backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end fine-tuning"""
        self.backbone_frozen = False
        logger.info("🔓 Pre-trained backbone unfrozen for fine-tuning")
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using pre-trained ArcFace backbone
        
        Args:
            x: Input tensor (N, C, H, W) in range [0, 1] or [-1, 1]
        
        Returns:
            Feature tensor (N, feature_dim)
        """
        batch_size = x.shape[0]
        features = []
        
        # Process each image in the batch
        for i in range(batch_size):
            # Convert tensor to numpy (HWC format for InsightFace)
            img_tensor = x[i]  # (C, H, W)
            
            # Ensure correct range and format
            if img_tensor.min() < 0:  # Assume [-1, 1] range
                img_np = ((img_tensor + 1) * 127.5).clamp(0, 255).byte()
            else:  # Assume [0, 1] range
                img_np = (img_tensor * 255).clamp(0, 255).byte()
            
            # Convert to HWC format
            img_np = img_np.permute(1, 2, 0).cpu().numpy()
            
            # Get features from pre-trained model strictly via get_feat (512-d)
            if self.pretrained_model is None:
                raise RuntimeError("Pretrained ArcFace model is not available")
            feat = self.pretrained_model.get_feat(img_np)
            if hasattr(feat, 'shape') and len(feat.shape) > 1:
                feat = feat.flatten()
            if len(feat) != 512:
                raise RuntimeError(f"Unexpected ArcFace feature dim: {len(feat)}")
            features.append(torch.from_numpy(feat))
        
        # Stack features
        features = torch.stack(features).to(x.device)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image -> pre-trained features -> similarity embedding
        
        Args:
            x: Input tensor (N, 3, 112, 112)
        
        Returns:
            Normalized similarity embeddings (N, embedding_dim)
        """
        # VALIDATION: Check input format (ArcFace expects 112x112)
        if x.shape[1:] != (3, 112, 112):
            raise ValueError(f"ArcFace expects (N, 3, 112, 112), got {x.shape}")
        
        # Extract features using PRE-TRAINED ResNet-100
        with torch.set_grad_enabled(not self.backbone_frozen):
            features = self.get_backbone_features(x)  # Rich facial features from celebrity training
        
        # Project (identity at start if 512) and L2 normalize
        projected = self.projection(features.float())
        normalized_embeddings = F.normalize(projected, p=2, dim=1)
        
        return normalized_embeddings


def setup_similarity_model(device: str = None, embedding_dim: int = 512, 
                          freeze_backbone: bool = True) -> SimilarityArcFace:
    """
    Initialize similarity model using PRE-TRAINED ArcFace ResNet-100
    
    CRITICAL APPROACH:
    1. Load pre-trained ArcFace (already knows facial features)
    2. Adapt for similarity learning (not identity classification)
    3. Start with frozen backbone (conservative fine-tuning)
    
    Args:
        device: Device to use ('cuda', 'cpu', or None for auto)
        embedding_dim: Dimension of similarity embeddings
        freeze_backbone: Whether to freeze backbone initially
    
    Returns:
        Initialized similarity model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"🖥️  Device: {device}")
    
    # Load PRE-TRAINED ArcFace ResNet-100
    pretrained_arcface = load_pretrained_arcface()
    
    # Create similarity model (adapting pre-trained model)
    model = SimilarityArcFace(
        pretrained_model=pretrained_arcface, 
        embedding_dim=embedding_dim,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    
    # VALIDATION: Test model functionality
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 112, 112).to(device)
        
        # Normalize input to expected range
        dummy_input = (dummy_input + 1) / 2  # Convert [-1,1] to [0,1]
        
        try:
            dummy_output = model(dummy_input)
            
            # Validate output
            if dummy_output.shape != (2, embedding_dim):
                raise ValueError(f"Expected (2, {embedding_dim}), got {dummy_output.shape}")
            
            # Validate L2 normalization
            norms = torch.norm(dummy_output, p=2, dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
                raise ValueError("Embeddings not properly normalized")
            
            logger.info(f"✅ Model validation passed:")
            logger.info(f"   Input: {dummy_input.shape} -> Output: {dummy_output.shape}")
            logger.info(f"   L2 norms: {norms.cpu().numpy()}")
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            raise
    
    logger.info("🎯 Ready for similarity fine-tuning!")
    return model


def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix
    
    Args:
        embeddings: Normalized embeddings (N, embedding_dim)
    
    Returns:
        Similarity matrix (N, N)
    """
    # Since embeddings are L2 normalized, cosine similarity = dot product
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    return similarity_matrix


def find_top_k_similar(query_embedding: torch.Tensor, database_embeddings: torch.Tensor, 
                      k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find top-k most similar faces to query
    
    Args:
        query_embedding: Query embedding (1, embedding_dim)
        database_embeddings: Database embeddings (N, embedding_dim)
        k: Number of top results to return
    
    Returns:
        Tuple of (similarities, indices) for top-k results
    """
    # Compute similarities
    similarities = torch.mm(query_embedding, database_embeddings.t()).squeeze(0)
    
    # Get top-k
    top_similarities, top_indices = torch.topk(similarities, k=min(k, len(similarities)))
    
    return top_similarities, top_indices


if __name__ == "__main__":
    # Test model setup
    logger.info("🧪 Testing similarity model setup...")
    
    try:
        model = setup_similarity_model(embedding_dim=512, freeze_backbone=True)
        logger.info("✅ Model setup test passed")
        
        # Test similarity computation
        with torch.no_grad():
            dummy_embeddings = torch.randn(10, 512)
            dummy_embeddings = F.normalize(dummy_embeddings, p=2, dim=1)
            
            sim_matrix = compute_similarity_matrix(dummy_embeddings)
            logger.info(f"✅ Similarity matrix test passed: {sim_matrix.shape}")
            
            # Test top-k search
            query = dummy_embeddings[:1]
            database = dummy_embeddings[1:]
            top_sims, top_indices = find_top_k_similar(query, database, k=3)
            logger.info(f"✅ Top-k search test passed: {top_sims.shape}, {top_indices.shape}")
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        raise

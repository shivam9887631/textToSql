import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

from config import DEFAULT_EMBEDDING_MODEL

class LocalEmbedder:
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model locally"""
        self.model = SentenceTransformer(self.model_name)
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings using local sentence transformer model"""
        if not isinstance(texts, list):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "paraphrase-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            return self.model.encode(texts)
        
        return self.model.encode(texts)

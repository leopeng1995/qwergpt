import faiss
import numpy as np
import pandas as pd
from typing import List, Dict

from qwergpt.embedders import Embedder
from .base import VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, embedder: Embedder, dimension: int = 768):
        """Initialize FaissVectorStore
        
        Args:
            embedder: Embedder instance to generate vectors
            dimension: Vector dimension, defaults to 768
        """
        super().__init__(embedder, dimension)
        self.embedder = embedder
        self.dimension = dimension
        self.df = None
        self.index = None

    def add_texts(self, texts: List[str], metadata: List[Dict] = None) -> Dict:
        vectors = self.embedder.embed(texts)
        vector_dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        
    def search(self, query: str, limit: int = 3) -> List[Dict]:
        # Create a search vector
        search_vector = self.embedder.embed(query)
        _vector = np.array([search_vector])
        faiss.normalize_L2(_vector)

        # Search
        k = self.index.ntotal
        distances, ann = self.index.search(_vector, k=k)

        # Sort search results
        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
            
        return results
        
    def delete_collection(self):
        self.index = None
        self.df = None

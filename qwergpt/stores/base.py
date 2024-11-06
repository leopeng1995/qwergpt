from typing import List, Dict
from abc import ABC, abstractmethod

from qwergpt.embedders import Embedder


class VectorStore(ABC):
    """Abstract base class for vector stores"""

    def __init__(self, embedder: Embedder, dimension: int = 768):
        """Initialize VectorStore
        
        Args:
            embedder: Embedder instance to generate vectors
            dimension: Vector dimension, defaults to 768
        """
        self.embedder = embedder
        self.dimension = dimension
    
    @abstractmethod
    def add_texts(self, texts: List[str], metadata: List[Dict] = None) -> Dict:
        """Add texts to vector store
        
        Args:
            texts: List of text strings
            metadata: List of metadata dictionaries for each text
            
        Returns:
            Insert result
        """
        pass
    
    @abstractmethod 
    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """Search similar vectors
        
        Args:
            query: Query text
            limit: Number of results to return
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def delete_collection(self):
        """Delete the collection"""
        pass

from typing import List, Dict
from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
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

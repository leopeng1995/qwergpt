from typing import List, Dict

from pymilvus import model
from pymilvus import MilvusClient

from .base import VectorStore


class MilvusVectorStore(VectorStore):
    def __init__(self, db_path: str, collection_name: str, dimension: int = 768):
        """Initialize MilvusVectorStore
        
        Args:
            db_path: Milvus database path
            collection_name: Name of the collection
            dimension: Vector dimension, defaults to 768
        """
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.dimension = dimension 
        self.embedding_fn = model.DefaultEmbeddingFunction()
        
        # Initialize collection
        self._init_collection()
        
    def _init_collection(self):
        """Initialize collection if not exists"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension
        )
        
    def add_texts(self, texts: List[str], metadata: List[Dict] = None) -> Dict:
        """Add texts to vector store
        
        Args:
            texts: List of text strings
            metadata: List of metadata dictionaries for each text
            
        Returns:
            Insert result from Milvus
        """
        vectors = self.embedding_fn.encode_documents(texts)
        
        if metadata is None:
            metadata = [{"subject": "default"} for _ in texts]
            
        data = [
            {
                "id": i,
                "vector": vectors[i], 
                "text": texts[i],
                **metadata[i]
            }
            for i in range(len(texts))
        ]
        
        return self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
    
    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """Search similar vectors
        
        Args:
            query: Query text
            limit: Number of results to return
            
        Returns:
            List of search results
        """
        query_vector = self.embedding_fn.encode_documents([query])[0]
        
        return self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit
        )
    
    def delete_collection(self):
        """Delete the collection"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

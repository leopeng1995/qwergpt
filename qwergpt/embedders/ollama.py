from typing import Dict, List

import requests
import numpy as np

from .base import Embedder


class OllamaEmbedder(Embedder):
    """Embedder implementation using Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", 
                 model: str = "quentinz/bge-large-zh-v1.5"):
        """Initialize Ollama embedder
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use for embeddings
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        
    def embed(self, text: str | List[str]) -> np.ndarray:
        """Get embedding for text using Ollama API
        
        Args:
            text: Input text or list of texts to embed
            
        Returns:
            numpy array of embedding vectors. If input is a single string,
            returns a 1D array. If input is a list, returns a 2D array.
            
        Raises:
            RuntimeError: If API call fails
        """
        # 处理单个字符串输入
        if isinstance(text, str):
            return self._get_single_embedding(text)
        
        # 处理字符串列表输入
        embeddings = []
        for t in text:
            embedding = self._get_single_embedding(t)
            embeddings.append(embedding)
        return np.stack(embeddings)

    def _get_single_embedding(self, text: str) -> np.ndarray:
        """Helper method to get embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embedding vector
        """
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API call failed: {response.text}")
            
        result: Dict = response.json()
        
        if "embedding" not in result:
            raise RuntimeError(f"No embedding in response: {result}")
            
        return np.array(result["embedding"])

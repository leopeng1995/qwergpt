import requests
import numpy as np

from .base import Embedder


class ZhipuEmbedder(Embedder):
    def __init__(self, api_key: str, model: str = "embedding-3", dimensions: int = 2048):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.dimensions = dimensions

    def embed(self, text: str | list[str]) -> np.ndarray:
        payload = {
            "model": self.model,
            "input": text if isinstance(text, list) else [text],
            "dimensions": self.dimensions
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            if "data" in result and result["data"]:
                embeddings = [item["embedding"] for item in result["data"]]
                embedding_array = np.array(embeddings, dtype=np.float32)
                return embedding_array[0] if not isinstance(text, list) else embedding_array
            else:
                raise ValueError("API response does not contain embedding data")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

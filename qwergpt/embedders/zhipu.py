import requests
import numpy as np

from .base import Embedder


class ZhipuEmbedder(Embedder):
    def __init__(self, api_key: str, model: str = "embedding-3"):
        """
        初始化智谱Embedder
        
        Args:
            api_key: 智谱API的认证token
            model: 使用的模型名称,默认为embedding-3
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def embed(self, text: str) -> np.ndarray:
        """
        将文本转换为向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 文本的向量表示
        """
        payload = {
            "model": self.model,
            "input": text,
            "dimensions": 2048
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                # 转换为numpy数组并归一化
                embedding_array = np.array(embedding)
                normalized_embedding = embedding_array / np.linalg.norm(embedding_array)
                return normalized_embedding
            else:
                raise ValueError("API response does not contain embedding data")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

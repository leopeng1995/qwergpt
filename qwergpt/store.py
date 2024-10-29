from abc import ABC
from typing import List

import numpy as np
from fastembed import TextEmbedding


class DocumentStore(ABC):

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.embedding_model = TextEmbedding(
            model_name=model_name,
            cache_dir='models'
        )

    def insert(self, doc: str) -> None:
        self.documents.append(doc)
        embedding = list(self.embedding_model.passage_embed([doc]))[0]
        self.embeddings.append(embedding)

    def batch_insert(self, docs: List[str]) -> None:
        embeddings = self.embedding_model.passage_embed(docs)
        self.documents.extend(docs)
        self.embeddings.extend(embeddings)

    def query(self, doc: str, k: int = 5) -> List[str]:
        query_embedding = list(self.embedding_model.query_embed(doc))[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        return [self.documents[i] for i in top_k_indices]

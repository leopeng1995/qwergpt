from abc import ABC, abstractmethod
from typing import List, Tuple

from qwergpt.document import Document


class Retriever(ABC):
    @abstractmethod
    def create_chunks(self, doc: Document, chunk_size: int) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, float, str]]:
        pass

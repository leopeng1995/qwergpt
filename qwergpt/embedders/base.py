from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass

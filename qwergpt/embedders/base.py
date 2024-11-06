from typing import List
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str | List[str]) -> np.ndarray:
        pass

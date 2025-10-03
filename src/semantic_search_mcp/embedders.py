from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        ...


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", normalize: bool = True):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return embs.astype(np.float32)

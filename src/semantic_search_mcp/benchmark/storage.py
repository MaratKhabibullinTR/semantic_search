from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
import shutil
import os

from semantic_search_mcp.utils import is_dir_empty


@dataclass
class StorageBackend(ABC):
    @abstractmethod
    def save_index(self, src_dir: Path, combo_id: str): ...

    @abstractmethod
    def load_index_dir(self, combo_id: str) -> Path: ...


@dataclass
class LocalFaissStorage(StorageBackend):
    output_dir: Path

    def save_index(self, src_dir: Path, combo_id: str):
        target = self.output_dir / combo_id
        target.mkdir(parents=True, exist_ok=True)
        for name in ("vectors.faiss", "chunks.jsonl"):
            p = src_dir / name
            if p.exists():
                shutil.move(p, target / name)
        if is_dir_empty(src_dir):
            os.rmdir(src_dir)

    def load_index_dir(self, combo_id: str) -> Path:
        p = self.output_dir / combo_id
        if not p.exists():
            raise FileNotFoundError(f"Index not found: {p}")
        return p

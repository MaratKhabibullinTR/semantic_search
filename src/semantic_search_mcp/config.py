from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Config(BaseSettings):
    # Embeddings
    emb_model: str = Field(
        default="all-MiniLM-L12-v2",
        description="Sentence-Transformers model",
    )
    emb_batch_size: int = 64
    emb_normalize: bool = True

    # Chunking
    chunk_tokens: int = 500
    chunk_overlap: int = 80

    # Paths
    input_dir: Path = Path("data/corpus")
    index_dir: Path = Path("data/index")

    # Hybrid weighting
    alpha_dense: float = 0.6  # 0..1; higher -> prioritize dense similarity

    class Config:
        env_file = ".env"

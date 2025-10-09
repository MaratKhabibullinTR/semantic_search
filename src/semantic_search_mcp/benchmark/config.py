from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path

@dataclass
class CorpusSpec:
    type: str            # 'local' | 's3' | 'url_list'
    path: Optional[str] = None
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    urls_file: Optional[str] = None

@dataclass
class ChunkerSpec:
    id: str
    type: str            # 'spacy' | 'recursive'
    chunk_size: int = 1000
    chunk_overlap: int = 100
    spacy_model: Optional[str] = None

@dataclass
class EmbeddingSpec:
    id: str
    type: str            # 'sentence_transformers' | 'openai' | 'bedrock'
    model_name: str
    api_key_env: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = None

@dataclass
class StorageSpec:
    backend: str         # 'faiss_local' | 'faiss_s3'
    output_dir: str
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None

@dataclass
class IndexingSpec:
    combinations: List[Dict[str, str]]
    parallelism: int = 1
    batch_size: int = 256

@dataclass
class QuerySpec:
    top_k: int = 5

@dataclass
class ReportSpec:
    top_k: int = 10

@dataclass
class AppConfig:
    corpora: List[CorpusSpec]
    chunkers: List[ChunkerSpec]
    embeddings: List[EmbeddingSpec]
    storage: StorageSpec
    indexing: IndexingSpec
    query: QuerySpec
    report: ReportSpec

def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    corpora = [CorpusSpec(**c) for c in data.get("corpora", [])]
    chunkers = [ChunkerSpec(**c) for c in data.get("chunkers", [])]
    embeddings = [EmbeddingSpec(**e) for e in data.get("embeddings", [])]
    storage = StorageSpec(**data["storage"])
    indexing = IndexingSpec(**data["indexing"])
    query = QuerySpec(**data.get("query", {}))
    report = ReportSpec(**data.get("report", {}))

    return AppConfig(
        corpora=corpora,
        chunkers=chunkers,
        embeddings=embeddings,
        storage=storage,
        indexing=indexing,
        query=query,
        report=report,
    )

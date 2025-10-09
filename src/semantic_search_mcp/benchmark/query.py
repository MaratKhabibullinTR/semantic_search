from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple

import faiss

from semantic_search_mcp.search import hybrid_search
from .config import AppConfig
from semantic_search_mcp.benchmark.storage import LocalFaissStorage
from semantic_search_mcp.utils import combo_id


def _storage(cfg: AppConfig):
    if cfg.storage.backend == "faiss_local":
        return LocalFaissStorage(Path(cfg.storage.output_dir))
    else:
        raise ValueError(f"Unknown storage backend: {cfg.storage.backend}")


def _combo_ids(cfg: AppConfig) -> List[str]:
    return [f"{x['chunker']}__{x['embedding']}" for x in cfg.indexing.combinations]


def run_query(
    cfg: AppConfig, query: str, top_k: int | None = None
) -> Dict[str, List[Tuple[str, float, Dict]]]:
    st = _storage(cfg)
    ids = _combo_ids(cfg)

    results = {}
    for cid in ids:
        idx_dir = st.load_index_dir(cid)

        model_id = [
            c["embedding"]
            for c in cfg.indexing.combinations
            if combo_id(c["chunker"], c["embedding"]) == cid
        ][0]
        model_name = [e.model_name for e in cfg.embeddings if e.id == model_id][0]
        
        k = top_k if top_k is not None else cfg.query.top_k

        search_result = hybrid_search(
            index_dir=idx_dir,
            query=query,
            model_name=model_name,
            k=k,
        )

        triples = []
        for r in search_result:
            triples.append((r["chunk"], r["score"], {
                "source": r["source"],
                "doc_id": r["doc_id"],
                "chunk_idx": r["chunk_idx"],
                "chunker_id": r["chunker_id"],
                "embedding_id": r["embedding_id"],
            }))
        results[cid] = triples
    return results

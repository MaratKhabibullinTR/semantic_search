from pathlib import Path
from typing import List, Dict, Tuple

from semantic_search_mcp.search import hybrid_search
from .config import AppConfig
from semantic_search_mcp.benchmark.storage import get_storage
from semantic_search_mcp.utils import combo_id


def _get_combo_ids(cfg: AppConfig) -> List[str]:
    return [combo_id(x["chunker"], x["embedding"]) for x in cfg.indexing.combinations]


def _get_model_name(combindation_id: str, cfg: AppConfig) -> str:
    embedding_id = [
        c["embedding"]
        for c in cfg.indexing.combinations
        if combo_id(c["chunker"], c["embedding"]) == combindation_id
    ][0]
    return [e.model_name for e in cfg.embeddings if e.id == embedding_id][0]


def run_query(
    cfg: AppConfig, query: str, top_k: int | None = None
) -> Dict[str, List[Tuple[str, float, Dict]]]:
    st = get_storage(cfg)
    combo_ids = _get_combo_ids(cfg)

    results = {}
    for cid in combo_ids:
        idx_dir = st.load_index_dir(cid)

        model_name = _get_model_name(combindation_id=cid, cfg=cfg)

        k = top_k if top_k is not None else cfg.query.top_k

        search_result = hybrid_search(
            index_dir=idx_dir,
            query=query,
            model_name=model_name,
            k=k,
        )

        triples = []
        for r in search_result:
            triples.append(
                (
                    r["chunk"],
                    r["score"],
                    {
                        "source": r["source"],
                        "doc_id": r["doc_id"],
                        "chunk_idx": r["chunk_idx"],
                        "chunker_id": r["chunker_id"],
                        "embedding_id": r["embedding_id"],
                    },
                )
            )
        results[cid] = triples
    return results

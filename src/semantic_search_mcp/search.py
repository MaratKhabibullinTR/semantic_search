from pathlib import Path
from typing import List, Dict, Tuple
import json
import numpy as np
import faiss
import pickle
import regex as re

from .embedders import LocalEmbedder

_WORD_RE = re.compile(r"\w+", re.UNICODE)

def _simple_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _load_chunks(chunks_path: Path) -> List[Dict]:
    out = []
    with chunks_path.open("r", encoding="utf-8") as fr:
        for line in fr:
            out.append(json.loads(line))
    return out


def _mmr(
    doc_vectors: np.ndarray,
    idxs: List[int],
    query_vec: np.ndarray,
    k: int,
    lambda_mult: float = 0.7,
) -> List[int]:
    """Maximal Marginal Relevance to diversify top results."""
    selected: List[int] = []
    candidates: List[int] = list(idxs)

    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    doc_norms = np.linalg.norm(doc_vectors[candidates], axis=1, keepdims=True) + 1e-12
    doc_unit = doc_vectors[candidates] / doc_norms
    sim_to_query = (doc_unit @ q.reshape(-1, 1)).ravel()

    while candidates and len(selected) < k:
        if not selected:
            best_idx = int(np.argmax(sim_to_query))
            selected.append(candidates.pop(best_idx))
            sim_to_query = np.delete(sim_to_query, best_idx)
            doc_unit = np.delete(doc_unit, best_idx, axis=0)
            continue
        # Compute max similarity to already selected
        selected_vecs = doc_vectors[selected]
        sel_norms = np.linalg.norm(selected_vecs, axis=1, keepdims=True) + 1e-12
        sel_unit = selected_vecs / sel_norms
        sim_to_selected = doc_unit @ sel_unit.T  # (n_cand, n_sel)
        max_sim = sim_to_selected.max(axis=1)

        mmr_scores = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim
        pick = int(np.argmax(mmr_scores))
        selected.append(candidates.pop(pick))
        sim_to_query = np.delete(mmr_scores, pick)
        doc_unit = np.delete(doc_unit, pick, axis=0)

    return selected


def hybrid_search(
    index_dir: Path,
    query: str,
    model_name: str = "all-MiniLM-L12-v2",
    k: int = 10,
    alpha_dense: float = 0.6,
    use_mmr: bool = True,
) -> List[Dict]:
    # Load FAISS + metadata
    faiss_path = index_dir / "vectors.faiss"
    chunks_path = index_dir / "chunks.jsonl"

    index = faiss.read_index(str(faiss_path))
    records = _load_chunks(chunks_path)

    # Dense retrieval
    embedder = LocalEmbedder(model_name, normalize=True)
    qv = embedder.encode([query])[0]
    D, I = index.search(qv.reshape(1, -1), min(100, len(records)))
    dense_scores = D[0]  # already cosine if normalized
    dense_idxs = I[0]

    # Score fusion
    ds = dense_scores
    scores = ds        

    # Rank
    order = np.argsort(-scores)
    ranked = dense_idxs[order].tolist()

    # Optional MMR diversification on top 50
    top_pool = ranked[: min(50, len(ranked))]
    final_idxs = _mmr(index.reconstruct_n(0, index.ntotal), top_pool, qv, k) if use_mmr else top_pool[:k]

    # Build results
    out: List[Dict] = []
    for ridx in final_idxs[:k]:
        rec = records[ridx]
        out.append(
            {
                "score": float(scores[order.tolist().index(dense_idxs.tolist().index(ridx))]) if len(order) else 0.0,
                "source": rec["source"] if "source" in rec else None,
                "chunk": (rec["chunk"][:600] + ("…" if len(rec["chunk"]) > 600 else "") ) if "chunk" in rec else None,
                "chunk_idx": rec["chunk_idx"] if "chunk_idx" in rec else None,

                "doc_id": rec["doc_id"] if "doc_id" in rec else None,
                "chunker_id": rec["chunker_id"] if "chunker_id" in rec else None,
                "embedding_id": rec["embedding_id"] if "embedding_id" in rec else None,
            }
        )
    return out
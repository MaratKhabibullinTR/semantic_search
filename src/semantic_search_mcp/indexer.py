from pathlib import Path
from typing import List, Dict, Tuple
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from .embedders import LocalEmbedder
from .chunkers.custom_chunker import chunk_text


def _read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def discover_files(input_dir: Path, exts=(".txt", ".md")) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(input_dir.rglob(f"*{ext}")))
    return files


def build_index(
    input_dir: Path,
    out_dir: Path,
    emb_model: str = "all-MiniLM-L12-v2",
    chunk_tokens: int = 500,
    chunk_overlap: int = 80,
    normalize: bool = True,
    enable_bm25: bool = True,
) -> Tuple[Path, Path, Path, Path]:
    """Builds FAISS index (+ optional BM25) from scratch. No change detection."""
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No input files found under {input_dir}")

    # 1) Chunk all docs
    records: List[Dict] = []
    for f in files:
        text = _read_text_file(f)
        chunks = chunk_text(text, tokens_per_chunk=chunk_tokens, overlap=chunk_overlap)
        for i, ch in enumerate(chunks):
            records.append(
                {
                    "id": len(records),
                    "source": str(f),
                    "chunk_idx": i,
                    "chunk": ch["text"],
                }
            )

    # 2) Embed
    embedder = LocalEmbedder(model_name=emb_model, normalize=normalize)
    texts = [r["chunk"] for r in records]
    mat = embedder.encode(texts)  # (N, D) float32

    # 3) FAISS (cosine via inner product on normalized vectors)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    # 4) Persist FAISS + ids
    faiss_path = out_dir / "vectors.faiss"
    ids_path = out_dir / "ids.npy"
    faiss.write_index(index, str(faiss_path))
    np.save(ids_path, np.arange(mat.shape[0], dtype=np.int64))

    # 5) Persist chunks metadata as JSONL
    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fw:
        for r in records:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 6) Optional BM25
    bm25_path = out_dir / "bm25.pkl"
    if enable_bm25:
        from semantic_search_mcp.search import _simple_tokens
        from sklearn.feature_extraction.text import strip_accents_ascii
        import pickle

        corpus_tokens: List[List[str]] = []
        for r in records:
            t = strip_accents_ascii(r["chunk"]) if r["chunk"] else ""
            corpus_tokens.append(_simple_tokens(t))
        bm25 = BM25Okapi(corpus_tokens)
        with bm25_path.open("wb") as f:
            pickle.dump(bm25, f)

    return faiss_path, ids_path, chunks_path, (bm25_path if enable_bm25 else Path(""))

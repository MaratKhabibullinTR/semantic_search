from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json
import faiss
import uuid

from semantic_search_mcp.benchmark.storage import LocalFaissStorage
from semantic_search_mcp.utils import ensure_dir

from .config import AppConfig

from semantic_search_mcp.chunkers import make_chunker, chunk
from semantic_search_mcp.dataset.local_dataset import load_local, load_s3, Document
from semantic_search_mcp.embedders import make_embeder


def _load_corpora(cfg: AppConfig) -> Iterable[Document]:
    for c in cfg.corpora:
        if c.type == "local":
            yield from load_local(c.path)
        elif c.type == "s3":
            assert c.bucket and c.prefix, "s3 corpus requires bucket+prefix"
            yield from load_s3(c.bucket, c.prefix)
        else:
            raise ValueError(f"Unknown corpus type: {c.type}")

def build_single_combo(args: Tuple[dict, dict, str, int, List[Document]]):
    chunker_spec, embed_spec, out_dir, batch_size, docs = args

    splitter = make_chunker(chunker_spec)
    embedder = make_embeder(embed_spec)

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for d in docs:
        pieces = chunk(d.text, splitter)
        for i, ch in enumerate(pieces):
            texts.append(ch)
            metas.append({
                "source": d.metadata.get("source", d.doc_id),
                "doc_id": d.doc_id,
                "chunk_idx": i,
                "chunk": ch,
                "chunker_id": chunker_spec["id"],
                "embedding_id": embed_spec["id"],
            })

    # Build FAISS incrementally to keep memory reasonable
    # Start with first batch, then add in chunks
    assert texts, "No text to index"

    # Save to a temp dir; the storage backend will copy/publish it
    tmp_folder_name = str(uuid.uuid4())
    tmp = Path(out_dir) / tmp_folder_name
    tmp.mkdir(parents=True, exist_ok=True)

    # 1) Embed
    mat = embedder.encode(texts)  # (N, D) float32

    # 2) FAISS (cosine via inner product on normalized vectors)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    # 3) Persist FAISS
    faiss_path = tmp / "vectors.faiss"
    faiss.write_index(index, str(faiss_path))

    # 4) Persist chunks metadata as JSONL
    chunks_path = tmp / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fw:
        for r in metas:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

    return tmp


def index_all(cfg: AppConfig):
    # init storage
    out_dir = ensure_dir(cfg.storage.output_dir)
    if cfg.storage.backend == "faiss_local":
        storage = LocalFaissStorage(out_dir)
    else:
        raise NotImplementedError(f"No storage for: {cfg.storage.backend}")

    # materialize corpus into memory
    docs = list(_load_corpora(cfg))
    if not docs:
        raise RuntimeError("No documents found in the corpus.")

    # map id -> spec for quick lookup
    chunkers = {c.id: c.__dict__ for c in cfg.chunkers}
    embeds = {e.id: e.__dict__ for e in cfg.embeddings}

    combo_specs = []
    for comb in cfg.indexing.combinations:
        ch_id, em_id = comb["chunker"], comb["embedding"]
        if ch_id not in chunkers:
            raise KeyError(f"Chunker id not found: {ch_id}")
        if em_id not in embeds:
            raise KeyError(f"Embedding id not found: {em_id}")
        combo_specs.append( (chunkers[ch_id], embeds[em_id]) )

    jobs = []
    with ProcessPoolExecutor(max_workers=cfg.indexing.parallelism) as ex:
        for ch_spec, em_spec in combo_specs:
            cid = f"{ch_spec['id']}__{em_spec['id']}"
            jobs.append( (cid, ex.submit(build_single_combo, (ch_spec, em_spec, str(out_dir), cfg.indexing.batch_size, docs))) )
        for cid, fut in tqdm(jobs, desc="Building combos"):
            tmp_dir = fut.result()
            storage.save_index(tmp_dir, cid)
    print("Completed indexing for all combos.")

import json
from pathlib import Path
from typing import List, Optional
import typer

from semantic_search_mcp.benchmark.config import load_config as load_bench_config
from semantic_search_mcp.benchmark.index import index_all as index_all_bench
from semantic_search_mcp.benchmark.query import run_query
from semantic_search_mcp.config import Config
from semantic_search_mcp.indexer import build_index
from semantic_search_mcp.search import hybrid_search
from semantic_search_mcp.dataset.local_dataset import convert_json_to_txt
from semantic_search_mcp.dataset.s3_dataset import download_artifacts_from_s3 as download_artifacts

from semantic_search_mcp.logging_utils import setup_logging
from semantic_search_mcp.utils import ensure_dir
from semantic_search_mcp.benchmark.report import quality_with_qrels, quality_unsupervised

app = typer.Typer(add_completion=False, no_args_is_help=True)

setup_logging(level="INFO")


@app.command()
def reindex(
    input: Path = typer.Option(Path("data/corpus"), help="Folder with .txt/.md"),
    out: Path = typer.Option(Path("data/index"), help="Index output folder"),
    model: str = typer.Option("all-MiniLM-L12-v2", help="SentenceTransformer model"),
    chunk_tokens: int = typer.Option(500),
    chunk_overlap: int = typer.Option(80),
    no_bm25: bool = typer.Option(False, help="Disable BM25"),
):
    cfg = Config()
    faiss_path, ids_path, chunks_path, bm25_path = build_index(
        input_dir=input,
        out_dir=out,
        emb_model=model,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
        normalize=True,
        enable_bm25=not no_bm25,
    )
    typer.echo(f"FAISS: {faiss_path}\nIDS: {ids_path}\nCHUNKS: {chunks_path}\nBM25: {bm25_path if not no_bm25 else 'disabled'}")


@app.command()
def search(
    index: Path = typer.Option(Path("data/index")),
    query: str = typer.Option(..., prompt=True),
    k: int = typer.Option(8),
    model: str = typer.Option("all-MiniLM-L12-v2"),
    alpha_dense: float = typer.Option(0.6),
    mmr: bool = typer.Option(True),
):
    res = hybrid_search(
        index_dir=index,
        query=query,
        model_name=model,
        k=k,
        alpha_dense=alpha_dense,
        use_mmr=mmr,
    )
    typer.echo(json.dumps(res, ensure_ascii=False, indent=2))


@app.command()
def convert_json_corpus_to_txt(
    input: Path = typer.Option(Path("data/corpus/raw_json"), help="Folder with .json"),
    out: Path = typer.Option(Path("data/corpus"), help="Output folder")):
    convert_json_to_txt(json_dir=input, txt_dir=out)
    

@app.command()
def index_all(config: str = typer.Option(..., help="Path to config YAML")):
    cfg = load_bench_config(config)
    index_all_bench(cfg)

@app.command()
def query(
    config: str = typer.Option(..., help="Path to config YAML"),
    query: str = typer.Option(..., help="User query string"),
    top_k: Optional[int] = typer.Option(None, help="Override top_k")
):
    cfg = load_bench_config(config)
    results = run_query(cfg, query, top_k=top_k)
    for cid, triples in results.items():
        print(f"\n=== {cid} ===")
        for i, (text, score, meta) in enumerate(triples, start=1):
            src = meta.get("source", "unknown")
            did = meta.get("doc_id", "unknown")
            print(f"[{i}] score={score:.4f} | source={src} | doc_id={did}")
            snippet = (text[:300] + "...") if len(text) > 300 else text
            print(snippet.replace("\n"," "))


@app.command()
def report(
    config: str = typer.Option(..., help="Path to config YAML"),
    out_dir: str = typer.Option("./reports", help="Where to write CSVs + PNGs"),
    qrels: Optional[str] = typer.Option(None, help="JSONL with {'query', 'relevant_doc_ids': [...]}"),
    queries: Optional[List[str]] = typer.Option(None, help="If no qrels, provide queries for unsupervised report"),
    top_k: Optional[int] = typer.Option(None, help="Override top_k in config"),
):
    cfg = load_bench_config(config)
    out = ensure_dir(out_dir)
    k = top_k if top_k is not None else cfg.report.top_k
    if qrels:
        agg = quality_with_qrels(cfg, Path(qrels), out, k)
        print("\nSupervised metrics written to:", out)
        print(agg.to_string(index=False))
    else:
        if not queries:
            raise typer.BadParameter("When no --qrels is provided, you must pass --queries.")
        agg = quality_unsupervised(cfg, list(queries), out, k)
        print("\nUnsupervised metrics written to:", out)
        print(agg.to_string(index=False))


@app.command()
def download_artifacts_from_s3():
    download_artifacts()



if __name__ == "__main__":
    app()

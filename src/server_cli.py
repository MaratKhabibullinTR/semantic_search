import json
from pathlib import Path
import typer

from semantic_search_mcp.config import Config
from semantic_search_mcp.indexer import build_index
from semantic_search_mcp.search import hybrid_search
from semantic_search_mcp.dataset import convert_json_dataset_to_txt

from semantic_search_mcp.logging_utils import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)

setup_logging(level="INFO")


@app.command()
def reindex(
    input: Path = typer.Option(Path("data/corpus"), help="Folder with .txt/.md"),
    out: Path = typer.Option(Path("data/index"), help="Index output folder"),
    model: str = typer.Option("paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer model"),
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
    model: str = typer.Option("paraphrase-multilingual-MiniLM-L12-v2"),
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
    convert_json_dataset_to_txt(input_dir=input, out_dir=out)

if __name__ == "__main__":
    app()

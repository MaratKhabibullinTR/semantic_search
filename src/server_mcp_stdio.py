"""A minimal MCP stdio server exposing two tools: reindex, search.
Requires the `mcp` Python package. Adjust imports if the API changes.
"""
from pathlib import Path
import json
import anyio

try:
    from mcp.server import Server, NotificationOptions
    from mcp.types import Tool, TextContent
    from mcp.server.stdio import stdio_server
    from mcp.server.models import InitializationOptions
except Exception as e:
    raise SystemExit(
        "The 'mcp' package is required. Try: 'poetry add mcp' or pin a version matching your host.\n"
        f"Original import error: {e}"
    )

from semantic_search_mcp.indexer import build_index
from semantic_search_mcp.search import hybrid_search
from semantic_search_mcp.logging_utils import setup_logging

setup_logging(level="INFO")


server = Server("rag-mcp-skeleton")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="reindex", description="Rebuild the index from a folder of .txt/.md"),
        Tool(name="search", description="Hybrid semantic search over the built index"),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "reindex":
        input_dir = Path(arguments.get("input", "data/corpus"))
        out_dir = Path(arguments.get("out", "data/index"))
        model = arguments.get("model", "paraphrase-multilingual-MiniLM-L12-v2")
        chunk_tokens = int(arguments.get("chunk_tokens", 500))
        chunk_overlap = int(arguments.get("chunk_overlap", 80))
        enable_bm25 = bool(arguments.get("enable_bm25", True))

        faiss_path, ids_path, chunks_path, bm25_path = build_index(
            input_dir=input_dir,
            out_dir=out_dir,
            emb_model=model,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            normalize=True,
            enable_bm25=enable_bm25,
        )
        payload = {
            "faiss": str(faiss_path),
            "ids": str(ids_path),
            "chunks": str(chunks_path),
            "bm25": (str(bm25_path) if enable_bm25 else None),
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "search":
        index_dir = Path(arguments.get("index", "data/index"))
        query = arguments["query"]
        k = int(arguments.get("k", 8))
        model = arguments.get("model", "paraphrase-multilingual-MiniLM-L12-v2")
        alpha_dense = float(arguments.get("alpha_dense", 0.6))
        mmr = bool(arguments.get("mmr", True))

        res = hybrid_search(index_dir, query, model_name=model, k=k, alpha_dense=alpha_dense, use_mmr=mmr)
        return [TextContent(type="text", text=json.dumps(res, ensure_ascii=False))]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_strem):
        await server.run(
            read_stream=read_stream,
            write_stream=write_strem,
            initialization_options=InitializationOptions(
                server_name="rag-mcp-skeleton",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
        )

if __name__ == "__main__":
    anyio.run(main)
    # stdio_server(server).run()

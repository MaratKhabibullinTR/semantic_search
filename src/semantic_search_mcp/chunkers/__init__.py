from __future__ import annotations
from typing import List
from dataclasses import dataclass

from langchain_text_splitters import SpacyTextSplitter, RecursiveCharacterTextSplitter

from .custom_chunker import CustomTextSplitter

@dataclass
class Chunked:
    chunks: List[str]

def make_chunker(spec: dict):
    t = spec["type"]
    match t:
        case "spacy":
            model = spec.get("spacy_model", "en_core_web_sm")
            return SpacyTextSplitter(
                pipeline=model,
                chunk_size=spec.get("chunk_size", 1000),
                chunk_overlap=spec.get("chunk_overlap", 100),
            )
        case "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=spec.get("chunk_size", 1000),
                chunk_overlap=spec.get("chunk_overlap", 100),
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        case "custom":
            tokens_per_chunk = spec.get("tokens_per_chunk", 500)
            overlap = spec.get("chunk_overlap", 0)
            return CustomTextSplitter(
                tokens_per_chunk=tokens_per_chunk,
                overlap=overlap,
            )
        case _:
            raise ValueError(f"Unknown chunker type: {t}")

def chunk(text: str, splitter) -> List[str]:
    return splitter.split_text(text)

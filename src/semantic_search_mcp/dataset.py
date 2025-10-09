from __future__ import annotations
from pathlib import Path
import logging
import json
import os
from dataclasses import dataclass
from typing import Iterator, Dict, Any
import tempfile
import boto3


logger = logging.getLogger(__name__)


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


def load_local(root: str | Path) -> Iterator[Document]:
    for p in __iter_local_files(root):
        text = p.read_text(encoding="utf-8", errors="ignore")
        yield Document(
            doc_id=str(p.resolve()), text=text, metadata={"source": str(p.resolve())}
        )


def load_s3(bucket: str, prefix: str) -> Iterator[Document]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ext = os.path.splitext(key)[1].lower()
            if ext != ".txt":
                continue
            tmp = tempfile.NamedTemporaryFile(delete=False)
            s3.download_file(bucket, key, tmp.name)
            with open(tmp.name, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            yield Document(
                doc_id=f"s3://{bucket}/{key}",
                text=text,
                metadata={"source": f"s3://{bucket}/{key}"},
            )


def __iter_local_files(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    for p in root.rglob("*"):
        if p.suffix.lower() == ".txt" and p.is_file():
            yield p


def convert_json_to_txt(json_dir: Path, txt_dir: Path):
    """
    Reads all .json files from `json_dir`, combines 'title' and 'text' attributes,
    and creates new .txt files with the combined content.
    """
    logger.info("Start converting...")

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_filepath = os.path.join(json_dir, filename)

            try:
                with open(json_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                title = data.get("title", "")
                text = data.get("text", "")

                content = f"{title}\n\n{text}"

                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_filepath = os.path.join(txt_dir, txt_filename)

                with open(txt_filepath, "w", encoding="utf-8") as f:
                    f.write(content)

                logger.info(f"Successfully processed {filename} -> {txt_filename}")

            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error processing {filename}: {e}")

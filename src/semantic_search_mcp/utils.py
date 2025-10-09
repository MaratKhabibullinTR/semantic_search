from pathlib import Path
from typing import Any
import hashlib
import json
import os


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def combo_id(chunker_id: str, embedding_id: str) -> str:
    return f"{chunker_id}__{embedding_id}"


def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def save_json(p: str | Path, obj: Any) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def env_or_raise(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Environment variable {name} not set.")
    return v

def is_dir_empty(path):
    """Checks if a directory is empty."""
    try:
        return next(os.scandir(path), None) is None
    except FileNotFoundError:
        # Handle the case where the path doesn't exist
        return False 
    except NotADirectoryError:
        # Handle the case where the path is a file
        return False
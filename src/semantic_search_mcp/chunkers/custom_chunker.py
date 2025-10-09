from typing import List, Dict, Callable, Any
import regex as re
import logging

from langchain_text_splitters.base import TextSplitter

logger = logging.getLogger(__name__)

# Tokenizer primitives
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def default_token_count(s: str) -> int:
    return len(_WORD_RE.findall(s))


# Simple paragraph and sentence splitters
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[^\s])", re.UNICODE)


def split_paragraphs(text: str) -> List[str]:
    paras = _PARA_SPLIT_RE.split(text.strip())
    return [p.strip() for p in paras if p.strip()]


def split_sentences(para: str) -> List[str]:
    parts = _SENT_SPLIT_RE.split(para.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    tokens_per_chunk: int,
    overlap: int = 0,
    token_count: Callable[[str], int] = default_token_count,
) -> List[Dict]:
    """
    Paragraph-aware, sentence-first greedy chunker with sentence-level overlap.
    Guarantees: every chunk's token_count <= tokens_per_chunk.
    Returns: [{"text": str, "token_count": int}].
    """
    if tokens_per_chunk <= 0:
        raise ValueError("tokens_per_chunk must be > 0")
    # Clamp overlap to [0, tokens_per_chunk-1]
    overlap = max(0, min(overlap, max(tokens_per_chunk - 1, 0)))
    chunks: List[Dict] = []
    buf: List[str] = []  # buffered sentences (possibly across paragraphs)
    buf_tokens = 0

    def flush():
        nonlocal buf, buf_tokens
        if not buf:
            return
        out_text = " ".join(buf)  # sentences preserve punctuation; join with a space
        chunks.append({"text": out_text, "token_count": token_count(out_text)})
        # Build sentence-level overlap suffix for the next chunk
        if overlap > 0:
            keep: List[str] = []
            t = 0
            # take as many trailing sentences as needed to reach >= overlap tokens
            for s in reversed(buf):
                sc = token_count(s)
                if (
                    t + sc > tokens_per_chunk
                ):  # safety: never seed more than a full window
                    break
                keep.append(s)
                t += sc
                if t >= overlap:
                    break
            keep.reverse()
            buf = keep
            buf_tokens = sum(token_count(s) for s in buf)
        else:
            buf = []
            buf_tokens = 0

    def emit_sentence_or_split(s: str):
        """
        Add a sentence to the buffer, splitting to word windows if sentence alone
        exceeds tokens_per_chunk.
        """
        nonlocal buf, buf_tokens
        sc = token_count(s)
        if sc <= tokens_per_chunk:
            # Normal path: place sentence, flushing if needed
            if buf_tokens + sc > tokens_per_chunk:
                flush()
            buf.append(s)
            buf_tokens += sc
            return
        
        # Fallback: sentence itself is too big -> split by words with stride
        words = _WORD_RE.findall(s)
        if not words:
            # weird case (no "words"): just hard-cut characters
            hard = [
                s[i : i + tokens_per_chunk] for i in range(0, len(s), tokens_per_chunk)
            ]
            for piece in hard:
                if buf_tokens > 0:
                    flush()
                chunks.append({"text": piece, "token_count": token_count(piece)})
            return
        stride = max(1, tokens_per_chunk - overlap)  # maintain configured overlap
        i = 0
        while i < len(words):
            window_words = words[i : i + tokens_per_chunk]
            piece_text = " ".join(window_words)
            piece_tokens = token_count(piece_text)
            if buf_tokens > 0:
                flush()  # ensure "oversized sentence windows" are their own chunks
            chunks.append({"text": piece_text, "token_count": piece_tokens})
            # move by stride to create overlap between windows of a long sentence
            i += stride
        # After splitting a long sentence, the next normal sentence can go to buffer

    # Walk paragraphs -> sentences, greedily pack into chunks
    for para in split_paragraphs(text):
        sentences = split_sentences(para)
        if not sentences:
            continue
        for s in sentences:
            emit_sentence_or_split(s)
    # flush trailing buffer
    flush()
    return chunks


class CustomTextSplitter(TextSplitter):
    def __init__(
        self,
        tokens_per_chunk: int,
        overlap: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tokens_per_chunk = tokens_per_chunk
        self._overlap = overlap

    def split_text(self, text: str) -> list[str]:
        chunks = chunk_text(
            text=text,
            tokens_per_chunk=self._tokens_per_chunk,
            overlap=self._overlap)
        return [ch["text"] for ch in chunks]

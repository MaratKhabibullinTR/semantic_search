import re

import pytest

from semantic_search_mcp.chunker import chunk_text


_WORD_RE = re.compile(r"\w+", re.UNICODE)

def toks(s: str):
    return _WORD_RE.findall(s.lower())

def common_prefix_suffix_len(a, b):
    """Max length k such that a[-k:] == b[:k]."""
    k = min(len(a), len(b))
    for m in range(k, -1, -1):
        if a[-m:] == b[:m]:
            return m
    return 0

def test_empty_input_returns_empty_list():
    chunks = chunk_text("", tokens_per_chunk=50, overlap=10)
    assert chunks == []

def test_structure_and_limits_are_respected():
    text = (
        "Para A line1.\nline2.\n\n"
        "Para B line1.\nline2.\n\n"
        "Para C line1."

    )
    chunks = chunk_text(text, tokens_per_chunk=6, overlap=0)

    assert len(chunks) > 0

    for c in chunks:
        assert isinstance(c, dict)
        assert "text" in c and "token_count" in c
        assert isinstance(c["text"], str)
        assert isinstance(c["token_count"], int)
        assert c["token_count"] <= 6

def test_overlap_creates_shared_prefix_suffix_tokens():
    text = (
        "One two three four five six seven eight nine ten eleven twelve thirteen fourteen."
    )
    tokens_per_chunk = 6
    overlap = 2

    chunks = chunk_text(text, tokens_per_chunk=tokens_per_chunk, overlap=overlap)

    # Need at least 2 chunks to test overlap

    assert len(chunks) >= 2

    for i in range(len(chunks) - 1):
        prev_tokens = toks(chunks[i]["text"])
        next_tokens = toks(chunks[i + 1]["text"])

        # No chunk should exceed the limit
        assert len(prev_tokens) == chunks[i]["token_count"] <= tokens_per_chunk
        assert len(next_tokens) == chunks[i + 1]["token_count"] <= tokens_per_chunk

        # The overlap should be at least 'min(overlap, len(prev_tokens))' tokens
        k = common_prefix_suffix_len(prev_tokens, next_tokens)

        assert k >= min(overlap, len(prev_tokens))

def test_long_single_paragraph_is_split_under_limit():
    # 80 tokens with no punctuation/newlines to force internal splitting
    words = "word"  # tokenized as a single word
    text = " ".join([words] * 80)
    chunks = chunk_text(text, tokens_per_chunk=16, overlap=4)

    assert len(chunks) >= 5  # 80 with window 16 & overlap 4 → stride 12 → ~7 chunks (rough check)

    for c in chunks:
        assert c["token_count"] <= 16

def test_unicode_and_punctuation_are_ok():

    text = (
        "UN fires Central Africa legal adviser who accused peacekeepers of massacre May 31, 2018 \n\n"
        "Oil prices dip on unexpected growth in US crude stocks May 31, 2018"
    )
    chunks = chunk_text(text, tokens_per_chunk=10, overlap=3)

    assert len(chunks) >= 1

    # Ensure token count aligns with our simple tokenizer
    for c in chunks:
        assert c["token_count"] == len(toks(c["text"])) <= 10

def test_huge_overlap_is_clamped_and_does_not_break():
    text = (
        "U.N. fires Central Africa legal adviser who accused peacekeepers of massacre May 31, 2018."
    )

    chunks = chunk_text(text, tokens_per_chunk=8, overlap=10_000)

    assert len(chunks) >= 1

    for c in chunks:
        assert c["token_count"] <= 8

def test_tokens_per_chunk_must_be_positive():
    with pytest.raises((AssertionError, ValueError)):
        _ = chunk_text("a b c", tokens_per_chunk=0, overlap=0)

def test_deterministic_results():
    text = (
        "Para A line1.\nline2.\n\n"
        "Para B line1.\nline2.\n\n"
        "Para C line1."
    )

    out1 = chunk_text(text, tokens_per_chunk=7, overlap=2)
    out2 = chunk_text(text, tokens_per_chunk=7, overlap=2)

    assert out1 == out2
"""Tokenization utilities for BDH inference."""
import torch
from typing import List


def tokenize_text(text: str) -> torch.Tensor:
    """Convert text to byte-level tokens (vocab_size=256)."""
    byte_array = bytearray(text, "utf-8")
    return torch.tensor(list(byte_array), dtype=torch.long)


def chunk_tokens(tokens: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    """Split tokens into chunks."""
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(tokens[i:i+chunk_size])
    return chunks

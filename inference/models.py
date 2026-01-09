"""Data models for BDH inference."""
import torch
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Constraint:
    """Represents a constraint extracted from backstory."""
    text: str
    category: str  # 'trait', 'belief', 'vow', 'fear', 'commitment'
    polarity: str  # 'positive', 'negative', 'neutral'
    confidence: float


@dataclass
class TemporalEvent:
    """Tracks when something happens in the story."""
    chunk_idx: int
    event_type: str
    description: str
    constraint_related: Optional[str] = None


@dataclass
class StateSnapshot:
    """Snapshot of BDH state at a given chunk - MEMORY SAFE.
    
    Stores only lightweight summaries:
    - Mean pooled embedding [D] on CPU
    - Scalar diagnostics
    - Constraint signals
    
    Memory per chunk: ~4 KB
    """
    chunk_idx: int
    state_mean: torch.Tensor  # [D] - mean pooled embedding on CPU
    state_norm: float
    sparsity: float
    delta_norm: float
    constraint_signals: Dict[str, float]

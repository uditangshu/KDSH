"""Level 1: Semantic Relevance Detection - IMPROVED.

Finds chunks that are semantically relevant to each constraint.
Outputs: relevant_chunks_per_constraint (used by Level 2)
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Set
from ..models import Constraint


class Level1SemanticRelevance:
    """Level 1: Find semantically relevant chunks for each constraint.
    
    Instead of just computing global similarity, we:
    1. Encode each constraint as an embedding
    2. Find which story chunks are relevant to which constraints
    3. Pass this to Level 2 for temporal analysis
    """
    
    def __init__(self, state_tracker, model, config, device):
        self.state_tracker = state_tracker
        self.model = model
        self.config = config
        self.device = device
        self.constraint_embeddings: Dict[str, torch.Tensor] = {}
    
    def compute(self, constraints: List[Constraint], backstory_summary: torch.Tensor) -> Dict:
        """Find relevant chunks for each constraint.
        
        Returns:
            relevant_chunks: Dict[constraint_text -> List[chunk_idx]]
            chunk_constraint_scores: Dict[chunk_idx -> Dict[constraint_text -> score]]
        """
        if not self.state_tracker.state_history:
            return {
                'relevant_chunks': {},
                'chunk_constraint_scores': {},
                'alignment_score': 0.5,
            }
        
        # Step 1: Encode constraints as embeddings
        self._encode_constraints(constraints)
        
        # Step 2: For each chunk, compute relevance to each constraint
        relevant_chunks: Dict[str, List[int]] = {c.text: [] for c in constraints}
        chunk_constraint_scores: Dict[int, Dict[str, float]] = {}
        
        relevance_threshold = 0.3  # Chunks with score > threshold are relevant
        
        for snapshot in self.state_tracker.state_history:
            chunk_idx = snapshot.chunk_idx
            chunk_mean = snapshot.state_mean
            chunk_scores: Dict[str, float] = {}
            
            for constraint in constraints:
                # Compute semantic similarity between chunk and constraint
                constraint_emb = self.constraint_embeddings.get(constraint.text)
                if constraint_emb is not None:
                    score = self._compute_relevance(chunk_mean, constraint_emb, constraint)
                else:
                    score = 0.0
                
                chunk_scores[constraint.text] = score
                
                # If relevant, add to relevant_chunks
                if score > relevance_threshold:
                    relevant_chunks[constraint.text].append(chunk_idx)
            
            chunk_constraint_scores[chunk_idx] = chunk_scores
        
        # Compute overall alignment score (how well story aligns with backstory)
        all_scores = []
        for chunk_scores in chunk_constraint_scores.values():
            all_scores.extend(chunk_scores.values())
        
        alignment_score = np.mean(all_scores) if all_scores else 0.5
        
        return {
            'relevant_chunks': relevant_chunks,
            'chunk_constraint_scores': chunk_constraint_scores,
            'alignment_score': alignment_score,
            'constraint_embeddings': self.constraint_embeddings,
        }
    
    def _encode_constraints(self, constraints: List[Constraint]):
        """Encode each constraint text as an embedding using BDH."""
        self.model.eval()
        with torch.no_grad():
            for constraint in constraints:
                # Tokenize constraint text
                tokens = self._tokenize(constraint.text)
                if len(tokens) == 0:
                    continue
                
                tokens = tokens.unsqueeze(0).to(self.device)
                
                # Get embedding via BDH
                x = self.model.embed(tokens)  # [1, T, D]
                x = self.model.ln(x.unsqueeze(1)).squeeze(1)  # [1, T, D]
                
                # Mean pool
                constraint_emb = x.mean(dim=1).squeeze(0).cpu()  # [D]
                self.constraint_embeddings[constraint.text] = constraint_emb
    
    def _compute_relevance(self, chunk_emb: torch.Tensor, constraint_emb: torch.Tensor, 
                          constraint: Constraint) -> float:
        """Compute semantic relevance between chunk and constraint.
        
        Uses cosine similarity + polarity-aware boosting.
        """
        # Base cosine similarity
        similarity = F.cosine_similarity(
            chunk_emb.unsqueeze(0), constraint_emb.unsqueeze(0), dim=1
        ).item()
        
        # Normalize to [0, 1]
        score = (similarity + 1) / 2
        
        # Boost based on constraint category importance
        category_boost = {
            'vow': 1.3,
            'belief': 1.2,
            'fear': 1.1,
            'trait': 1.0,
            'commitment': 1.2,
        }
        score *= category_boost.get(constraint.category, 1.0)
        
        return min(score, 1.0)
    
    def _tokenize(self, text: str) -> torch.Tensor:
        byte_array = bytearray(text, "utf-8")
        return torch.tensor(list(byte_array), dtype=torch.long)

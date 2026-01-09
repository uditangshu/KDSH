"""Level 1: Semantic Relevance Detection - OPTIMIZED.

Finds chunks that are semantically relevant to each constraint.
OPTIMIZATION: Batched constraint encoding.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Set
from ..models import Constraint


class Level1SemanticRelevance:
    """Level 1: Find semantically relevant chunks for each constraint."""
    
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
        
        # Step 1: Encode constraints as embeddings (BATCHED OPTIMIZATION)
        self._encode_constraints_batched(constraints)
        
        # Step 2: For each chunk, compute relevance to each constraint
        relevant_chunks: Dict[str, List[int]] = {c.text: [] for c in constraints}
        chunk_constraint_scores: Dict[int, Dict[str, float]] = {}
        
        relevance_threshold = 0.3
        
        # Get all chunk embeddings as a single tensor [T_chunks, D]
        chunk_embeddings = torch.stack([s.state_mean for s in self.state_tracker.state_history]).to(self.device)
        
        # Get all constraint embeddings as a single tensor [N_constraints, D]
        # Filter constraints that were successfully embedded
        valid_constraints = [c for c in constraints if c.text in self.constraint_embeddings]
        if not valid_constraints:
             return {'relevant_chunks': {}, 'chunk_constraint_scores': {}, 'alignment_score': 0.5}

        constraint_matrix = torch.stack([self.constraint_embeddings[c.text] for c in valid_constraints]).to(self.device)
        
        # Compute cosine similarity in batches to avoid OOM
        # chunk_embeddings: [T_chunks, D]
        # constraint_matrix: [N_constraints, D]
        
        num_chunks = chunk_embeddings.size(0)
        BATCH_SIZE = 1000  # Adjust based on memory
        scores_list = []
        
        # Compute boost factors [N_constraints]
        boosts = torch.tensor([
            self._get_category_boost(c.category) for c in valid_constraints
        ], device=self.device)
        
        for i in range(0, num_chunks, BATCH_SIZE):
            chunk_batch = chunk_embeddings[i:i+BATCH_SIZE]  # [B, D]
            
            # [B, 1, D] vs [1, N, D] -> [B, N]
            sim_batch = F.cosine_similarity(
                chunk_batch.unsqueeze(1),
                constraint_matrix.unsqueeze(0),
                dim=2
            )
            
            # Apply boosts and normalization immediately to save memory
            scores_batch = ((sim_batch + 1) / 2) * boosts.unsqueeze(0)
            scores_batch = torch.clamp(scores_batch, max=1.0)
            
            # Move to CPU immediately
            scores_list.append(scores_batch.cpu())
            
        # Concatenate all results on CPU
        if scores_list:
            scores_cpu = torch.cat(scores_list, dim=0).numpy()
        else:
            scores_cpu = np.array([])
            
        # scores_matrix for alignment score calculation (approximate or re-mean)
        # We can just use the CPU scores for the mean
        alignment_score = float(np.mean(scores_cpu)) if scores_cpu.size > 0 else 0.5
        
        for i, snapshot in enumerate(self.state_tracker.state_history):
            chunk_idx = snapshot.chunk_idx
            chunk_scores = {}
            
            for j, constraint in enumerate(valid_constraints):
                score = float(scores_cpu[i, j])
                chunk_scores[constraint.text] = score
                
                if score > relevance_threshold:
                    relevant_chunks[constraint.text].append(chunk_idx)
            
            chunk_constraint_scores[chunk_idx] = chunk_scores
        

        
        return {
            'relevant_chunks': relevant_chunks,
            'chunk_constraint_scores': chunk_constraint_scores,
            'alignment_score': alignment_score,
            'constraint_embeddings': self.constraint_embeddings,
        }
    
    def _encode_constraints_batched(self, constraints: List[Constraint]):
        """Encode constraints in batches to speed up processing."""
        self.model.eval()
        texts = [c.text for c in constraints if c.text not in self.constraint_embeddings]
        if not texts:
            return
            
        with torch.no_grad():
            # Pad and batch tokens
            # Simple batching: process one by one but could be optimized to true batching
            # Given varied lengths, one-by-one is safer for now but we reuse computation context
            for text in texts:
                tokens = self._tokenize(text)
                if len(tokens) == 0:
                    continue
                
                tokens = tokens.unsqueeze(0).to(self.device)
                x = self.model.embed(tokens)
                x = self.model.ln(x.unsqueeze(1)).squeeze(1)
                emb = x.mean(dim=1).squeeze(0).cpu() # [D]
                self.constraint_embeddings[text] = emb

    def _get_category_boost(self, category: str) -> float:
        return {
            'vow': 1.3,
            'belief': 1.2,
            'fear': 1.1,
            'trait': 1.0,
            'commitment': 1.2,
        }.get(category, 1.0)
    
    def _tokenize(self, text: str) -> torch.Tensor:
        byte_array = bytearray(text, "utf-8")
        return torch.tensor(list(byte_array), dtype=torch.long)

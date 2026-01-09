"""Level 1: Local Semantic Alignment."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


class Level1LocalSemanticAlignment:
    """Compute local semantic alignment using state history."""
    
    def __init__(self, state_tracker):
        self.state_tracker = state_tracker
    
    def compute(self, backstory_summary: torch.Tensor) -> Dict:
        """Compute local semantic alignment."""
        if not self.state_tracker.state_history:
            return {
                'mean_similarity': 0.5,
                'max_similarity': 0.5,
                'min_similarity': 0.5,
                'chunk_similarities': [],
                'alignment_score': 0.5,
            }
        
        chunk_similarities = []
        backstory_mean = backstory_summary.cpu()
        
        for snapshot in self.state_tracker.state_history:
            chunk_mean = snapshot.state_mean
            similarity = F.cosine_similarity(
                backstory_mean.unsqueeze(0), chunk_mean.unsqueeze(0), dim=1
            ).item()
            chunk_similarities.append(similarity)
        
        mean_similarity = np.mean(chunk_similarities) if chunk_similarities else 0.5
        max_similarity = np.max(chunk_similarities) if chunk_similarities else 0.5
        min_similarity = np.min(chunk_similarities) if chunk_similarities else 0.5
        
        return {
            'mean_similarity': mean_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'chunk_similarities': chunk_similarities,
            'alignment_score': mean_similarity,
        }

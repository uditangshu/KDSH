"""Level 2: Temporal Semantic Alignment."""
import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict
from ..models import Constraint, TemporalEvent
from ..tokenizer import tokenize_text, chunk_tokens

CHUNK_SIZE = 512


class Level2TemporalSemanticAlignment:
    """Temporal Semantic Alignment - WHEN constraints apply."""
    
    def __init__(self, constraint_extractor, state_tracker):
        self.extractor = constraint_extractor
        self.tracker = state_tracker
        self._backstory_summary = None
    
    def compute(self, backstory_text: str, story_tokens: torch.Tensor, 
               constraints: List[Constraint]) -> Dict:
        """Compute temporal alignment scores."""
        if not constraints:
            constraints = self.extractor.extract(backstory_text)
        
        story_chunks = chunk_tokens(story_tokens, CHUNK_SIZE)
        temporal_events: List[TemporalEvent] = []
        
        backstory_tokens = tokenize_text(backstory_text)
        initial_state = self.tracker.initialize_with_backstory(backstory_tokens)
        self._backstory_summary = initial_state.state_mean
        torch.cuda.empty_cache()
        
        constraint_first_occurrence: Dict[str, int] = {}
        constraint_violations: Dict[str, List[int]] = defaultdict(list)
        
        for chunk_idx, chunk in enumerate(story_chunks):
            snapshot = self.tracker.process_story_chunk(chunk, chunk_idx)
            
            for constraint in constraints:
                relevance = self._check_constraint_relevance(chunk, constraint)
                
                if relevance > 0.5:
                    if constraint.text not in constraint_first_occurrence:
                        constraint_first_occurrence[constraint.text] = chunk_idx
                    
                    if self._check_violation(snapshot, constraint):
                        constraint_violations[constraint.text].append(chunk_idx)
            
            if chunk_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        temporal_scores = {}
        for constraint in constraints:
            first_occurrence = constraint_first_occurrence.get(constraint.text, -1)
            violations = constraint_violations[constraint.text]
            
            if first_occurrence >= 0 and violations:
                early_violations = [v for v in violations if v < first_occurrence]
                temporal_scores[constraint.text] = {
                    'first_occurrence': first_occurrence,
                    'violations': violations,
                    'early_violations': early_violations,
                    'consistent': len(early_violations) == 0,
                }
            else:
                temporal_scores[constraint.text] = {
                    'first_occurrence': first_occurrence,
                    'violations': violations,
                    'early_violations': [],
                    'consistent': True,
                }
        
        consistent_count = sum(1 for s in temporal_scores.values() if s['consistent'])
        total_count = len(temporal_scores) if temporal_scores else 1
        temporal_alignment_score = consistent_count / total_count
        
        return {
            'temporal_scores': temporal_scores,
            'alignment_score': temporal_alignment_score,
            'events': temporal_events,
        }
    
    def _check_constraint_relevance(self, chunk_tokens: torch.Tensor, constraint: Constraint) -> float:
        return np.random.random() * 0.3
    
    def _check_violation(self, snapshot, constraint: Constraint) -> bool:
        intensity = snapshot.constraint_signals.get('intensity', 0)
        if constraint.polarity == 'negative' and intensity > 0.7:
            return True
        return False

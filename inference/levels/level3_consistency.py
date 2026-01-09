"""Level 3: Constraint Violation Detection - IMPROVED.

Uses Level 2's temporal violations to identify WHICH constraints are violated.
Adds semantic identity to violations (not just "something is wrong").
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from ..models import Constraint, StateSnapshot


class Level3ConstraintViolation:
    """Level 3: Identify specific constraint violations with semantic identity.
    
    Uses Level 2's temporal_violations and valid_windows to:
    1. Confirm which constraints are actually violated (not just temporally off)
    2. Compute violation severity with semantic grounding
    3. Output violations with identity for Level 4 causal check
    """
    
    def __init__(self, state_tracker):
        self.state_tracker = state_tracker
    
    def compute(self, constraints: List[Constraint],
                level1_results: Dict,
                level2_results: Dict) -> Dict:
        """Detect constraint violations with semantic identity.
        
        Args:
            constraints: List of extracted constraints
            level1_results: Contains constraint_embeddings and chunk_constraint_scores
            level2_results: Contains temporal_violations and valid_windows
        
        Returns:
            confirmed_violations: List of violations with constraint identity
            violation_details: Per-constraint violation analysis
        """
        temporal_violations = level2_results.get('temporal_violations', [])
        valid_windows = level2_results.get('valid_windows', {})
        chunk_constraint_scores = level1_results.get('chunk_constraint_scores', {})
        constraint_embeddings = level1_results.get('constraint_embeddings', {})
        
        confirmed_violations: List[Dict] = []
        violation_details: Dict[str, Dict] = {}
        
        for constraint in constraints:
            constraint_text = constraint.text
            constraint_emb = constraint_embeddings.get(constraint_text)
            window = valid_windows.get(constraint_text, (-1, -1))
            
            # Get all temporal violations for this constraint
            related_temporal = [v for v in temporal_violations 
                               if v['constraint'] == constraint_text]
            
            # Analyze each chunk in the story for semantic violations
            chunk_violations = self._analyze_semantic_violations(
                constraint, constraint_emb, window, chunk_constraint_scores
            )
            
            # Combine temporal and semantic violations
            all_violations = related_temporal + chunk_violations
            
            if all_violations:
                # Compute aggregate severity
                severity = np.mean([v.get('severity', 0.5) for v in all_violations])
                
                confirmed_violations.append({
                    'constraint': constraint_text,
                    'constraint_obj': constraint,
                    'category': constraint.category,
                    'polarity': constraint.polarity,
                    'violations': all_violations,
                    'violation_count': len(all_violations),
                    'aggregate_severity': severity,
                    'valid_window': window,
                })
            
            violation_details[constraint_text] = {
                'temporal_violations': related_temporal,
                'semantic_violations': chunk_violations,
                'total_violations': len(all_violations),
                'is_violated': len(all_violations) > 0,
            }
        
        # Compute overall consistency
        violated_count = sum(1 for v in violation_details.values() if v['is_violated'])
        total_count = len(constraints) if constraints else 1
        overall_consistency = 1.0 - (violated_count / total_count)
        
        return {
            'confirmed_violations': confirmed_violations,
            'violation_details': violation_details,
            'overall_consistency': overall_consistency,
            'violation_count': len(confirmed_violations),
            'violated_constraints': [v['constraint'] for v in confirmed_violations],
        }
    
    def _analyze_semantic_violations(self, constraint: Constraint,
                                     constraint_emb: torch.Tensor,
                                     valid_window: tuple,
                                     chunk_scores: Dict) -> List[Dict]:
        """Analyze chunks for semantic constraint violations."""
        violations = []
        
        if constraint_emb is None:
            return violations
        
        start_idx, end_idx = valid_window
        
        for snapshot in self.state_tracker.state_history:
            chunk_idx = snapshot.chunk_idx
            chunk_mean = snapshot.state_mean
            
            # Get pre-computed relevance score
            scores = chunk_scores.get(chunk_idx, {})
            relevance = scores.get(constraint.text, 0.0)
            
            # Compute semantic contradiction
            contradiction = self._compute_contradiction(
                chunk_mean, constraint_emb, constraint, relevance
            )
            
            if contradiction['is_contradiction']:
                violations.append({
                    'constraint': constraint.text,
                    'type': 'semantic_contradiction',
                    'chunk_idx': chunk_idx,
                    'contradiction_score': contradiction['score'],
                    'reason': contradiction['reason'],
                    'severity': contradiction['score'],
                })
        
        return violations
    
    def _compute_contradiction(self, chunk_emb: torch.Tensor,
                               constraint_emb: torch.Tensor,
                               constraint: Constraint,
                               relevance: float) -> Dict:
        """Check if chunk semantically contradicts constraint."""
        
        # Cosine similarity - Ensure tensors are on same device/CPU
        # Level 3 logic runs on CPU, but chunk_emb might be GPU tensor if not careful
        
        # Ensure CPU
        c_emb = chunk_emb.cpu()
        k_emb = constraint_emb.cpu()
        
        # Manual cosine similarity for scalar result
        dot_product = float(torch.dot(c_emb, k_emb))
        norm_c = float(torch.norm(c_emb))
        norm_k = float(torch.norm(k_emb))
        
        similarity = dot_product / (norm_c * norm_k + 1e-8)
        
        # Contradiction detection based on polarity
        is_contradiction = False
        score = 0.0
        reason = ""
        
        if constraint.polarity == 'negative':
            # Negative constraint (e.g., "never lies")
            # High similarity to negative action = contradiction
            if similarity > 0.7 and relevance > 0.5:
                is_contradiction = True
                score = similarity
                reason = f"High similarity ({similarity:.2f}) to negative constraint"
        
        elif constraint.polarity == 'positive':
            # Positive constraint (e.g., "always honest")
            # Low similarity when expected = contradiction
            if similarity < 0.2 and relevance < 0.2:
                is_contradiction = True
                score = 1.0 - similarity
                reason = f"Low similarity ({similarity:.2f}) to positive constraint"
        
        else:  # neutral
            # For neutral constraints, large deviation is suspicious
            if abs(similarity) < 0.1:
                is_contradiction = True
                score = 0.5
                reason = "Neutral constraint has near-zero alignment"
        
        return {
            'is_contradiction': is_contradiction,
            'score': score,
            'reason': reason,
            'similarity': similarity,
        }

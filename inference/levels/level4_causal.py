"""Level 4: Causal Justification Check - IMPROVED.

Uses Level 3's violations to check if they are causally justified.
A violation might be acceptable if there's narrative support (e.g., character arc).
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from ..models import StateSnapshot


class Level4CausalJustification:
    """Level 4: Check if Level 3's violations are causally justified.
    
    Not all violations are errors - some are intentional character changes.
    This level checks:
    1. Is there gradual state change leading to the violation?
    2. Is there narrative build-up (gradual delta_norm increase)?
    3. Does the violation make causal sense in context?
    """
    
    def __init__(self, state_tracker):
        self.state_tracker = state_tracker
    
    def compute(self, level3_results: Dict) -> Dict:
        """Check causal justification for each violation.
        
        Args:
            level3_results: Contains confirmed_violations with chunk indices
        
        Returns:
            justified_violations: Violations that have causal support
            unjustified_violations: True errors (no causal support)
            final_violations: Only unjustified ones (for decision)
        """
        confirmed_violations = level3_results.get('confirmed_violations', [])
        
        justified_violations: List[Dict] = []
        unjustified_violations: List[Dict] = []
        
        for violation_group in confirmed_violations:
            constraint_text = violation_group['constraint']
            violations = violation_group.get('violations', [])
            
            for violation in violations:
                chunk_idx = violation.get('chunk_idx', -1)
                
                if chunk_idx < 0:
                    # No specific chunk - can't check causality
                    unjustified_violations.append(violation)
                    continue
                
                # Check causal justification
                justification = self._check_causal_justification(
                    chunk_idx, violation, constraint_text
                )
                
                if justification['is_justified']:
                    violation['justification'] = justification
                    justified_violations.append(violation)
                else:
                    violation['justification'] = justification
                    unjustified_violations.append(violation)
        
        # Also check for implausible state jumps (original Level 4 logic)
        implausible_jumps = self._detect_implausible_jumps()
        
        # Compute plausibility score
        total_violations = len(justified_violations) + len(unjustified_violations)
        unjustified_count = len(unjustified_violations)
        plausibility_score = 1.0 - (unjustified_count / max(total_violations, 1))
        
        return {
            'justified_violations': justified_violations,
            'unjustified_violations': unjustified_violations,
            'final_violations': unjustified_violations,  # These are the real errors
            'implausible_jumps': implausible_jumps,
            'plausibility_score': plausibility_score,
            'justified_count': len(justified_violations),
            'unjustified_count': unjustified_count,
        }
    
    def _check_causal_justification(self, chunk_idx: int, 
                                    violation: Dict,
                                    constraint_text: str) -> Dict:
        """Check if the violation at chunk_idx has causal support."""
        
        # Get surrounding context
        history = self.state_tracker.state_history
        
        if chunk_idx < 0 or chunk_idx >= len(history):
            return {'is_justified': False, 'reason': 'Invalid chunk index'}
        
        current_snapshot = history[chunk_idx]
        
        # Check 1: Is there gradual build-up? (look at previous 5 chunks)
        lookback = 5
        start_idx = max(0, chunk_idx - lookback)
        preceding = history[start_idx:chunk_idx]
        
        if not preceding:
            return {'is_justified': False, 'reason': 'No preceding context'}
        
        # Compute delta_norm progression
        delta_progression = [s.delta_norm for s in preceding]
        if len(delta_progression) >= 2:
            is_gradual = self._is_gradual_progression(delta_progression)
        else:
            is_gradual = False
        
        # Check 2: Is there semantic continuity?
        semantic_continuity = self._check_semantic_continuity(preceding, current_snapshot)
        
        # Check 3: Is the sparsity changing gradually?
        sparsity_progression = [s.sparsity for s in preceding]
        sparsity_change = abs(current_snapshot.sparsity - np.mean(sparsity_progression)) if sparsity_progression else 1.0
        is_smooth_sparsity = sparsity_change < 0.2
        
        # Combine checks
        justification_score = (
            0.4 * float(is_gradual) +
            0.4 * semantic_continuity +
            0.2 * float(is_smooth_sparsity)
        )
        
        is_justified = justification_score > 0.5
        
        return {
            'is_justified': is_justified,
            'justification_score': justification_score,
            'is_gradual': is_gradual,
            'semantic_continuity': semantic_continuity,
            'sparsity_smooth': is_smooth_sparsity,
            'reason': 'Gradual narrative progression' if is_justified else 'Abrupt unexplained change',
        }
    
    def _is_gradual_progression(self, delta_progression: List[float]) -> bool:
        """Check if delta_norm increases gradually (not suddenly)."""
        if len(delta_progression) < 2:
            return True
        
        # Check if changes are monotonic-ish (gradual increase/decrease)
        diffs = [delta_progression[i+1] - delta_progression[i] 
                 for i in range(len(delta_progression)-1)]
        
        # Small variance in diffs = gradual
        variance = np.var(diffs) if diffs else 0
        return variance < 0.1
    
    def _check_semantic_continuity(self, preceding: List[StateSnapshot],
                                   current: StateSnapshot) -> float:
        """Check semantic continuity with preceding chunks."""
        if not preceding:
            return 0.0
        
        # Average similarity with preceding chunks
        similarities = []
        for prev in preceding:
            sim = F.cosine_similarity(
                prev.state_mean.unsqueeze(0),
                current.state_mean.unsqueeze(0),
                dim=1
            ).item()
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def _detect_implausible_jumps(self) -> List[Dict]:
        """Detect implausible state jumps (sudden large changes)."""
        history = self.state_tracker.state_history
        implausible = []
        
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            # Compute magnitude of jump
            magnitude = curr.delta_norm / max(prev.state_norm, 1e-8)
            magnitude = min(magnitude, 1.0)
            
            if magnitude > 0.8:
                # Large jump - check if justified
                has_support = self._has_narrative_support(prev, curr)
                
                if not has_support:
                    implausible.append({
                        'from_chunk': prev.chunk_idx,
                        'to_chunk': curr.chunk_idx,
                        'magnitude': magnitude,
                        'reason': 'Unsupported large state change',
                    })
        
        return implausible
    
    def _has_narrative_support(self, prev: StateSnapshot, curr: StateSnapshot) -> bool:
        """Check if state transition has narrative support."""
        delta_normalized = curr.delta_norm / max(prev.state_norm, 1e-8)
        return delta_normalized < 0.5

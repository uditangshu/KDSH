"""Level 3: Constraint Consistency Checking."""
import numpy as np
from typing import Dict, List
from ..models import Constraint


class Level3ConstraintConsistencyChecking:
    """Constraint Consistency Checking - BDH state-based violation detection."""
    
    def __init__(self, state_tracker):
        self.tracker = state_tracker
    
    def compute(self, constraints: List[Constraint]) -> Dict:
        """Check constraint consistency using BDH state history."""
        violation_signals = []
        consistency_scores = []
        
        for constraint in constraints:
            constraint_strength_history = []
            
            for snapshot in self.tracker.state_history:
                strength = self._get_constraint_strength(snapshot, constraint)
                constraint_strength_history.append(strength)
            
            violations = self._detect_violations(constraint, constraint_strength_history)
            violation_signals.extend(violations)
            
            consistency = 1.0 - (len(violations) / max(len(constraint_strength_history), 1))
            consistency_scores.append(consistency)
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        return {
            'violation_signals': violation_signals,
            'consistency_scores': consistency_scores,
            'overall_consistency': overall_consistency,
            'violation_count': len(violation_signals),
        }
    
    def _get_constraint_strength(self, snapshot, constraint: Constraint) -> float:
        summary = snapshot.state_mean
        state_variance = summary.var().item()
        return min(state_variance * 10, 1.0)
    
    def _detect_violations(self, constraint: Constraint, strength_history: List[float]) -> List[int]:
        violations = []
        
        if constraint.polarity == 'negative':
            threshold = 0.7
            for idx, strength in enumerate(strength_history):
                if strength > threshold:
                    violations.append(idx)
        elif constraint.polarity == 'positive':
            threshold = 0.2
            for idx, strength in enumerate(strength_history):
                if strength < threshold:
                    violations.append(idx)
        
        return violations

"""Level 2: Temporal Window Validation - IMPROVED.

Uses Level 1's relevant chunks to validate temporal consistency.
Checks: Does the constraint appear at valid times? Before/after expected?
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from ..models import Constraint, StateSnapshot


class Level2TemporalValidation:
    """Level 2: Validate temporal consistency of constraints.
    
    Uses Level 1's relevant_chunks to:
    1. Find when each constraint first appears
    2. Check if constraint usage is temporally consistent
    3. Identify temporal violations (used by Level 3)
    """
    
    def __init__(self, state_tracker):
        self.state_tracker = state_tracker
    
    def compute(self, constraints: List[Constraint], 
                level1_results: Dict) -> Dict:
        """Validate temporal consistency using Level 1's relevant chunks.
        
        Args:
            constraints: List of extracted constraints
            level1_results: Output from Level 1 containing relevant_chunks
        
        Returns:
            valid_windows: Dict[constraint_text -> (start_idx, end_idx)]
            temporal_violations: List of violations to pass to Level 3
        """
        relevant_chunks = level1_results.get('relevant_chunks', {})
        chunk_constraint_scores = level1_results.get('chunk_constraint_scores', {})
        
        valid_windows: Dict[str, Tuple[int, int]] = {}
        temporal_violations: List[Dict] = []
        constraint_timelines: Dict[str, List[float]] = {}
        
        for constraint in constraints:
            constraint_text = constraint.text
            relevant = relevant_chunks.get(constraint_text, [])
            
            if not relevant:
                # Constraint never appears - could be a violation
                valid_windows[constraint_text] = (-1, -1)
                if constraint.polarity == 'positive':
                    # Positive constraint should appear but doesn't
                    temporal_violations.append({
                        'constraint': constraint_text,
                        'type': 'missing_positive_constraint',
                        'expected_presence': True,
                        'actual_presence': False,
                        'severity': 0.7,
                    })
                continue
            
            # Find temporal window
            first_occurrence = min(relevant)
            last_occurrence = max(relevant)
            valid_windows[constraint_text] = (first_occurrence, last_occurrence)
            
            # Build timeline of constraint strength
            timeline = []
            for snapshot in self.state_tracker.state_history:
                chunk_idx = snapshot.chunk_idx
                scores = chunk_constraint_scores.get(chunk_idx, {})
                score = scores.get(constraint_text, 0.0)
                timeline.append(score)
            constraint_timelines[constraint_text] = timeline
            
            # Check for temporal violations
            violations = self._check_temporal_violations(
                constraint, timeline, first_occurrence, last_occurrence
            )
            temporal_violations.extend(violations)
        
        # Compute alignment score
        total_constraints = len(constraints)
        violated_constraints = len(set(v['constraint'] for v in temporal_violations))
        alignment_score = 1.0 - (violated_constraints / max(total_constraints, 1))
        
        return {
            'valid_windows': valid_windows,
            'temporal_violations': temporal_violations,
            'constraint_timelines': constraint_timelines,
            'alignment_score': alignment_score,
            'violation_count': len(temporal_violations),
        }
    
    def _check_temporal_violations(self, constraint: Constraint, 
                                   timeline: List[float],
                                   first_occ: int, last_occ: int) -> List[Dict]:
        """Check for temporal violations in constraint timeline."""
        violations = []
        
        if not timeline:
            return violations
        
        # Violation 1: Negative constraint appears strongly before expected
        if constraint.polarity == 'negative':
            # Check if there's strong presence before first explicit mention
            for idx, score in enumerate(timeline):
                if idx < first_occ and score > 0.6:
                    violations.append({
                        'constraint': constraint.text,
                        'type': 'early_negative_presence',
                        'chunk_idx': idx,
                        'score': score,
                        'severity': 0.8,
                    })
        
        # Violation 2: Positive constraint disappears after being established
        if constraint.polarity == 'positive' and last_occ > first_occ:
            window_avg = np.mean(timeline[first_occ:last_occ+1])
            post_window = timeline[last_occ+1:] if last_occ+1 < len(timeline) else []
            
            if post_window:
                post_avg = np.mean(post_window)
                if post_avg < window_avg * 0.3:  # Significant drop
                    violations.append({
                        'constraint': constraint.text,
                        'type': 'positive_constraint_fades',
                        'window_avg': window_avg,
                        'post_avg': post_avg,
                        'severity': 0.6,
                    })
        
        # Violation 3: Inconsistent presence (high variance)
        if len(timeline) > 5:
            variance = np.var(timeline)
            if variance > 0.15:  # High variance indicates inconsistency
                violations.append({
                    'constraint': constraint.text,
                    'type': 'inconsistent_presence',
                    'variance': variance,
                    'severity': 0.5,
                })
        
        return violations

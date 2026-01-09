"""Level 4: Causal Plausibility Matching."""
import torch.nn.functional as F
from typing import Dict


class Level4CausalPlausibilityMatching:
    """Causal Plausibility Matching - detect implausible transitions."""
    
    def __init__(self, state_tracker):
        self.tracker = state_tracker
    
    def compute(self) -> Dict:
        """Detect implausible causal transitions."""
        state_transitions = []
        implausible_jumps = []
        
        for i in range(1, len(self.tracker.state_history)):
            prev_snapshot = self.tracker.state_history[i-1]
            curr_snapshot = self.tracker.state_history[i]
            
            magnitude = curr_snapshot.delta_norm / max(prev_snapshot.state_norm, 1e-8)
            magnitude = min(magnitude, 1.0)
            
            direction = F.cosine_similarity(
                prev_snapshot.state_mean.unsqueeze(0),
                curr_snapshot.state_mean.unsqueeze(0),
                dim=1
            ).item()
            
            state_transitions.append({
                'from_chunk': prev_snapshot.chunk_idx,
                'to_chunk': curr_snapshot.chunk_idx,
                'magnitude': magnitude,
                'direction': direction,
            })
            
            if magnitude > 0.8:
                has_narrative_support = self._check_narrative_support(prev_snapshot, curr_snapshot)
                
                if not has_narrative_support:
                    implausible_jumps.append({
                        'from_chunk': prev_snapshot.chunk_idx,
                        'to_chunk': curr_snapshot.chunk_idx,
                        'magnitude': magnitude,
                        'reason': 'unsupported_state_flip',
                    })
        
        total_transitions = len(state_transitions)
        implausible_count = len(implausible_jumps)
        plausibility_score = 1.0 - (implausible_count / max(total_transitions, 1))
        
        return {
            'state_transitions': state_transitions,
            'implausible_jumps': implausible_jumps,
            'plausibility_score': plausibility_score,
            'implausible_count': implausible_count,
        }
    
    def _check_narrative_support(self, prev_snapshot, curr_snapshot) -> bool:
        delta_norm_normalized = curr_snapshot.delta_norm / max(prev_snapshot.state_norm, 1e-8)
        return delta_norm_normalized < 0.5

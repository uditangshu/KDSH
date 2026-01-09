"""Core inference orchestration."""
import torch
from typing import Dict, Tuple, List

from .models import Constraint
from .tokenizer import tokenize_text
from .constraints import ConstraintExtractor
from .state_tracker import BDHStateTracker
from .levels import (
    Level1LocalSemanticAlignment,
    Level2TemporalSemanticAlignment,
    Level3ConstraintConsistencyChecking,
    Level4CausalPlausibilityMatching,
)
from .pathway_streams import PathwayInferencePipeline


class AdvancedBDHInference:
    """Main inference class combining all 4 levels + Pathway streaming."""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        self.constraint_extractor = ConstraintExtractor()
        self.state_tracker = BDHStateTracker(model, config, device)
        self.level1 = Level1LocalSemanticAlignment(self.state_tracker)
        self.level2 = Level2TemporalSemanticAlignment(self.constraint_extractor, self.state_tracker)
        self.level3 = Level3ConstraintConsistencyChecking(self.state_tracker)
        self.level4 = Level4CausalPlausibilityMatching(self.state_tracker)
        
        self.pathway_pipeline = None
    
    def predict(self, backstory_text: str, story_text: str, use_streaming: bool = False) -> Dict:
        """Run complete 4-level inference pipeline."""
        self.state_tracker.reset()
        
        story_tokens = tokenize_text(story_text)
        constraints = self.constraint_extractor.extract(backstory_text)
        
        # Initialize Pathway if streaming mode
        if use_streaming:
            self.pathway_pipeline = PathwayInferencePipeline(
                self.model, self.config, self.device, constraints
            )
        
        # Level 2: Process story and build state_history
        level2_results = self.level2.compute(backstory_text, story_tokens, constraints)
        backstory_summary = self.level2._backstory_summary
        torch.cuda.empty_cache()
        
        # Initialize Pathway streams with backstory
        if self.pathway_pipeline:
            self.pathway_pipeline.initialize_with_backstory(backstory_summary)
            for snapshot in self.state_tracker.state_history:
                self.pathway_pipeline.process_chunk(snapshot)
        
        # Level 1: Reuse state_history
        level1_results = self.level1.compute(backstory_summary)
        torch.cuda.empty_cache()
        
        # Level 3: Constraint Consistency
        level3_results = self.level3.compute(constraints)
        torch.cuda.empty_cache()
        
        # Level 4: Causal Plausibility
        level4_results = self.level4.compute()
        torch.cuda.empty_cache()
        
        # Combine results
        prediction, confidence, rationale = self._combine_results(
            level1_results, level2_results, level3_results, level4_results
        )
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'rationale': rationale,
            'level1': level1_results,
            'level2': level2_results,
            'level3': level3_results,
            'level4': level4_results,
            'constraints': constraints,
        }
        
        # Add Pathway streaming report
        if self.pathway_pipeline:
            result['pathway_report'] = self.pathway_pipeline.get_final_report()
        
        self.state_tracker.reset()
        return result
    
    def _combine_results(self, l1: Dict, l2: Dict, l3: Dict, l4: Dict) -> Tuple[int, float, str]:
        """Combine results from all levels."""
        weights = {'l1': 0.2, 'l2': 0.3, 'l3': 0.3, 'l4': 0.2}
        
        l1_score = l1['alignment_score']
        l2_score = l2['alignment_score']
        l3_score = l3['overall_consistency']
        l4_score = l4['plausibility_score']
        
        combined_score = (
            weights['l1'] * l1_score +
            weights['l2'] * l2_score +
            weights['l3'] * l3_score +
            weights['l4'] * l4_score
        )
        
        prediction = 1 if combined_score > 0.5 else 0
        confidence = abs(combined_score - 0.5) * 2
        
        rationale_parts = []
        if l3['violation_count'] > 0:
            rationale_parts.append(f"Found {l3['violation_count']} constraint violations")
        if l4['implausible_count'] > 0:
            rationale_parts.append(f"Detected {l4['implausible_count']} implausible causal jumps")
        if l2['alignment_score'] < 0.5:
            rationale_parts.append("Temporal alignment issues detected")
        
        if not rationale_parts:
            rationale_parts.append("No major inconsistencies detected")
        
        rationale = ". ".join(rationale_parts)
        
        return prediction, confidence, rationale

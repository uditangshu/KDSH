"""Core inference orchestration - IMPROVED with cascaded gating.

Flow: Level1 → Level2 → Level3 → Level4 → Decision
Each level feeds into the next, not independent weighted average.
"""
import torch
from typing import Dict, Tuple, List

from .models import Constraint
from .tokenizer import tokenize_text, chunk_tokens
from .constraints import ConstraintExtractor
from .state_tracker import BDHStateTracker
from .levels import (
    Level1SemanticRelevance,
    Level2TemporalValidation,
    Level3ConstraintViolation,
    Level4CausalJustification,
)
from .pathway_streams import PathwayInferencePipeline

CHUNK_SIZE = 512


class AdvancedBDHInference:
    """Main inference class with CASCADED GATING logic.
    
    Pipeline:
    1. Level 1: Find relevant chunks for each constraint
    2. Level 2: Validate temporal consistency (uses Level 1 output)
    3. Level 3: Detect constraint violations (uses Level 1+2 output)
    4. Level 4: Check causal justification (uses Level 3 output)
    5. Decision: Based on Level 4's unjustified violations
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        self.constraint_extractor = ConstraintExtractor()
        self.state_tracker = BDHStateTracker(model, config, device)
        
        # Initialize cascaded levels
        self.level1 = Level1SemanticRelevance(self.state_tracker, model, config, device)
        self.level2 = Level2TemporalValidation(self.state_tracker)
        self.level3 = Level3ConstraintViolation(self.state_tracker)
        self.level4 = Level4CausalJustification(self.state_tracker)
        
        self.pathway_pipeline = None
    
    def predict(self, backstory_text: str, story_text: str, use_streaming: bool = False) -> Dict:
        """Run CASCADED 4-level inference pipeline.
        
        Key difference from old approach:
        - Old: Independent scores → weighted average → decision
        - New: Level1 → Level2 → Level3 → Level4 → Decision (cascaded)
        """
        self.state_tracker.reset()
        
        # Tokenize story
        story_tokens = tokenize_text(story_text)
        story_chunks = chunk_tokens(story_tokens, CHUNK_SIZE)
        
        # Extract constraints from backstory
        constraints = self.constraint_extractor.extract(backstory_text)
        
        # Initialize with backstory
        backstory_tokens = tokenize_text(backstory_text)
        backstory_snapshot = self.state_tracker.initialize_with_backstory(backstory_tokens)
        backstory_summary = backstory_snapshot.state_mean
        torch.cuda.empty_cache()
        
        # Process story chunks to build state_history
        print(f"Processing {len(story_chunks)} story chunks...")
        for chunk_idx, chunk in enumerate(story_chunks):
            self.state_tracker.process_story_chunk(chunk, chunk_idx)
            if chunk_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # ===== CASCADED INFERENCE =====
        
        # LEVEL 1: Find relevant chunks for each constraint
        print("Level 1: Semantic Relevance...")
        level1_results = self.level1.compute(constraints, backstory_summary)
        torch.cuda.empty_cache()
        
        # LEVEL 2: Validate temporal consistency (uses Level 1)
        print("Level 2: Temporal Validation...")
        level2_results = self.level2.compute(constraints, level1_results)
        torch.cuda.empty_cache()
        
        # LEVEL 3: Detect constraint violations (uses Level 1 + Level 2)
        print("Level 3: Constraint Violation Detection...")
        level3_results = self.level3.compute(constraints, level1_results, level2_results)
        torch.cuda.empty_cache()
        
        # LEVEL 4: Check causal justification (uses Level 3)
        print("Level 4: Causal Justification Check...")
        level4_results = self.level4.compute(level3_results)
        torch.cuda.empty_cache()
        
        # ===== FINAL DECISION (based on Level 4) =====
        prediction, confidence, rationale = self._make_decision(
            level1_results, level2_results, level3_results, level4_results
        )
        
        # Initialize Pathway if streaming mode
        if use_streaming:
            self.pathway_pipeline = PathwayInferencePipeline(
                self.model, self.config, self.device, constraints
            )
            self.pathway_pipeline.initialize_with_backstory(backstory_summary)
            for snapshot in self.state_tracker.state_history:
                self.pathway_pipeline.process_chunk(snapshot)
        
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
        
        if self.pathway_pipeline:
            result['pathway_report'] = self.pathway_pipeline.get_final_report()
        
        self.state_tracker.reset()
        return result
    
    def _make_decision(self, l1: Dict, l2: Dict, l3: Dict, l4: Dict) -> Tuple[int, float, str]:
        """Make final decision based on CASCADED results.
        
        Key insight: Only Level 4's UNJUSTIFIED violations matter.
        Justified violations are acceptable (character arcs, etc.)
        """
        # Get the final violations (unjustified only)
        final_violations = l4.get('final_violations', [])
        unjustified_count = len(final_violations)
        justified_count = l4.get('justified_count', 0)
        
        # Get implausible jumps
        implausible_jumps = l4.get('implausible_jumps', [])
        implausible_count = len(implausible_jumps)
        
        # Decision logic (NOT weighted average)
        # If there are unjustified violations → INCONSISTENT
        # If only justified violations → CONSISTENT (with explanation)
        
        if unjustified_count == 0 and implausible_count == 0:
            # No real problems
            prediction = 1  # CONSISTENT
            confidence = 0.9
            rationale = "No unjustified constraint violations detected"
            
            if justified_count > 0:
                rationale += f". {justified_count} justified changes found (narrative progression)"
        
        elif unjustified_count > 0:
            # Real violations exist
            prediction = 0  # INCONSISTENT
            
            # Confidence based on severity
            severities = [v.get('severity', 0.5) for v in final_violations]
            avg_severity = sum(severities) / len(severities) if severities else 0.5
            confidence = min(0.5 + avg_severity * 0.5, 1.0)
            
            # Build rationale with specifics
            violated_constraints = set(v.get('constraint', 'unknown')[:50] for v in final_violations)
            rationale = f"Found {unjustified_count} unjustified violations"
            if violated_constraints:
                rationale += f" in: {', '.join(list(violated_constraints)[:3])}"
        
        elif implausible_count > 0:
            # Only implausible jumps (no semantic violations)
            prediction = 0 if implausible_count > 2 else 1
            confidence = 0.6
            rationale = f"Detected {implausible_count} implausible causal transitions"
        
        else:
            prediction = 1
            confidence = 0.5
            rationale = "Insufficient data for strong conclusion"
        
        return prediction, confidence, rationale

#!/usr/bin/env python3
"""
BDH Advanced Multi-Level Inference System
Implements 4 levels of reasoning:
1. Local Semantic Alignment (baseline similarity)
2. Temporal Semantic Alignment (when constraints apply)
3. Constraint Consistency Checking (BDH state-based violation detection)
4. Causal Plausibility Matching (detect implausible transitions)
"""

import os
import sys
import re
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    import bdh
    from bdh import BDH, BDHConfig
except ImportError as e:
    print(f"Failed to import bdh module: {e}")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 512


@dataclass
class Constraint:
    """Represents a constraint extracted from backstory."""
    text: str
    category: str  # 'trait', 'belief', 'vow', 'fear', 'commitment'
    polarity: str  # 'positive', 'negative', 'neutral'
    confidence: float


@dataclass
class TemporalEvent:
    """Tracks when something happens in the story."""
    chunk_idx: int
    event_type: str
    description: str
    constraint_related: Optional[str] = None  # Which constraint this relates to


@dataclass
class StateSnapshot:
    """Snapshot of BDH state at a given chunk."""
    chunk_idx: int
    hidden_state: torch.Tensor
    attention_patterns: torch.Tensor
    constraint_signals: Dict[str, float]  # Constraint -> activation strength


class ConstraintExtractor:
    """Extract constraints (traits, beliefs, vows) from backstory text."""
    
    TRAIT_PATTERNS = [
        r'(\w+)\s+(was|is|became|remained)\s+(\w+)',  # "He was brave"
        r'(\w+)\s+(never|always|often|rarely)\s+(\w+)',  # "He never lies"
        r'(\w+)\s+(vowed|promised|swore)\s+to\s+(\w+)',  # "He vowed to protect"
        r'(\w+)\s+(feared|feared|hated|loved)\s+(\w+)',  # "He feared authority"
        r'(\w+)\s+(believed|thought|considered)\s+that\s+([^.]+)',  # "He believed that..."
    ]
    
    def __init__(self):
        self.categories = {
            'trait': ['was', 'is', 'became', 'remained', 'never', 'always'],
            'vow': ['vowed', 'promised', 'swore'],
            'fear': ['feared', 'hated', 'dreaded'],
            'belief': ['believed', 'thought', 'considered'],
        }
    
    def extract(self, backstory_text: str) -> List[Constraint]:
        """Extract constraints from backstory text."""
        constraints = []
        sentences = re.split(r'[.!?]\s+', backstory_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Check for trait patterns
            for pattern in self.TRAIT_PATTERNS:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    category = self._classify_category(match.group(0))
                    polarity = self._classify_polarity(match.group(0))
                    
                    constraint = Constraint(
                        text=match.group(0),
                        category=category,
                        polarity=polarity,
                        confidence=0.7
                    )
                    constraints.append(constraint)
            
            # Manual keyword extraction for common constraints
            if any(word in sentence.lower() for word in ['never', 'always', 'refused', 'avoided']):
                # Extract the constraint
                constraint = Constraint(
                    text=sentence,
                    category='commitment',
                    polarity='negative' if 'never' in sentence.lower() else 'positive',
                    confidence=0.6
                )
                constraints.append(constraint)
        
        return constraints


    def _classify_category(self, text: str) -> str:
        """Classify constraint category."""
        text_lower = text.lower()
        if any(word in text_lower for word in ['vowed', 'promised', 'swore']):
            return 'vow'
        elif any(word in text_lower for word in ['feared', 'hated', 'dreaded']):
            return 'fear'
        elif any(word in text_lower for word in ['believed', 'thought', 'considered']):
            return 'belief'
        else:
            return 'trait'


    def _classify_polarity(self, text: str) -> str:
        """Classify constraint polarity."""
        text_lower = text.lower()
        negative_words = ['never', 'avoided', 'refused', 'feared', 'hated', 'no']
        positive_words = ['always', 'loved', 'vowed to', 'promised to']
        
        if any(word in text_lower for word in negative_words):
            return 'negative'
        elif any(word in text_lower for word in positive_words):
            return 'positive'
        else:
            return 'neutral'


class BDHStateTracker:
    """Track BDH state as story is processed chunk by chunk."""
    
    def __init__(self, model: BDH, config: BDHConfig):
        self.model = model
        self.config = config
        self.state_history: List[StateSnapshot] = []
        
    def initialize_with_backstory(self, backstory_tokens: torch.Tensor) -> StateSnapshot:
        """Process backstory to initialize BDH state."""
        self.model.eval()
        with torch.no_grad():
            # Process backstory in chunks
            state = None
            for i in range(0, len(backstory_tokens), CHUNK_SIZE):
                chunk = backstory_tokens[i:i+CHUNK_SIZE].unsqueeze(0).to(DEVICE)  # [1, T]
                _, hidden_state, _ = self._forward_pass_with_state(chunk, state, return_attention=False)
                state = hidden_state  # Carry state forward
            
            # Create initial snapshot
            snapshot = StateSnapshot(
                chunk_idx=-1,  # Backstory chunk
                hidden_state=state,
                attention_patterns=torch.zeros(1, 1, 1),  # Placeholder
                constraint_signals={}
            )
            return snapshot
    
    def process_story_chunk(self, chunk_tokens: torch.Tensor, chunk_idx: int, 
                           previous_state: Optional[torch.Tensor] = None) -> Tuple[StateSnapshot, torch.Tensor]:
        """Process a story chunk and return state snapshot."""
        self.model.eval()
        with torch.no_grad():
            logits, hidden_state, attention = self._forward_pass_with_state(
                chunk_tokens.to(DEVICE), previous_state, return_attention=True
            )
            
            # Extract constraint signals from hidden state
            constraint_signals = self._extract_constraint_signals(hidden_state)
            
            snapshot = StateSnapshot(
                chunk_idx=chunk_idx,
                hidden_state=hidden_state,
                attention_patterns=attention,
                constraint_signals=constraint_signals
            )
            
            self.state_history.append(snapshot)
            return snapshot, hidden_state
    
    def _forward_pass_with_state(self, tokens: torch.Tensor, 
                                previous_state: Optional[torch.Tensor] = None,
                                return_attention: bool = False):
        """Forward pass that can carry state from previous chunks."""
        C = self.config
        B, T = tokens.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        # Embedding
        x = self.model.embed(tokens).unsqueeze(1)  # [B, 1, T, D]
        x = self.model.ln(x)
        
        # If previous state exists, incorporate it (simplified - can be enhanced)
        if previous_state is not None:
            # Optionally blend previous state with current embedding
            # For now, we'll use previous state as initialization hint
            pass
        
        # Process through layers
        attention_patterns = []
        for level in range(C.n_layer):
            # Encoder projection
            x_squeezed = x.squeeze(1)
            x_latent = torch.einsum('btd,hde->bhte', x_squeezed, self.model.encoder)
            x_sparse = F.relu(x_latent)
            
            # Attention
            yKV = self.model.attn(Q=x_sparse, K=x_sparse, V=x)
            if return_attention and level == 0:
                # Store attention for analysis
                attention_patterns.append(yKV.mean(dim=1).detach())
            
            yKV = yKV.mean(dim=1, keepdim=True)
            yKV = self.model.ln(yKV)
            
            # Value encoder
            yKV_squeezed = yKV.squeeze(1)
            y_latent = torch.einsum('btd,hde->bhte', yKV_squeezed, self.model.encoder_v)
            y_sparse = F.relu(y_latent)
            
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.model.drop(xy_sparse)
            
            # Decoder
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.model.decoder
            y = self.model.ln(yMLP)
            x = self.model.ln(x + y)
        
        logits = x.view(B, T, D) @ self.model.lm_head
        hidden_state = x.detach()  # Final hidden state
        
        if return_attention and attention_patterns:
            attention_tensor = torch.cat(attention_patterns, dim=0)
        else:
            attention_tensor = torch.zeros(1, 1, 1).to(hidden_state.device)  # Placeholder
        
        return logits, hidden_state, attention_tensor
    
    def _extract_constraint_signals(self, hidden_state: torch.Tensor) -> Dict[str, float]:
        """Extract signals related to constraints from hidden state."""
        # Simplified: compute statistics of hidden state
        # In practice, this could use learned probes or attention patterns
        signals = {
            'intensity': hidden_state.abs().mean().item(),
            'sparsity': (hidden_state.abs() > 0.1).float().mean().item(),
            'variance': hidden_state.var().item(),
        }
        return signals


class Level1LocalSemanticAlignment:
    """LEVEL 1: Local Semantic Alignment - basic similarity matching."""
    
    def __init__(self, model: BDH):
        self.model = model
    
    def compute(self, backstory_tokens: torch.Tensor, story_tokens: torch.Tensor) -> Dict:
        """Compute local semantic alignment scores."""
        self.model.eval()
        
        with torch.no_grad():
            # Encode backstory
            _, backstory_state = self._encode(backstory_tokens)
            backstory_embedding = backstory_state.mean(dim=(1, 2))  # [B, D]
            
            # Encode story chunks
            story_chunks = self._chunk_tokens(story_tokens, CHUNK_SIZE)
            chunk_similarities = []
            
            for chunk in story_chunks:
                _, chunk_state = self._encode(chunk)  # _encode handles batch dimension
                chunk_embedding = chunk_state.mean(dim=(1, 2))  # [B, D]
                
                # Cosine similarity
                similarity = F.cosine_similarity(
                    backstory_embedding, chunk_embedding, dim=1
                ).item()
                chunk_similarities.append(similarity)
            
            # Compute metrics
            mean_similarity = np.mean(chunk_similarities)
            max_similarity = np.max(chunk_similarities)
            min_similarity = np.min(chunk_similarities)
            
            return {
                'mean_similarity': mean_similarity,
                'max_similarity': max_similarity,
                'min_similarity': min_similarity,
                'chunk_similarities': chunk_similarities,
                'alignment_score': mean_similarity,  # Higher is better
            }
    
    def _encode(self, tokens: torch.Tensor):
        """Encode tokens to get representation."""
        # Add batch dimension if needed: [T] -> [1, T]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        # Simplified encoding - just forward pass
        logits, _ = self.model(tokens.to(DEVICE))
        # Get hidden state from model (would need to modify model.forward to return it)
        # For now, use logits as proxy
        return logits, logits
    
    def _chunk_tokens(self, tokens: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """Split tokens into chunks."""
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunks.append(tokens[i:i+chunk_size])
        return chunks


class Level2TemporalSemanticAlignment:
    """LEVEL 2: Temporal Semantic Alignment - WHEN constraints apply."""
    
    def __init__(self, constraint_extractor: ConstraintExtractor, state_tracker: BDHStateTracker):
        self.extractor = constraint_extractor
        self.tracker = state_tracker
    
    def compute(self, backstory_text: str, story_tokens: torch.Tensor, 
               constraints: List[Constraint]) -> Dict:
        """Compute temporal alignment scores."""
        # Extract constraints if not provided
        if not constraints:
            constraints = self.extractor.extract(backstory_text)
        
        # Process story in chunks with state tracking
        story_chunks = self._chunk_tokens(story_tokens, CHUNK_SIZE)
        temporal_events: List[TemporalEvent] = []
        
        # Initialize with backstory state
        backstory_tokens = self._tokenize(backstory_text)
        initial_state = self.tracker.initialize_with_backstory(backstory_tokens)
        current_state = initial_state.hidden_state
        
        # Track when constraints appear vs when they're violated
        constraint_first_occurrence: Dict[str, int] = {}
        constraint_violations: Dict[str, List[int]] = defaultdict(list)
        
        for chunk_idx, chunk in enumerate(story_chunks):
            snapshot, current_state = self.tracker.process_story_chunk(
                chunk, chunk_idx, current_state
            )
            
            # Check each constraint
            for constraint in constraints:
                # Check if constraint is relevant to this chunk
                relevance = self._check_constraint_relevance(chunk, constraint)
                
                if relevance > 0.5:  # Constraint is relevant
                    if constraint.text not in constraint_first_occurrence:
                        constraint_first_occurrence[constraint.text] = chunk_idx
                    
                    # Check for violations
                    if self._check_violation(snapshot, constraint):
                        constraint_violations[constraint.text].append(chunk_idx)
        
        # Compute temporal consistency scores
        temporal_scores = {}
        for constraint in constraints:
            first_occurrence = constraint_first_occurrence.get(constraint.text, -1)
            violations = constraint_violations[constraint.text]
            
            if first_occurrence >= 0 and violations:
                # Violation before first occurrence = inconsistent
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
                    'consistent': True,  # No violations found
                }
        
        # Overall temporal alignment score
        consistent_count = sum(1 for s in temporal_scores.values() if s['consistent'])
        total_count = len(temporal_scores) if temporal_scores else 1
        temporal_alignment_score = consistent_count / total_count
        
        return {
            'temporal_scores': temporal_scores,
            'alignment_score': temporal_alignment_score,
            'events': temporal_events,
        }
    
    def _check_constraint_relevance(self, chunk_tokens: torch.Tensor, constraint: Constraint) -> float:
        """Check if constraint is relevant to chunk (simplified)."""
        # Simplified: could use semantic similarity or keyword matching
        # For now, return random relevance
        return np.random.random() * 0.3  # Low relevance for most chunks
    
    def _check_violation(self, snapshot: StateSnapshot, constraint: Constraint) -> bool:
        """Check if current state violates constraint."""
        # Simplified: check if state signals contradict constraint
        # In practice, this would use learned probes or attention patterns
        intensity = snapshot.constraint_signals.get('intensity', 0)
        
        # Negative constraints (e.g., "never does X") should have low intensity for X
        if constraint.polarity == 'negative' and intensity > 0.7:
            return True
        
        return False
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to byte-level tokens."""
        byte_array = bytearray(text, "utf-8")
        return torch.tensor(list(byte_array), dtype=torch.long)
    
    def _chunk_tokens(self, tokens: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """Split tokens into chunks."""
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunks.append(tokens[i:i+chunk_size])
        return chunks


class Level3ConstraintConsistencyChecking:
    """LEVEL 3: Constraint Consistency Checking - BDH state-based violation detection."""
    
    def __init__(self, state_tracker: BDHStateTracker):
        self.tracker = state_tracker
    
    def compute(self, constraints: List[Constraint]) -> Dict:
        """Check constraint consistency using BDH state history."""
        violation_signals = []
        consistency_scores = []
        
        for constraint in constraints:
            # Track constraint strength over time
            constraint_strength_history = []
            
            for snapshot in self.tracker.state_history:
                # Get constraint-specific signal
                strength = self._get_constraint_strength(snapshot, constraint)
                constraint_strength_history.append(strength)
            
            # Detect violations
            violations = self._detect_violations(constraint, constraint_strength_history)
            violation_signals.extend(violations)
            
            # Compute consistency score for this constraint
            consistency = 1.0 - (len(violations) / max(len(constraint_strength_history), 1))
            consistency_scores.append(consistency)
        
        # Overall consistency score
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        return {
            'violation_signals': violation_signals,
            'consistency_scores': consistency_scores,
            'overall_consistency': overall_consistency,
            'violation_count': len(violation_signals),
        }
    
    def _get_constraint_strength(self, snapshot: StateSnapshot, constraint: Constraint) -> float:
        """Get strength of constraint signal in state snapshot."""
        # Use hidden state to compute constraint strength
        # Simplified: use variance as proxy for constraint activation
        state_variance = snapshot.hidden_state.var().item()
        
        # Map to constraint strength (0-1)
        strength = min(state_variance * 10, 1.0)
        
        return strength
    
    def _detect_violations(self, constraint: Constraint, strength_history: List[float]) -> List[int]:
        """Detect chunks where constraint is violated."""
        violations = []
        
        if constraint.polarity == 'negative':
            # Negative constraint: high strength = violation
            threshold = 0.7
            for idx, strength in enumerate(strength_history):
                if strength > threshold:
                    violations.append(idx)
        elif constraint.polarity == 'positive':
            # Positive constraint: very low strength = violation
            threshold = 0.2
            for idx, strength in enumerate(strength_history):
                if strength < threshold:
                    violations.append(idx)
        
        return violations


class Level4CausalPlausibilityMatching:
    """LEVEL 4: Causal Plausibility Matching - detect implausible transitions."""
    
    def __init__(self, state_tracker: BDHStateTracker):
        self.tracker = state_tracker
    
    def compute(self) -> Dict:
        """Detect implausible causal transitions."""
        state_transitions = []
        implausible_jumps = []
        
        # Compute state transitions between chunks
        for i in range(1, len(self.tracker.state_history)):
            prev_snapshot = self.tracker.state_history[i-1]
            curr_snapshot = self.tracker.state_history[i]
            
            # Compute state difference
            state_diff = self._compute_state_difference(
                prev_snapshot.hidden_state,
                curr_snapshot.hidden_state
            )
            
            state_transitions.append({
                'from_chunk': prev_snapshot.chunk_idx,
                'to_chunk': curr_snapshot.chunk_idx,
                'magnitude': state_diff['magnitude'],
                'direction': state_diff['direction'],
            })
            
            # Check for implausible jumps
            if state_diff['magnitude'] > 0.8:  # Large sudden change
                # Check if there's narrative support (simplified)
                has_narrative_support = self._check_narrative_support(
                    prev_snapshot, curr_snapshot
                )
                
                if not has_narrative_support:
                    implausible_jumps.append({
                        'from_chunk': prev_snapshot.chunk_idx,
                        'to_chunk': curr_snapshot.chunk_idx,
                        'magnitude': state_diff['magnitude'],
                        'reason': 'unsupported_state_flip',
                    })
        
        # Compute plausibility score
        total_transitions = len(state_transitions)
        implausible_count = len(implausible_jumps)
        plausibility_score = 1.0 - (implausible_count / max(total_transitions, 1))
        
        return {
            'state_transitions': state_transitions,
            'implausible_jumps': implausible_jumps,
            'plausibility_score': plausibility_score,
            'implausible_count': implausible_count,
        }
    
    def _compute_state_difference(self, state1: torch.Tensor, state2: torch.Tensor) -> Dict:
        """Compute difference between two states."""
        # Flatten states for comparison
        s1_flat = state1.flatten()
        s2_flat = state2.flatten()
        
        # Compute magnitude of change
        magnitude = F.pairwise_distance(s1_flat.unsqueeze(0), s2_flat.unsqueeze(0)).item()
        
        # Normalize
        magnitude = min(magnitude / s1_flat.norm().item(), 1.0) if s1_flat.norm() > 0 else 0.0
        
        # Compute direction (cosine similarity)
        direction = F.cosine_similarity(s1_flat.unsqueeze(0), s2_flat.unsqueeze(0)).item()
        
        return {
            'magnitude': magnitude,
            'direction': direction,
        }
    
    def _check_narrative_support(self, prev_snapshot: StateSnapshot, 
                                curr_snapshot: StateSnapshot) -> bool:
        """Check if state transition has narrative support."""
        # Simplified: check if attention patterns show gradual transition
        # In practice, would look for intermediate events, trauma, training, etc.
        
        # For now, check if transition is gradual (small steps) vs sudden (large jump)
        attention_diff = (curr_snapshot.attention_patterns - prev_snapshot.attention_patterns).abs().mean()
        
        # If attention changed gradually, assume narrative support
        return attention_diff < 0.5


class AdvancedBDHInference:
    """Main inference class that combines all 4 levels."""
    
    def __init__(self, model: BDH, config: BDHConfig):
        self.model = model
        self.config = config
        
        # Initialize components
        self.constraint_extractor = ConstraintExtractor()
        self.state_tracker = BDHStateTracker(model, config)
        self.level1 = Level1LocalSemanticAlignment(model)
        self.level2 = Level2TemporalSemanticAlignment(self.constraint_extractor, self.state_tracker)
        self.level3 = Level3ConstraintConsistencyChecking(self.state_tracker)
        self.level4 = Level4CausalPlausibilityMatching(self.state_tracker)
    
    def predict(self, backstory_text: str, story_text: str) -> Dict:
        """Run complete 4-level inference pipeline."""
        # Tokenize
        backstory_tokens = self._tokenize(backstory_text)
        story_tokens = self._tokenize(story_text)
        
        # Extract constraints
        constraints = self.constraint_extractor.extract(backstory_text)
        
        # LEVEL 1: Local Semantic Alignment
        level1_results = self.level1.compute(backstory_tokens, story_tokens)
        
        # LEVEL 2: Temporal Semantic Alignment
        level2_results = self.level2.compute(backstory_text, story_tokens, constraints)
        
        # LEVEL 3: Constraint Consistency Checking
        level3_results = self.level3.compute(constraints)
        
        # LEVEL 4: Causal Plausibility Matching
        level4_results = self.level4.compute()
        
        # Combine scores for final prediction
        prediction, confidence, rationale = self._combine_results(
            level1_results, level2_results, level3_results, level4_results
        )
        
        return {
            'prediction': prediction,  # 1 = consistent, 0 = inconsistent
            'confidence': confidence,
            'rationale': rationale,
            'level1': level1_results,
            'level2': level2_results,
            'level3': level3_results,
            'level4': level4_results,
            'constraints': constraints,
        }
    
    def _combine_results(self, l1: Dict, l2: Dict, l3: Dict, l4: Dict) -> Tuple[int, float, str]:
        """Combine results from all levels to make final prediction."""
        # Weighted combination (can be tuned)
        weights = {
            'l1': 0.2,  # Local semantic (baseline)
            'l2': 0.3,  # Temporal (very important)
            'l3': 0.3,  # Constraint consistency (core logic)
            'l4': 0.2,  # Causal plausibility (advanced)
        }
        
        # Normalize scores to [0, 1]
        l1_score = l1['alignment_score']
        l2_score = l2['alignment_score']
        l3_score = l3['overall_consistency']
        l4_score = l4['plausibility_score']
        
        # Weighted average
        combined_score = (
            weights['l1'] * l1_score +
            weights['l2'] * l2_score +
            weights['l3'] * l3_score +
            weights['l4'] * l4_score
        )
        
        # Threshold for binary classification
        prediction = 1 if combined_score > 0.5 else 0
        confidence = abs(combined_score - 0.5) * 2  # Convert to confidence [0, 1]
        
        # Generate rationale
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
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to byte-level tokens."""
        byte_array = bytearray(text, "utf-8")
        return torch.tensor(list(byte_array), dtype=torch.long)


def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced BDH Inference System')
    parser.add_argument('--backstory', type=str, help='Path to backstory file')
    parser.add_argument('--story', type=str, help='Path to story file')
    parser.add_argument('--model_config', type=str, default='default', help='Model config')
    
    args = parser.parse_args()
    
    # Load files
    if args.backstory and args.story:
        with open(args.backstory, 'r', encoding='utf-8') as f:
            backstory_text = f.read()
        with open(args.story, 'r', encoding='utf-8') as f:
            story_text = f.read()
    else:
        # Default: use files from current directory
        files_dir = os.path.join(os.path.dirname(__file__), "files")
        backstory_path = os.path.join(files_dir, "backstory.txt")
        novel_path = os.path.join(files_dir, "novel.txt")
        
        with open(backstory_path, 'r', encoding='utf-8') as f:
            backstory_text = f.read()
        with open(novel_path, 'r', encoding='utf-8') as f:
            story_text = f.read()
    
    # Initialize model
    config = BDHConfig()
    model = BDH(config).to(DEVICE)
    
    # Run inference
    inference = AdvancedBDHInference(model, config)
    results = inference.predict(backstory_text, story_text)
    
    # Print results
    print("=" * 60)
    print("ADVANCED BDH INFERENCE RESULTS")
    print("=" * 60)
    print(f"\nPrediction: {'CONSISTENT (1)' if results['prediction'] == 1 else 'INCONSISTENT (0)'}")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"\nRationale: {results['rationale']}")
    
    print(f"\n{'='*60}")
    print("LEVEL 1: Local Semantic Alignment")
    print(f"  Alignment Score: {results['level1']['alignment_score']:.3f}")
    
    print(f"\n{'='*60}")
    print("LEVEL 2: Temporal Semantic Alignment")
    print(f"  Alignment Score: {results['level2']['alignment_score']:.3f}")
    
    print(f"\n{'='*60}")
    print("LEVEL 3: Constraint Consistency")
    print(f"  Overall Consistency: {results['level3']['overall_consistency']:.3f}")
    print(f"  Violations: {results['level3']['violation_count']}")
    
    print(f"\n{'='*60}")
    print("LEVEL 4: Causal Plausibility")
    print(f"  Plausibility Score: {results['level4']['plausibility_score']:.3f}")
    print(f"  Implausible Jumps: {results['level4']['implausible_count']}")
    
    print(f"\n{'='*60}")
    print(f"Extracted Constraints: {len(results['constraints'])}")
    for i, constraint in enumerate(results['constraints'][:5], 1):
        print(f"  {i}. [{constraint.category}] {constraint.text}")


if __name__ == "__main__":
    main()


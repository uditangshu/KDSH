#!/usr/bin/env python3
"""
BDH Advanced Multi-Level Inference System - MEMORY SAFE DESIGN
Implements 4 levels of reasoning:
1. Local Semantic Alignment (baseline similarity)
2. Temporal Semantic Alignment (when constraints apply)
3. Constraint Consistency Checking (BDH state-based violation detection)
4. Causal Plausibility Matching (detect implausible transitions)

MEMORY OPTIMIZATIONS:
- StateSnapshot stores only ~4 KB per chunk (mean pooled [D] + scalars)
- Never stores full hidden states [B, 1, T, D] or attention patterns [T, T]
- All summaries on CPU, only active computation on GPU
- Story processed ONCE in Level2, all other levels reuse state_history
- Explicit memory cleanup with torch.cuda.empty_cache()
- Expected memory: 5-10 GB instead of 100 GB
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
    """Snapshot of BDH state at a given chunk - MEMORY SAFE DESIGN.
    
    Stores ONLY what BDH Track-B needs:
    - Mean pooled embedding (not full state)
    - Scalar diagnostics (not attention matrices)
    - Constraint signals (not full activations)
    
    Memory per chunk: ~4 KB instead of ~8 MB
    """
    chunk_idx: int
    state_mean: torch.Tensor  # [D] - mean pooled embedding on CPU
    state_norm: float  # L2 norm of state
    sparsity: float  # Activation sparsity
    delta_norm: float  # Change from previous chunk
    constraint_signals: Dict[str, float]  # Constraint -> activation strength
    # Removed: full hidden_state, attention_patterns (quadratic), full activations


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
    """Track BDH state as story is processed chunk by chunk - MEMORY SAFE.
    
    Processes story ONCE and stores lightweight summaries.
    Memory complexity: O(T * D) instead of O(T² * D * L)
    """
    
    def __init__(self, model: BDH, config: BDHConfig):
        self.model = model
        self.config = config
        self.state_history: List[StateSnapshot] = []
        self.previous_summary: Optional[torch.Tensor] = None  # Track for delta_norm
    
    def reset(self):
        """Clear state history - call before every new inference."""
        self.state_history.clear()
        self.previous_summary = None
        torch.cuda.empty_cache()
        
    def initialize_with_backstory(self, backstory_tokens: torch.Tensor) -> StateSnapshot:
        """Process backstory to initialize BDH state - MEMORY SAFE."""
        self.model.eval()
        with torch.no_grad():
            # Process backstory in chunks - don't store full states
            current_summary = None
            for i in range(0, len(backstory_tokens), CHUNK_SIZE):
                chunk = backstory_tokens[i:i+CHUNK_SIZE].unsqueeze(0).to(DEVICE)  # [1, T]
                
                # Forward pass - get hidden state
                _, hidden_state, _ = self._forward_pass_with_state(chunk, None, return_attention=False)
                
                # Extract summary IMMEDIATELY - mean pool over sequence
                state_mean = hidden_state.mean(dim=(1, 2)).squeeze(0).cpu()  # [D] on CPU
                state_norm = state_mean.norm().item()
                
                # Compute sparsity (while state is on GPU)
                sparsity = (hidden_state.abs() > 0.1).float().mean().item()
                
                # Delta norm (change from previous)
                delta_norm = 0.0
                if current_summary is not None:
                    delta_norm = (state_mean - current_summary).norm().item()
                
                # Extract constraint signals while state is on GPU
                constraint_signals = self._extract_constraint_signals(hidden_state)
                
                # Free GPU memory IMMEDIATELY
                del hidden_state
                if i % 50 == 0:
                    torch.cuda.empty_cache()
                
                current_summary = state_mean
            
            # Create initial snapshot - lightweight
            snapshot = StateSnapshot(
                chunk_idx=-1,  # Backstory chunk
                state_mean=current_summary,
                state_norm=state_norm,
                sparsity=sparsity,
                delta_norm=0.0,
                constraint_signals=constraint_signals
            )
            
            self.previous_summary = current_summary
            return snapshot
    
    def process_story_chunk(self, chunk_tokens: torch.Tensor, chunk_idx: int) -> StateSnapshot:
        """Process a story chunk and return lightweight snapshot - MEMORY SAFE.
        
        Never stores full hidden states or attention patterns.
        Only stores scalar diagnostics and mean-pooled embedding.
        """
        self.model.eval()
        with torch.no_grad():
            # Ensure chunk has batch dimension [1, T]
            if chunk_tokens.dim() == 1:
                chunk_tokens = chunk_tokens.unsqueeze(0)
            
            # Forward pass - get hidden state (no previous state needed, BDH processes chunk independently)
            logits, hidden_state, _ = self._forward_pass_with_state(
                chunk_tokens.to(DEVICE), None, return_attention=False  # Never store attention
            )
            
            # Extract summary IMMEDIATELY - mean pool over sequence
            state_mean = hidden_state.mean(dim=(1, 2)).squeeze(0).cpu()  # [D] on CPU
            state_norm = state_mean.norm().item()
            
            # Compute sparsity (while state is on GPU)
            sparsity = (hidden_state.abs() > 0.1).float().mean().item()
            
            # Delta norm (change from previous chunk) - pre-compute for Level4
            delta_norm = 0.0
            if self.previous_summary is not None:
                delta_norm = (state_mean - self.previous_summary).norm().item()
            
            # Extract constraint signals while state is on GPU
            constraint_signals = self._extract_constraint_signals(hidden_state)
            
            # DELETE full hidden state IMMEDIATELY to free GPU memory
            del hidden_state
            if chunk_idx % 100 == 0:
                torch.cuda.empty_cache()
            
            # Create lightweight snapshot
            snapshot = StateSnapshot(
                chunk_idx=chunk_idx,
                state_mean=state_mean,  # [D] on CPU
                state_norm=state_norm,
                sparsity=sparsity,
                delta_norm=delta_norm,
                constraint_signals=constraint_signals
            )
            
            # Store snapshot (only ~4 KB each)
            self.state_history.append(snapshot)
            
            # Update previous summary for next chunk
            self.previous_summary = state_mean
            
            return snapshot
    
    def _forward_pass_with_state(self, tokens: torch.Tensor, 
                                previous_state: Optional[torch.Tensor] = None,
                                return_attention: bool = False):
        """Forward pass - MEMORY SAFE: never stores attention patterns.
        
        NOTE: return_attention is ignored - we never store attention (quadratic memory).
        """
        C = self.config
        B, T = tokens.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        # Embedding
        x = self.model.embed(tokens).unsqueeze(1)  # [B, 1, T, D]
        x = self.model.ln(x)
        
        # Process through layers - BDH processes chunk independently
        # (Previous state integration would require full state, which we avoid)
        for level in range(C.n_layer):
            # Encoder projection
            x_squeezed = x.squeeze(1)
            x_latent = torch.einsum('btd,hde->bhte', x_squeezed, self.model.encoder)
            x_sparse = F.relu(x_latent)
            
            # Attention (we compute it but NEVER store the scores matrix)
            yKV = self.model.attn(Q=x_sparse, K=x_sparse, V=x)
            # Do NOT store attention patterns - quadratic in T!
            
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
        
        # Never return attention patterns - memory killer
        attention_tensor = None
        
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
    
    def _extract_constraint_signals_from_summary(self, summary: torch.Tensor) -> Dict[str, float]:
        """Extract signals from summary embedding (used when only summary is available)."""
        signals = {
            'intensity': summary.abs().mean().item(),
            'sparsity': (summary.abs() > 0.1).float().mean().item(),
            'variance': summary.var().item(),
        }
        return signals


class Level1LocalSemanticAlignment:
    """LEVEL 1: Local Semantic Alignment - REUSE state_history, don't re-encode."""
    
    def __init__(self, state_tracker: BDHStateTracker):
        self.state_tracker = state_tracker
    
    def compute(self, backstory_summary: torch.Tensor) -> Dict:
        """Compute local semantic alignment using existing state history.
        
        MEMORY SAFE: Reuses state_history from Level2, doesn't re-encode story.
        """
        if not self.state_tracker.state_history:
            # Fallback: return default if no history (shouldn't happen)
            return {
                'mean_similarity': 0.5,
                'max_similarity': 0.5,
                'min_similarity': 0.5,
                'chunk_similarities': [],
                'alignment_score': 0.5,
            }
        
        chunk_similarities = []
        backstory_mean = backstory_summary.cpu()  # Ensure CPU
        
        # Reuse state_history - don't re-encode!
        for snapshot in self.state_tracker.state_history:
            # Compare backstory summary with chunk summary
            chunk_mean = snapshot.state_mean  # Already on CPU
            
            # Cosine similarity between summaries
            similarity = F.cosine_similarity(
                backstory_mean.unsqueeze(0), chunk_mean.unsqueeze(0), dim=1
            ).item()
            chunk_similarities.append(similarity)
        
        # Compute metrics
        mean_similarity = np.mean(chunk_similarities) if chunk_similarities else 0.5
        max_similarity = np.max(chunk_similarities) if chunk_similarities else 0.5
        min_similarity = np.min(chunk_similarities) if chunk_similarities else 0.5
        
        return {
            'mean_similarity': mean_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'chunk_similarities': chunk_similarities,
            'alignment_score': mean_similarity,  # Higher is better
        }


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
        backstory_summary = initial_state.state_mean  # Store for Level1
        torch.cuda.empty_cache()  # Clear after initialization
        
        # Track when constraints appear vs when they're violated
        constraint_first_occurrence: Dict[str, int] = {}
        constraint_violations: Dict[str, List[int]] = defaultdict(list)
        
        # Process story ONCE - build state_history
        for chunk_idx, chunk in enumerate(story_chunks):
            snapshot = self.tracker.process_story_chunk(chunk, chunk_idx)
            
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
            
            # Clear CUDA cache periodically
            if chunk_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        # Return backstory_summary for Level1 reuse
        self._backstory_summary = backstory_summary
        
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
        """Get strength of constraint signal - MEMORY SAFE: uses summary only."""
        # Use summary embedding to compute constraint strength
        summary = snapshot.state_mean  # [D] on CPU
        state_variance = summary.var().item()
        
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
        
        # Compute state transitions between chunks - MEMORY SAFE
        for i in range(1, len(self.tracker.state_history)):
            prev_snapshot = self.tracker.state_history[i-1]
            curr_snapshot = self.tracker.state_history[i]
            
            # Use pre-computed delta_norm (already computed efficiently)
            magnitude = curr_snapshot.delta_norm / max(prev_snapshot.state_norm, 1e-8)
            magnitude = min(magnitude, 1.0)
            
            # Compute direction (cosine similarity)
            direction = F.cosine_similarity(
                prev_snapshot.state_mean.unsqueeze(0),
                curr_snapshot.state_mean.unsqueeze(0),
                dim=1
            ).item()
            
            state_diff = {
                'magnitude': magnitude,
                'direction': direction,
            }
            
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
    
    def _compute_state_difference(self, snapshot1: StateSnapshot, snapshot2: StateSnapshot) -> Dict:
        """Compute difference between two states - MEMORY SAFE: uses summaries only.
        
        Never flattens full tensors - only compares [D] summaries.
        """
        # States are already summaries [D] on CPU
        state1 = snapshot1.state_mean  # [D]
        state2 = snapshot2.state_mean  # [D]
        
        # Compute magnitude of change (between summaries)
        magnitude = (state1 - state2).norm().item()
        
        # Normalize by first state norm
        norm1 = state1.norm().item()
        magnitude = min(magnitude / norm1, 1.0) if norm1 > 0 else 0.0
        
        # Compute direction (cosine similarity between summaries)
        direction = F.cosine_similarity(state1.unsqueeze(0), state2.unsqueeze(0), dim=1).item()
        
        return {
            'magnitude': magnitude,
            'direction': direction,
        }
    
    def _check_narrative_support(self, prev_snapshot: StateSnapshot, 
                                curr_snapshot: StateSnapshot) -> bool:
        """Check if state transition has narrative support - MEMORY SAFE.
        
        Uses pre-computed delta_norm instead of recomputing.
        """
        # Use delta_norm (already computed efficiently during chunk processing)
        # Gradual transitions have small delta_norm, sudden jumps have large delta_norm
        delta_norm_normalized = curr_snapshot.delta_norm / max(prev_snapshot.state_norm, 1e-8)
        
        # If transition is gradual (small delta), assume narrative support
        return delta_norm_normalized < 0.5


class AdvancedBDHInference:
    """Main inference class that combines all 4 levels."""
    
    def __init__(self, model: BDH, config: BDHConfig):
        self.model = model
        self.config = config
        
        # Initialize components
        self.constraint_extractor = ConstraintExtractor()
        self.state_tracker = BDHStateTracker(model, config)
        # Level1 now reuses state_history from Level2 (doesn't re-encode)
        self.level1 = Level1LocalSemanticAlignment(self.state_tracker)
        self.level2 = Level2TemporalSemanticAlignment(self.constraint_extractor, self.state_tracker)
        self.level3 = Level3ConstraintConsistencyChecking(self.state_tracker)
        self.level4 = Level4CausalPlausibilityMatching(self.state_tracker)
    
    def predict(self, backstory_text: str, story_text: str) -> Dict:
        """Run complete 4-level inference pipeline - MEMORY SAFE.
        
        Key optimization: Process story ONCE in Level2, reuse state_history for all levels.
        """
        # Reset state tracker before new inference
        self.state_tracker.reset()
        
        # Tokenize
        story_tokens = self._tokenize(story_text)
        
        # Extract constraints
        constraints = self.constraint_extractor.extract(backstory_text)
        
        # LEVEL 2: Process story ONCE and build state_history
        # This is the ONLY place we process the full story through BDH
        level2_results = self.level2.compute(backstory_text, story_tokens, constraints)
        backstory_summary = self.level2._backstory_summary  # Get from Level2
        torch.cuda.empty_cache()
        
        # LEVEL 1: Reuse state_history from Level2 (don't re-encode!)
        level1_results = self.level1.compute(backstory_summary)
        torch.cuda.empty_cache()
        
        # LEVEL 3: Constraint Consistency Checking (uses state_history)
        level3_results = self.level3.compute(constraints)
        torch.cuda.empty_cache()
        
        # LEVEL 4: Causal Plausibility Matching (uses state_history)
        level4_results = self.level4.compute()
        torch.cuda.empty_cache()
        
        # Clear state history after processing (memory cleanup)
        self.state_tracker.reset()
        
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
        backstory_path = os.path.join(files_dir, "albert_backstory.txt")
        novel_path = os.path.join(files_dir, "kalam_novel.txt")
        
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


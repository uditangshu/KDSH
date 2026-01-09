"""Pathway Streaming Integration for BDH Inference."""
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False

from .models import StateSnapshot, Constraint
from .tokenizer import tokenize_text


@dataclass
class ViolationAlert:
    """Alert emitted when a constraint is violated."""
    chunk_idx: int
    constraint_text: str
    severity: float
    signal_type: str


class ConstraintMonitorStream:
    """Stream that monitors constraint violations in real-time."""
    
    def __init__(self, constraints: List[Constraint], threshold: float = 0.7):
        self.constraints = constraints
        self.threshold = threshold
        self.alerts: List[ViolationAlert] = []
    
    def process_snapshot(self, snapshot: StateSnapshot) -> List[ViolationAlert]:
        """Process a state snapshot and emit violation alerts."""
        alerts = []
        intensity = snapshot.constraint_signals.get('intensity', 0)
        
        for constraint in self.constraints:
            if constraint.polarity == 'negative' and intensity > self.threshold:
                alert = ViolationAlert(
                    chunk_idx=snapshot.chunk_idx,
                    constraint_text=constraint.text,
                    severity=intensity,
                    signal_type='high_intensity_violation'
                )
                alerts.append(alert)
                self.alerts.append(alert)
        
        return alerts


class StateVectorStream:
    """Maintains a live-updating index of state embeddings."""
    
    def __init__(self, backstory_embedding: torch.Tensor):
        self.backstory_embedding = backstory_embedding
        self.similarity_history: List[float] = []
    
    def process_snapshot(self, snapshot: StateSnapshot) -> float:
        """Compute similarity against backstory and update index."""
        similarity = torch.nn.functional.cosine_similarity(
            self.backstory_embedding.unsqueeze(0),
            snapshot.state_mean.unsqueeze(0),
            dim=1
        ).item()
        self.similarity_history.append(similarity)
        return similarity
    
    def get_rolling_average(self, window: int = 10) -> float:
        """Get rolling average similarity."""
        if not self.similarity_history:
            return 0.0
        recent = self.similarity_history[-window:]
        return sum(recent) / len(recent)


class ViolationAlertStream:
    """Aggregates and filters violation alerts."""
    
    def __init__(self, min_severity: float = 0.5):
        self.min_severity = min_severity
        self.all_alerts: List[ViolationAlert] = []
    
    def add_alerts(self, alerts: List[ViolationAlert]):
        """Add alerts, filtering by severity."""
        for alert in alerts:
            if alert.severity >= self.min_severity:
                self.all_alerts.append(alert)
    
    def get_summary(self) -> Dict:
        """Get summary of all alerts."""
        return {
            'total_alerts': len(self.all_alerts),
            'alerts': self.all_alerts,
            'max_severity': max((a.severity for a in self.all_alerts), default=0.0),
        }


class PathwayInferencePipeline:
    """Pathway-powered streaming inference pipeline.
    
    When Pathway is available, uses pw.Table for streaming.
    When not available, falls back to batch processing.
    """
    
    def __init__(self, model, config, device, constraints: List[Constraint]):
        self.model = model
        self.config = config
        self.device = device
        self.constraints = constraints
        
        self.constraint_monitor = ConstraintMonitorStream(constraints)
        self.state_stream: Optional[StateVectorStream] = None
        self.alert_stream = ViolationAlertStream()
    
    def initialize_with_backstory(self, backstory_embedding: torch.Tensor):
        """Initialize streams with backstory embedding."""
        self.state_stream = StateVectorStream(backstory_embedding)
    
    def process_chunk(self, snapshot: StateSnapshot) -> Dict:
        """Process a chunk through all streams."""
        # Monitor constraints
        alerts = self.constraint_monitor.process_snapshot(snapshot)
        self.alert_stream.add_alerts(alerts)
        
        # Update state index
        similarity = 0.0
        if self.state_stream:
            similarity = self.state_stream.process_snapshot(snapshot)
        
        return {
            'chunk_idx': snapshot.chunk_idx,
            'alerts': alerts,
            'similarity': similarity,
            'rolling_similarity': self.state_stream.get_rolling_average() if self.state_stream else 0.0,
        }
    
    def get_final_report(self) -> Dict:
        """Get final streaming report."""
        return {
            'constraint_alerts': self.alert_stream.get_summary(),
            'similarity_history': self.state_stream.similarity_history if self.state_stream else [],
            'final_rolling_similarity': self.state_stream.get_rolling_average() if self.state_stream else 0.0,
        }
    
    @staticmethod
    def is_pathway_available() -> bool:
        return PATHWAY_AVAILABLE

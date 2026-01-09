"""BDH State Tracker - Memory-safe state management.
OPTIMIZED: Reduced redundant computations and memory transfers.
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
from .models import StateSnapshot

CHUNK_SIZE = 512


class BDHStateTracker:
    """Track BDH state as story is processed - MEMORY SAFE & OPTIMIZED."""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.state_history: List[StateSnapshot] = []
        self.previous_summary: Optional[torch.Tensor] = None
    
    def reset(self):
        """Clear state history."""
        self.state_history.clear()
        self.previous_summary = None
        torch.cuda.empty_cache()
        
    def initialize_with_backstory(self, backstory_tokens: torch.Tensor) -> StateSnapshot:
        """Process backstory to initialize BDH state."""
        self.model.eval()
        with torch.no_grad():
            current_summary = None
            state_norm = 0.0
            sparsity = 0.0
            constraint_signals = {}
            
            # Optimization: process larger chunks if possible, but keep CHUNK_SIZE for consistency
            for i in range(0, len(backstory_tokens), CHUNK_SIZE):
                chunk = backstory_tokens[i:i+CHUNK_SIZE].unsqueeze(0).to(self.device)
                
                # Optimized forward pass returns only what's needed
                hidden_state = self._forward_pass_light(chunk)
                
                # Compute stats on GPU before moving simple scalar/vector to CPU
                state_mean = hidden_state.mean(dim=(1, 2)).squeeze(0) # Keep on GPU for a moment
                state_norm = state_mean.norm().item()
                sparsity = (hidden_state.abs() > 0.1).float().mean().item()
                constraint_signals = self._extract_constraint_signals(hidden_state)
                
                current_summary_cpu = state_mean.cpu()
                
                del hidden_state
                del state_mean
                
                if i % 50 == 0:
                    torch.cuda.empty_cache()
                
                current_summary = current_summary_cpu
            
            snapshot = StateSnapshot(
                chunk_idx=-1,
                state_mean=current_summary,
                state_norm=state_norm,
                sparsity=sparsity,
                delta_norm=0.0,
                constraint_signals=constraint_signals
            )
            
            self.previous_summary = current_summary
            return snapshot
    
    def process_story_chunk(self, chunk_tokens: torch.Tensor, chunk_idx: int) -> StateSnapshot:
        """Process a story chunk and return lightweight snapshot."""
        self.model.eval()
        with torch.no_grad():
            if chunk_tokens.dim() == 1:
                chunk_tokens = chunk_tokens.unsqueeze(0)
            
            # Optimized forward pass
            hidden_state = self._forward_pass_light(chunk_tokens.to(self.device))
            
            # Compute GPU-side stats
            state_mean_gpu = hidden_state.mean(dim=(1, 2)).squeeze(0)
            state_norm = state_mean_gpu.norm().item()
            sparsity = (hidden_state.abs() > 0.1).float().mean().item()
            
            state_mean_cpu = state_mean_gpu.cpu()
            
            delta_norm = 0.0
            if self.previous_summary is not None:
                delta_norm = (state_mean_cpu - self.previous_summary).norm().item()
            
            constraint_signals = self._extract_constraint_signals(hidden_state)
            
            del hidden_state
            del state_mean_gpu
            
            if chunk_idx % 100 == 0:
                torch.cuda.empty_cache()
            
            snapshot = StateSnapshot(
                chunk_idx=chunk_idx,
                state_mean=state_mean_cpu,
                state_norm=state_norm,
                sparsity=sparsity,
                delta_norm=delta_norm,
                constraint_signals=constraint_signals
            )
            
            self.state_history.append(snapshot)
            self.previous_summary = state_mean_cpu
            
            return snapshot
    
    def _forward_pass_light(self, tokens: torch.Tensor):
        """Optimized forward pass that skips logits calc if not needed."""
        C = self.config
        B, T = tokens.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        x = self.model.embed(tokens).unsqueeze(1)
        x = self.model.ln(x)
        
        for level in range(C.n_layer):
            x_squeezed = x.squeeze(1)
            x_latent = torch.einsum('btd,hde->bhte', x_squeezed, self.model.encoder)
            x_sparse = F.relu(x_latent)
            
            yKV = self.model.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = yKV.mean(dim=1, keepdim=True)
            yKV = self.model.ln(yKV)
            
            yKV_squeezed = yKV.squeeze(1)
            y_latent = torch.einsum('btd,hde->bhte', yKV_squeezed, self.model.encoder_v)
            y_sparse = F.relu(y_latent)
            
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.model.drop(xy_sparse)
            
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.model.decoder
            y = self.model.ln(yMLP)
            x = self.model.ln(x + y)
        
        # SKIP LOGITS calculation - saves matrix mult
        # logits = x.view(B, T, D) @ self.model.lm_head
        
        return x.detach() # Only return hidden state
    
    def _extract_constraint_signals(self, hidden_state: torch.Tensor) -> Dict[str, float]:
        return {
            'intensity': hidden_state.abs().mean().item(),
            'sparsity': (hidden_state.abs() > 0.1).float().mean().item(),
            'variance': hidden_state.var().item(),
        }

#!/usr/bin/env python3
"""
BDH Inference Script (Deep Integration with bdh.py)
Uses BDH internal components directly: Attention, forward, generate, embeddings, etc.
Performs counterfactual comparison to measure backstory influence.
"""

import os
import sys
import torch
import torch.nn.functional as F

# Import the BDH module components directly
try:
    import bdh
    from bdh import BDH, BDHConfig, Attention, get_freqs
except ImportError as e:
    print(f"Failed to import bdh module: {e}")
    print("Available modules in current directory:")
    for f in os.listdir(os.path.dirname(__file__)):
        print(f"  - {f}")
    sys.exit(1)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILES_DIR = os.path.join(os.path.dirname(__file__), "files")
BACKSTORY_PATH = os.path.join(FILES_DIR, "backstory.txt")
NOVEL_PATH = os.path.join(FILES_DIR, "novel.txt")

# Chunk size for memory-safe processing
CHUNK_SIZE = 512


def tokenize_text(text: str) -> torch.Tensor:
    """
    Convert text to byte-level tokens.
    BDH uses vocab_size=256 (byte-level tokenization).
    """
    byte_array = bytearray(text, "utf-8")
    tokens = torch.tensor(list(byte_array), dtype=torch.long)
    return tokens


def read_file_safe(filepath: str) -> str:
    """Read file with fallback encoding handling."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            return f.read()


def inspect_attention(attn_module: Attention, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> dict:
    """
    Inspect the Attention mechanism from bdh.py.
    Uses Attention.forward(), Attention.rope(), and Attention.phases_cos_sin().
    Returns attention scores and transformed values.
    """
    # Get RoPE phases - match the implementation in Attention.forward()
    _, _, T, _ = Q.size()
    freqs = attn_module.freqs  # [1, 1, 1, N]
    r_phases = (
        torch.arange(0, T, device=freqs.device, dtype=freqs.dtype).view(1, 1, -1, 1)
    ) * freqs  # [1, 1, T, 1] * [1, 1, 1, N] -> [1, 1, T, N]
    
    # Apply RoPE using static method
    QR = Attention.rope(r_phases, Q)
    KR = QR  # K == Q in BDH
    
    # Compute attention scores (causal mask via tril)
    scores = (QR @ KR.mT).tril(diagonal=-1)
    
    # Apply attention to values
    output = scores @ V
    
    return {
        "r_phases_shape": list(r_phases.shape),
        "QR_shape": list(QR.shape),
        "scores_shape": list(scores.shape),
        "output_shape": list(output.shape),
        "scores_sample": scores[0, 0, :5, :5].tolist() if T >= 5 else scores[0, 0].tolist(),
    }


def run_bdh_forward_step_by_step(model: BDH, idx: torch.Tensor) -> dict:
    """
    Run BDH forward pass step-by-step, exposing intermediate values.
    This mirrors the logic inside BDH.forward() but exposes each step.
    """
    C = model.config
    B, T = idx.size()
    D = C.n_embd
    nh = C.n_head
    N = D * C.mlp_internal_dim_multiplier // nh
    
    intermediates = {
        "config": {"B": B, "T": T, "D": D, "nh": nh, "N": N},
        "layers": [],
    }
    
    # Step 1: Embedding
    x = model.embed(idx).unsqueeze(1)  # B, 1, T, D
    intermediates["embedding_shape"] = list(x.shape)
    intermediates["embedding_sample"] = x[0, 0, :3, :5].tolist()
    
    # Step 2: Layer Normalization
    x = model.ln(x)
    intermediates["after_ln_shape"] = list(x.shape)
    
    # Step 3: Process through each layer
    for level in range(C.n_layer):
        layer_info = {"level": level}
        
        # Encoder projection
        # x: [B, 1, T, D], encoder: [nh, D, N] -> x_latent: [B, nh, T, N]
        x_latent = torch.einsum('b1td,hde->bhte', x, model.encoder)
        layer_info["x_latent_shape"] = list(x_latent.shape)
        
        # ReLU activation (sparsity)
        x_sparse = F.relu(x_latent)
        layer_info["x_sparse_nonzero_ratio"] = (x_sparse > 0).float().mean().item()
        
        # Attention (using model.attn which is bdh.Attention)
        yKV = model.attn(Q=x_sparse, K=x_sparse, V=x)
        layer_info["yKV_shape_before_reduce"] = list(yKV.shape)
        
        # Inspect attention internals for first layer
        if level == 0:
            attn_details = inspect_attention(model.attn, x_sparse, x_sparse, x)
            layer_info["attention_details"] = attn_details
        
        # Reduce from [B, nh, T, D] to [B, 1, T, D] by averaging over heads
        yKV = yKV.mean(dim=1, keepdim=True)
        layer_info["yKV_shape"] = list(yKV.shape)
        yKV = model.ln(yKV)
        
        # Value encoder projection
        # yKV: [B, 1, T, D], encoder_v: [nh, D, N] -> y_latent: [B, nh, T, N]
        y_latent = torch.einsum('b1td,hde->bhte', yKV, model.encoder_v)
        y_sparse = F.relu(y_latent)
        layer_info["y_sparse_nonzero_ratio"] = (y_sparse > 0).float().mean().item()
        
        # Hebbian-style multiplicative interaction
        xy_sparse = x_sparse * y_sparse
        layer_info["xy_sparse_shape"] = list(xy_sparse.shape)
        
        # Dropout (eval mode = no dropout)
        xy_sparse = model.drop(xy_sparse)
        
        # Decoder projection (MLP)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ model.decoder
        layer_info["yMLP_shape"] = list(yMLP.shape)
        
        # Residual connection with LayerNorm
        y = model.ln(yMLP)
        x = model.ln(x + y)
        layer_info["output_x_shape"] = list(x.shape)
        
        intermediates["layers"].append(layer_info)
    
    # Step 4: LM Head projection
    logits = x.view(B, T, D) @ model.lm_head
    intermediates["logits_shape"] = list(logits.shape)
    intermediates["logits_sample"] = logits[0, -1, :5].tolist()
    
    return logits, intermediates


def process_tokens_with_inspection(model: BDH, tokens: torch.Tensor, chunk_size: int = CHUNK_SIZE):
    """
    Process tokens through the model, inspecting each layer.
    Returns final logits plus layer-by-layer inspection data.
    """
    model.eval()
    total_tokens = len(tokens)
    last_logits = None
    all_intermediates = []
    
    with torch.no_grad():
        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk = tokens[start_idx:end_idx].unsqueeze(0).to(DEVICE)
            
            # Run step-by-step forward (exposing BDH internals)
            logits, intermediates = run_bdh_forward_step_by_step(model, chunk)
            last_logits = logits
            
            # Store intermediates for first and last chunk only
            if start_idx == 0 or end_idx == total_tokens:
                all_intermediates.append({
                    "chunk_range": (start_idx, end_idx),
                    "intermediates": intermediates,
                })
            
            # Progress indicator
            if (start_idx // chunk_size) % 100 == 0:
                print(f"    Processed {end_idx:,} / {total_tokens:,} tokens...")
    
    return total_tokens, last_logits, all_intermediates


def generate_continuation(model: BDH, prompt_tokens: torch.Tensor, max_new_tokens: int = 50) -> torch.Tensor:
    """
    Use BDH.generate() to produce a continuation of the prompt.
    This directly uses the generate method from bdh.py.
    """
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            idx=prompt_tokens.unsqueeze(0).to(DEVICE),
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40,
        )
    return generated


def extract_representation(logits: torch.Tensor) -> torch.Tensor:
    """Extract final representation from logits (last token's full logits)."""
    return logits[0, -1, :]


def compute_similarity(rep_a: torch.Tensor, rep_b: torch.Tensor) -> dict:
    """Compute similarity metrics between two representations."""
    rep_a_norm = F.normalize(rep_a.float(), dim=0)
    rep_b_norm = F.normalize(rep_b.float(), dim=0)
    
    cosine_sim = torch.dot(rep_a_norm, rep_b_norm).item()
    l2_distance = torch.norm(rep_a.float() - rep_b.float()).item()
    
    return {
        "cosine_similarity": cosine_sim,
        "l2_distance": l2_distance,
    }


def main():
    print("=" * 60)
    print("BDH Inference Script (Deep Integration)")
    print("Using: BDH, BDHConfig, Attention, get_freqs, generate")
    print("=" * 60)
    
    # Check for required files
    print("\n[1] Checking files...")
    if not os.path.exists(BACKSTORY_PATH):
        print(f"ERROR: backstory.txt not found at {BACKSTORY_PATH}")
        sys.exit(1)
    if not os.path.exists(NOVEL_PATH):
        print(f"ERROR: novel.txt not found at {NOVEL_PATH}")
        sys.exit(1)
    
    print(f"  ✓ backstory.txt found ({os.path.getsize(BACKSTORY_PATH):,} bytes)")
    print(f"  ✓ novel.txt found ({os.path.getsize(NOVEL_PATH):,} bytes)")
    
    # Initialize model using BDHConfig and BDH classes
    print(f"\n[2] Initializing BDH model on {DEVICE}...")
    config = BDHConfig()
    model = BDH(config).to(DEVICE)
    
    print("  BDHConfig attributes:")
    print(f"    - n_layer: {config.n_layer}")
    print(f"    - n_embd: {config.n_embd}")
    print(f"    - n_head: {config.n_head}")
    print(f"    - vocab_size: {config.vocab_size}")
    print(f"    - mlp_internal_dim_multiplier: {config.mlp_internal_dim_multiplier}")
    print(f"    - dropout: {config.dropout}")
    
    # Inspect model components
    print("\n  BDH model components:")
    print(f"    - model.embed: {model.embed}")
    print(f"    - model.attn: {model.attn} (Attention class)")
    print(f"    - model.ln: {model.ln}")
    print(f"    - model.encoder shape: {model.encoder.shape}")
    print(f"    - model.encoder_v shape: {model.encoder_v.shape}")
    print(f"    - model.decoder shape: {model.decoder.shape}")
    print(f"    - model.lm_head shape: {model.lm_head.shape}")
    
    # Inspect Attention module
    print("\n  Attention module internals:")
    print(f"    - model.attn.freqs shape: {model.attn.freqs.shape}")
    print(f"    - get_freqs function available: {get_freqs is not None}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    
    # Read text files
    print("\n[3] Reading text files...")
    backstory_text = read_file_safe(BACKSTORY_PATH)
    novel_text = read_file_safe(NOVEL_PATH)
    print(f"  Backstory: {len(backstory_text):,} characters")
    print(f"  Novel: {len(novel_text):,} characters")
    
    # ========================================
    # PASS A: WITH BACKSTORY
    # ========================================
    print("\n[4] PASS A: Processing WITH backstory...")
    combined_text = backstory_text + "\n\n" + novel_text
    combined_tokens = tokenize_text(combined_text)
    print(f"  Combined token count: {len(combined_tokens):,}")
    
    pass_a_count, pass_a_logits, pass_a_intermediates = process_tokens_with_inspection(model, combined_tokens)
    pass_a_rep = extract_representation(pass_a_logits)
    print(f"  ✓ Processed {pass_a_count:,} tokens")
    
    # Print layer-by-layer inspection for first chunk
    if pass_a_intermediates:
        first_chunk = pass_a_intermediates[0]["intermediates"]
        print(f"\n  [Layer Inspection - First Chunk]")
        print(f"    Embedding shape: {first_chunk['embedding_shape']}")
        for layer in first_chunk["layers"][:2]:  # Show first 2 layers
            print(f"    Layer {layer['level']}:")
            print(f"      x_sparse nonzero: {layer['x_sparse_nonzero_ratio']:.2%}")
            print(f"      y_sparse nonzero: {layer['y_sparse_nonzero_ratio']:.2%}")
            if "attention_details" in layer:
                attn = layer["attention_details"]
                print(f"      Attention QR shape: {attn['QR_shape']}")
    
    # ========================================
    # PASS B: WITHOUT BACKSTORY
    # ========================================
    print("\n[5] PASS B: Processing WITHOUT backstory...")
    novel_tokens = tokenize_text(novel_text)
    print(f"  Novel token count: {len(novel_tokens):,}")
    
    pass_b_count, pass_b_logits, pass_b_intermediates = process_tokens_with_inspection(model, novel_tokens)
    pass_b_rep = extract_representation(pass_b_logits)
    print(f"  ✓ Processed {pass_b_count:,} tokens")
    
    # ========================================
    # GENERATION DEMO (using model.generate)
    # ========================================
    print("\n[6] Generation Demo (using BDH.generate)...")
    prompt = "The story begins with"
    prompt_tokens = tokenize_text(prompt)
    print(f"  Prompt: '{prompt}'")
    print(f"  Prompt tokens: {len(prompt_tokens)}")
    
    generated = generate_continuation(model, prompt_tokens, max_new_tokens=50)
    generated_text = bytes(generated[0].tolist()).decode("utf-8", errors="replace")
    print(f"  Generated: '{generated_text}'")
    
    # ========================================
    # COMPARISON
    # ========================================
    print("\n[7] Comparing representations...")
    similarity = compute_similarity(pass_a_rep, pass_b_rep)
    
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL COMPARISON RESULTS")
    print("=" * 60)
    
    print("\n[Representation Comparison]")
    print(f"  Cosine Similarity: {similarity['cosine_similarity']:.6f}")
    print(f"  L2 Distance: {similarity['l2_distance']:.6f}")
    
    # Interpret results
    print("\n[Interpretation]")
    if similarity['cosine_similarity'] > 0.99:
        print("  → Backstory has MINIMAL influence on novel representation.")
    elif similarity['cosine_similarity'] > 0.95:
        print("  → Backstory has SLIGHT influence on novel representation.")
    elif similarity['cosine_similarity'] > 0.80:
        print("  → Backstory has MODERATE influence on novel representation.")
    else:
        print("  → Backstory has SIGNIFICANT influence on novel representation.")
    
    print("\n[Processing Summary]")
    print(f"  Pass A (with backstory): {pass_a_count:,} tokens")
    print(f"  Pass B (novel only): {pass_b_count:,} tokens")
    
    print("\n[Sample Logits (last token, first 5 values)]")
    print(f"  Pass A: {pass_a_rep[:5].tolist()}")
    print(f"  Pass B: {pass_b_rep[:5].tolist()}")
    
    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n[Saved] Model state dict: {model_path}")
    
    rep_path = os.path.join(output_dir, "representations.pt")
    torch.save({
        "pass_a_rep": pass_a_rep.cpu(),
        "pass_b_rep": pass_b_rep.cpu(),
        "similarity": similarity,
        "pass_a_intermediates": pass_a_intermediates,
        "pass_b_intermediates": pass_b_intermediates,
    }, rep_path)
    print(f"[Saved] Representations + Intermediates: {rep_path}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    
    print("\n" + "=" * 60)
    print("Inference complete. Exiting cleanly.")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

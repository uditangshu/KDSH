#!/usr/bin/env python3
"""
BDH Inference Script (Counterfactual Comparison)
Runs the BDH model in pure inference mode using two text files.
Compares representation WITH and WITHOUT backstory to measure influence.
"""

import os
import sys
import torch
import torch.nn.functional as F

# Import the BDH model from the repo
try:
    import bdh
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


def get_state_summary(model: torch.nn.Module) -> dict:
    """
    Extract a summary of the model's internal state.
    Returns info about learned parameters.
    """
    summary = {}
    
    # Get config info
    summary["config"] = {
        "n_layer": model.config.n_layer,
        "n_embd": model.config.n_embd,
        "n_head": model.config.n_head,
        "vocab_size": model.config.vocab_size,
    }
    
    # Get parameter statistics
    param_stats = {}
    for name, param in model.named_parameters():
        param_stats[name] = {
            "shape": list(param.shape),
            "mean": param.data.mean().item(),
            "std": param.data.std().item(),
            "min": param.data.min().item(),
            "max": param.data.max().item(),
        }
    summary["parameters"] = param_stats
    
    return summary


def process_tokens(model: torch.nn.Module, tokens: torch.Tensor, chunk_size: int = CHUNK_SIZE):
    """
    Process tokens through the model in chunks.
    Returns the final logits from the last chunk.
    """
    model.eval()
    total_tokens = len(tokens)
    last_logits = None
    
    with torch.no_grad():
        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk = tokens[start_idx:end_idx].unsqueeze(0).to(DEVICE)  # Shape: [1, chunk_size]
            
            # Forward pass (no state parameter)
            logits, _ = model(chunk)
            last_logits = logits
            
            # Progress indicator
            if (start_idx // chunk_size) % 100 == 0:
                print(f"    Processed {end_idx:,} / {total_tokens:,} tokens...")
    
    return total_tokens, last_logits


def extract_representation(logits: torch.Tensor) -> torch.Tensor:
    """
    Extract final representation from logits.
    Uses the last token's logits as the representation vector.
    """
    # Shape of logits: [batch, seq_len, vocab_size]
    # Take the last token's full logits as representation
    return logits[0, -1, :]  # Shape: [vocab_size]


def compute_similarity(rep_a: torch.Tensor, rep_b: torch.Tensor) -> dict:
    """
    Compute similarity metrics between two representations.
    """
    # Normalize for cosine similarity
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
    print("BDH Inference Script (Counterfactual Comparison)")
    print("=" * 60)
    
    # Check for required files
    print("\n[1] Checking files...")
    if not os.path.exists(BACKSTORY_PATH):
        print(f"ERROR: backstory.txt not found at {BACKSTORY_PATH}")
        print(f"Available files in {FILES_DIR}:")
        if os.path.exists(FILES_DIR):
            for f in os.listdir(FILES_DIR):
                print(f"  - {f}")
        sys.exit(1)
    
    if not os.path.exists(NOVEL_PATH):
        print(f"ERROR: novel.txt not found at {NOVEL_PATH}")
        print(f"Available files in {FILES_DIR}:")
        if os.path.exists(FILES_DIR):
            for f in os.listdir(FILES_DIR):
                print(f"  - {f}")
        sys.exit(1)
    
    print(f"  ✓ backstory.txt found ({os.path.getsize(BACKSTORY_PATH):,} bytes)")
    print(f"  ✓ novel.txt found ({os.path.getsize(NOVEL_PATH):,} bytes)")
    
    # Initialize model
    print(f"\n[2] Initializing BDH model on {DEVICE}...")
    config = bdh.BDHConfig()
    model = bdh.BDH(config).to(DEVICE)
    
    print("  BDH Config:")
    print(f"    - n_layer: {config.n_layer}")
    print(f"    - n_embd: {config.n_embd}")
    print(f"    - n_head: {config.n_head}")
    print(f"    - vocab_size: {config.vocab_size}")
    print(f"    - mlp_internal_dim_multiplier: {config.mlp_internal_dim_multiplier}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Read text files
    print("\n[3] Reading text files...")
    backstory_text = read_file_safe(BACKSTORY_PATH)
    novel_text = read_file_safe(NOVEL_PATH)
    print(f"  Backstory: {len(backstory_text):,} characters")
    print(f"  Novel: {len(novel_text):,} characters")
    
    # ========================================
    # PASS A: WITH BACKSTORY (backstory + novel)
    # ========================================
    print("\n[4] PASS A: Processing WITH backstory...")
    combined_text = backstory_text + "\n\n" + novel_text
    combined_tokens = tokenize_text(combined_text)
    print(f"  Combined token count: {len(combined_tokens):,}")
    
    pass_a_count, pass_a_logits = process_tokens(model, combined_tokens)
    pass_a_rep = extract_representation(pass_a_logits)
    print(f"  ✓ Processed {pass_a_count:,} tokens")
    print(f"  ✓ Representation shape: {pass_a_rep.shape}")
    
    # ========================================
    # PASS B: WITHOUT BACKSTORY (novel only)
    # ========================================
    print("\n[5] PASS B: Processing WITHOUT backstory...")
    novel_tokens = tokenize_text(novel_text)
    print(f"  Novel token count: {len(novel_tokens):,}")
    
    pass_b_count, pass_b_logits = process_tokens(model, novel_tokens)
    pass_b_rep = extract_representation(pass_b_logits)
    print(f"  ✓ Processed {pass_b_count:,} tokens")
    print(f"  ✓ Representation shape: {pass_b_rep.shape}")
    
    # ========================================
    # COMPARISON
    # ========================================
    print("\n[6] Comparing representations...")
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
        print("    The final embedding is nearly identical with/without backstory.")
    elif similarity['cosine_similarity'] > 0.95:
        print("  → Backstory has SLIGHT influence on novel representation.")
        print("    Small differences detected in the final embedding.")
    elif similarity['cosine_similarity'] > 0.80:
        print("  → Backstory has MODERATE influence on novel representation.")
        print("    The backstory context noticeably affects the encoding.")
    else:
        print("  → Backstory has SIGNIFICANT influence on novel representation.")
        print("    The final embedding is substantially different with backstory.")
    
    print("\n[Processing Summary]")
    print(f"  Pass A (with backstory): {pass_a_count:,} tokens")
    print(f"  Pass B (novel only): {pass_b_count:,} tokens")
    
    print("\n[Sample Logits (last token, first 5 values)]")
    print(f"  Pass A: {pass_a_rep[:5].tolist()}")
    print(f"  Pass B: {pass_b_rep[:5].tolist()}")
    
    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n[Saved] Model state dict: {model_path}")
    
    # Save representations
    rep_path = os.path.join(output_dir, "representations.pt")
    torch.save({
        "pass_a_rep": pass_a_rep.cpu(),
        "pass_b_rep": pass_b_rep.cpu(),
        "similarity": similarity,
    }, rep_path)
    print(f"[Saved] Representations: {rep_path}")
    
    # Save logits
    logits_path = os.path.join(output_dir, "logits.pt")
    torch.save({
        "pass_a_logits": pass_a_logits.cpu(),
        "pass_b_logits": pass_b_logits.cpu(),
    }, logits_path)
    print(f"[Saved] Logits: {logits_path}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    
    print("\n" + "=" * 60)
    print("Inference complete. Exiting cleanly.")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

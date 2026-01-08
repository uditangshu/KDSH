#!/usr/bin/env python3
"""
BDH Inference Script
Runs the BDH model in pure inference mode using two text files.
Processes backstory.txt first, then novel.txt with persistent state.
"""

import os
import sys
import torch

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
CHUNK_SIZE = 256


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


def feed_tokens_sequential(model: torch.nn.Module, tokens: torch.Tensor, state, chunk_size: int = CHUNK_SIZE):
    """
    Feed tokens sequentially to the model with persistent state.
    Returns (total_tokens_processed, final_logits, updated_state).
    """
    model.eval()
    total_tokens = len(tokens)
    last_logits = None
    
    with torch.no_grad():
        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk = tokens[start_idx:end_idx].unsqueeze(0).to(DEVICE)  # Shape: [1, chunk_size]
            
            # Forward pass with state
            logits, state = model(chunk, state=state)
            last_logits = logits.cpu()
            
            # Progress indicator
            if (start_idx // chunk_size) % 100 == 0:
                print(f"    Processed {end_idx:,} / {total_tokens:,} tokens...")
    
    return total_tokens, last_logits, state


def main():
    print("=" * 60)
    print("BDH Inference Script")
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
    
    # Print available objects from bdh module
    print("\n  Available objects in bdh module:")
    for obj in dir(bdh):
        if not obj.startswith("_"):
            print(f"    - {obj}")
    
    total_tokens_processed = 0
    
    # Initialize state to None before processing backstory
    state = None
    
    # Read and process backstory.txt
    print("\n[3] Processing backstory.txt...")
    backstory_text = read_file_safe(BACKSTORY_PATH)
    print(f"  Text length: {len(backstory_text):,} characters")
    
    backstory_tokens = tokenize_text(backstory_text)
    print(f"  Token count: {len(backstory_tokens):,}")
    
    backstory_count, backstory_logits, state = feed_tokens_sequential(model, backstory_tokens, state)
    total_tokens_processed += backstory_count
    print(f"  ✓ Processed {backstory_count:,} tokens")
    
    # Process novel.txt WITHOUT resetting state
    # The same state continues from backstory
    print("\n[4] Processing novel.txt (WITHOUT resetting state)...")
    novel_text = read_file_safe(NOVEL_PATH)
    print(f"  Text length: {len(novel_text):,} characters")
    
    novel_tokens = tokenize_text(novel_text)
    print(f"  Token count: {len(novel_tokens):,}")
    
    novel_count, novel_logits, state = feed_tokens_sequential(model, novel_tokens, state)
    total_tokens_processed += novel_count
    print(f"  ✓ Processed {novel_count:,} tokens")
    
    # Print final state summary
    print("\n" + "=" * 60)
    print("FINAL STATE SUMMARY")
    print("=" * 60)
    
    state_summary = get_state_summary(model)
    
    print("\n[Model Configuration]")
    for key, value in state_summary["config"].items():
        print(f"  {key}: {value}")
    
    print("\n[Parameter Statistics]")
    for name, stats in state_summary["parameters"].items():
        print(f"  {name}:")
        print(f"    shape: {stats['shape']}")
        print(f"    mean: {stats['mean']:.6f}, std: {stats['std']:.6f}")
        print(f"    range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    
    print("\n[Processing Summary]")
    print(f"  Backstory tokens: {backstory_count:,}")
    print(f"  Novel tokens: {novel_count:,}")
    print(f"  Total tokens processed: {total_tokens_processed:,}")
    
    print("\n[Final State]")
    if state is not None:
        print(f"  State type: {type(state)}")
        if isinstance(state, torch.Tensor):
            print(f"  State shape: {state.shape}")
            print(f"  State sample (first 5 values): {state.flatten()[:5].tolist()}")
    else:
        print("  State: None")
    
    print("\n[Last Logits Sample (first 5 values)]")
    if novel_logits is not None:
        print(f"  {novel_logits[0, -1, :5].tolist()}")
    
    print("\n" + "=" * 60)
    print("Inference complete. Exiting cleanly.")
    print("=" * 60)
    
    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n[Saved] Model state dict: {model_path}")
    
    # Save final state (if it exists)
    if state is not None:
        state_path = os.path.join(output_dir, "state.pt")
        torch.save(state, state_path)
        print(f"[Saved] Final state: {state_path}")
    
    # Save final logits
    if novel_logits is not None:
        logits_path = os.path.join(output_dir, "novel_logits.pt")
        torch.save(novel_logits, logits_path)
        print(f"[Saved] Novel logits: {logits_path}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

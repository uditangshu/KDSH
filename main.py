#!/usr/bin/env python3
"""
BDH Advanced Multi-Level Inference System
Modular architecture with Pathway streaming support.

Usage:
    python inference.py                                    # Batch mode (default)
    python inference.py --mode stream                      # Streaming mode with Pathway
    python inference.py --backstory FILE --story FILE      # Custom files
"""

import os
import sys
import argparse
import torch

try:
    import bdh
    from bdh import BDH, BDHConfig
except ImportError as e:
    print(f"Failed to import bdh module: {e}")
    sys.exit(1)

from inference import AdvancedBDHInference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='BDH Advanced Inference System')
    parser.add_argument('--backstory', type=str, help='Path to backstory file')
    parser.add_argument('--story', type=str, help='Path to story file')
    parser.add_argument('--mode', type=str, default='batch', choices=['batch', 'stream'],
                       help='Inference mode: batch or stream (Pathway)')
    
    args = parser.parse_args()
    
    # Load files
    if args.backstory and args.story:
        with open(args.backstory, 'r', encoding='utf-8') as f:
            backstory_text = f.read()
        with open(args.story, 'r', encoding='utf-8') as f:
            story_text = f.read()
    else:
        files_dir = os.path.join(os.path.dirname(__file__), "files")
        backstory_path = os.path.join(files_dir, "backstory1.txt")
        novel_path = os.path.join(files_dir, "novel1.txt")
        
        with open(backstory_path, 'r', encoding='utf-8') as f:
            backstory_text = f.read()
        with open(novel_path, 'r', encoding='utf-8') as f:
            story_text = f.read()
    
    # Initialize model
    print(f"Initializing BDH model on {DEVICE}...")
    config = BDHConfig()
    model = BDH(config).to(DEVICE)
    
    # Run inference
    use_streaming = args.mode == 'stream'
    inference = AdvancedBDHInference(model, config, DEVICE)
    
    print(f"Running inference in {'STREAMING' if use_streaming else 'BATCH'} mode...")
    results = inference.predict(backstory_text, story_text, use_streaming=use_streaming)
    
    # Print results
    print("\n" + "=" * 60)
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
    print(f"  Implausible Jumps: {results['level4'].get('implausible_count', len(results['level4'].get('implausible_jumps', [])))}")
    
    # Pathway streaming report
    if 'pathway_report' in results:
        print(f"\n{'='*60}")
        print("PATHWAY STREAMING REPORT")
        report = results['pathway_report']
        print(f"  Constraint Alerts: {report['constraint_alerts']['total_alerts']}")
        print(f"  Final Rolling Similarity: {report['final_rolling_similarity']:.3f}")
    
    print(f"\n{'='*60}")
    print(f"Extracted Constraints: {len(results['constraints'])}")
    for i, constraint in enumerate(results['constraints'][:5], 1):
        print(f"  {i}. [{constraint.category}] {constraint.text}")
    
    print("\n" + "=" * 60)
    print("Inference complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()

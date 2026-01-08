# Baby Dragon Hatchling

## **Bridging the Gap Between Transformers and the Brain**

**Baby Dragon Hatchling (BDH)** is a biologically inspired large language model architecture that connects principles of deep learning with the foundations of neuroscience. Developed by researchers at [Pathway](https://pathway.com), BDH provides a theoretical and practical framework for understanding the emergence of reasoning and generalization in artificial systems.

This repository contains the official implementation from the paper:
> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).


## Overview

BDH represents a **scale-free, locally interacting network of neurons** capable of intrinsic reasoning dynamics. BDH scales like a Transformer on performance benchmarks—yet retains full interpretability and theoretical grounding in the fine-grained dynamics of neuron interactions.

**Key properties:**

- **Scale-free network topology** mimicking biological connectivity
- **Locally interacting neuron particles** with excitatory/inhibitory dynamics
- **Hebbian working memory** based on synaptic plasticity, displaying monosemanticity
- **GPU-friendly state-space formulation** for efficient implementation
- **Interpretable activations** that are sparse and positive

BDH formalizes a bridge between **neural computation and machine-based language understanding**. It shows how **macro reasoning behavior** in large AI models emerges from **micro-level neuron dynamics**, guided by principles of graph theory and local computation.

Empirically, BDH matches **GPT-2–scale Transformers** across language and translation tasks at equivalent parameter scales (10M–1B).


***

## Architecture

<img src="figs/architecture.png" width="600"/>

***

## Relation to Transformers

<img src="figs/vocab.png" width="600"/>

BDH and the Transformer share attention-inspired computation; however, BDH’s graph-based architecture makes its attention **emerge naturally from neuron-level interactions**, reflecting attention as seen in biological systems.

***

## Scaling Laws

<img src="figs/bdh_scaling.png" width="600"/>

BDH follows **Transformer-like scaling laws**, maintaining parameter efficiency while achieving interpretability at any scale.

***

## Installation and Training

```bash
# install dependencies
pip install -r requirements.txt

# train BDH on a toy dataset
python train.py
```

---

# 🏆 Kharagpur Data Science Hackathon 2026 - Implementation Guide

## Problem Statement Summary

**Challenge**: Determine if a hypothetical backstory for a central character is **consistent** with a complete long-form narrative (100k+ words novel).

**Task**: Binary classification
- **Consistent (1)**: Backstory respects key constraints established throughout the narrative
- **Contradict (0)**: Backstory conflicts with later events, character development, or causal pathways

**Key Requirements**:
- **Consistency over time**: Check if backstory fits with how characters/events develop later
- **Causal reasoning**: Determine if later events make sense given earlier conditions
- **Respect for narrative constraints**: Detect mismatches beyond direct contradictions
- **Evidence-based decisions**: Support conclusions with signals from multiple parts of text

**Dataset**: `train.csv` (labeled) and `test.csv` (unlabeled) with:
- `id`: Example identifier
- `book_name`: Novel title (e.g., "In Search of the Castaways", "The Count of Monte Cristo")
- `char`: Character name
- `caption`: Backstory title
- `content`: Backstory text
- `label`: "consistent" or "contradict" (training only)

---

## 🎯 Competition Tracks

### **Track B: BDH-Driven Continuous Narrative Reasoning** ⭐ (Our Focus)

**Requirements**:
- Must incorporate BDH architecture or principles
- Can use open-source BDH implementation
- Can pretrain/adapt BDH on task-relevant signals
- Can use BDH for representations → classification head
- Can implement BDH-inspired components (persistent state, sparse updates, incremental belief)

**Evaluation**:
1. **Accuracy** on core classification task
2. **Pretraining/Representation Learning** using BDH
3. **Clarity** in how BDH mechanisms influence representations/decisions
4. **Evidence Rationale** (optional but encouraged)

**Why Track B is Better**:
- Already have BDH implementation
- Novel approach (fewer competitors)
- Aligns with research interests
- Allows focus on architecture innovation over pure accuracy

---

## 📋 Complete Implementation Roadmap

### **Phase 1: Data Preparation & Understanding** (Days 1-2)

#### Step 1.1: Dataset Analysis
- [ ] Load and analyze `train.csv` and `test.csv`
- [ ] Identify book files (check `files/` directory)
- [ ] Map book names to actual novel text files
- [ ] Calculate statistics:
  - Average novel length
  - Character distribution
  - Label distribution (consistent vs contradict)
  - Backstory length distribution

#### Step 1.2: Data Pipeline Setup
- [ ] Create `data_loader.py`:
  ```python
  - load_novel(book_name) → full text
  - load_backstory(row) → backstory text
  - create_example(novel, backstory) → formatted input
  ```

#### Step 1.3: Text Preprocessing
- [ ] Tokenization strategy (byte-level, 256 vocab as BDH default)
- [ ] Chunking strategy for long novels (512 tokens per chunk)
- [ ] Create train/validation split from `train.csv`

**Key Insights from Current Data**:
- Novels: "In Search of the Castaways" (~826K chars), "The Count of Monte Cristo"
- Training examples: Multiple per book with different characters
- Need to match book names to actual text files

---

### **Phase 2: BDH Model Adaptation** (Days 2-4)

#### Step 2.1: Baseline BDH Configuration
- [ ] Start with default BDH config:
  ```python
  n_layer=6
  n_embd=256
  n_head=4
  vocab_size=256
  ```
- [ ] Test BDH forward pass on sample novel chunks
- [ ] Verify attention patterns and sparse activations

#### Step 2.2: BDH for Narrative Understanding
**Key Modification**: Process novel + backstory together

**Approach A: Sequential Processing**
```
1. Process backstory → get backstory representation
2. Process novel in chunks → get novel representations
3. Compare/merge representations
4. Classification head
```

**Approach B: Concatenated Processing** (Better for causal reasoning)
```
1. Concatenate: backstory + separator + novel
2. Process entire sequence through BDH
3. Extract final representation
4. Classification head
```

**Approach C: Two-Pass with Comparison** (Current implementation)
```
1. Pass A: backstory + novel → representation_A
2. Pass B: novel only → representation_B
3. Compute similarity/difference
4. Classification based on divergence
```

#### Step 2.3: Classification Head Design
- [ ] Linear classifier: `BDH_representation → [0, 1]`
- [ ] Multi-layer classifier for better expressivity
- [ ] Consider attention pooling over sequence

---

### **Phase 3: Training Strategy** (Days 4-7)

#### Step 3.1: Pretraining on Narrative Corpus
**Why**: Help BDH understand narrative structure before fine-tuning

**Strategy**:
- [ ] Pretrain on all novel texts (unsupervised language modeling)
- [ ] Use masked language modeling or next-token prediction
- [ ] Train for 5K-10K iterations
- [ ] Save pretrained checkpoint

#### Step 3.2: Fine-tuning for Consistency Classification
**Data Format**:
```
Input: [BACKSTORY_TOKEN] ...backstory... [SEP] ...novel...
Target: 1 (consistent) or 0 (contradict)
```

**Training Loop**:
- [ ] Load pretrained BDH weights
- [ ] Freeze encoder layers OR use low learning rate
- [ ] Train classification head with higher learning rate
- [ ] Use binary cross-entropy loss
- [ ] Track validation accuracy

#### Step 3.3: Training Optimizations
- [ ] **Gradient Accumulation**: Handle long sequences (batch_size=1, accumulate over 8 steps)
- [ ] **Mixed Precision**: Use bfloat16/fp16 for speed
- [ ] **Learning Rate Schedule**: Cosine annealing or warmup + decay
- [ ] **Early Stopping**: Stop when validation loss plateaus
- [ ] **Checkpointing**: Save best model based on validation F1

**Hyperparameters to Tune**:
```python
LEARNING_RATE = 1e-4  # Start lower for fine-tuning
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1  # With gradient accumulation
ACCUMULATION_STEPS = 8
MAX_EPOCHS = 10
```

---

### **Phase 4: Long-Context Handling** (Days 5-8)

#### Step 4.1: Chunking Strategy
**Problem**: Novels are 100k+ words, BDH processes 512 tokens at a time

**Solution A: Hierarchical Chunking**
- [ ] Divide novel into overlapping chunks (512 tokens, 256 overlap)
- [ ] Process each chunk through BDH
- [ ] Pool chunk representations (mean, max, attention-weighted)

**Solution B: Sliding Window with State**
- [ ] Process chunks sequentially
- [ ] Maintain running state between chunks
- [ ] Use BDH's natural state-space formulation

**Solution C: Key Passage Retrieval** (Hybrid)
- [ ] Use keyword/character matching to find relevant passages
- [ ] Process only relevant chunks + full backstory
- [ ] Reduces computation while maintaining context

#### Step 4.2: Evidence Extraction
**For Track B (optional but impressive)**:
- [ ] Identify novel passages that support/contradict backstory
- [ ] Use attention scores to highlight relevant tokens
- [ ] Extract top-k passages with highest attention
- [ ] Generate rationale from extracted evidence

---

### **Phase 5: Model Improvements** (Days 8-12)

#### Step 5.1: Architecture Enhancements

**A. Larger Model** (if compute allows):
```python
BDHConfig(
    n_layer=12,  # More depth
    n_embd=512,  # Larger embeddings
    n_head=8,
    mlp_internal_dim_multiplier=128
)
```

**B. Multi-Task Learning**:
- [ ] Train on consistency + related tasks:
  - Character relation prediction
  - Event ordering
  - Cause-effect prediction

**C. Ensemble Approach**:
- [ ] Train 3-5 BDH models with different seeds/configs
- [ ] Average predictions for robustness

#### Step 5.2: Representation Learning Improvements

**A. Contrastive Learning**:
```python
# Positive: consistent backstory + novel
# Negative: contradict backstory + novel
# Maximize similarity for positives, minimize for negatives
```

**B. Feature Engineering**:
- [ ] Extract BDH intermediate activations (layer-wise)
- [ ] Compute statistics: sparsity, attention entropy, etc.
- [ ] Concatenate with classification features

---

### **Phase 6: Evaluation & Validation** (Days 10-13)

#### Step 6.1: Validation Strategy
- [ ] Cross-validation on training set (if enough examples)
- [ ] Hold-out validation (80/20 split)
- [ ] Per-book validation (train on one book, test on another)

#### Step 6.2: Metrics
- [ ] **Accuracy**: Overall classification correctness
- [ ] **F1-Score**: Balance precision and recall
- [ ] **ROC-AUC**: Classification quality
- [ ] **Per-Class Accuracy**: Consistent vs Contradict separately

#### Step 6.3: Error Analysis
- [ ] Identify failure cases
- [ ] Analyze: which novels/characters are hardest?
- [ ] Check: are we missing subtle contradictions?
- [ ] Verify: are we too conservative or too permissive?

---

### **Phase 7: Final Pipeline** (Days 12-14)

#### Step 7.1: End-to-End Script
Create `predict.py`:
```python
def predict_consistency(novel_path, backstory_text, model_path):
    # Load model
    # Preprocess novel + backstory
    # Run inference
    # Return: prediction (0/1), confidence, rationale
```

#### Step 7.2: Batch Prediction
Create `generate_results.py`:
```python
# Load test.csv
# For each row:
#   - Load corresponding novel
#   - Get backstory from row['content']
#   - Predict consistency
#   - Save to results.csv
```

#### Step 7.3: Results CSV Format
```csv
Story ID,Prediction,Rationale
1,1,"Earlier economic shock makes outcome necessary"
2,0,"Proposed backstory contradicts later actions"
```

---

## 🚀 Optimization Strategies to Win

### **1. Data-Level Optimizations**

#### A. Data Augmentation
- [ ] **Paraphrasing**: Create variations of backstories (consistent ones stay consistent, contradicts stay contradict)
- [ ] **Negative Sampling**: Generate plausible-but-wrong backstories
- [ ] **Synthetic Examples**: Use GPT to generate training examples

#### B. Smart Data Selection
- [ ] **Hard Example Mining**: Focus training on borderline cases
- [ ] **Balanced Sampling**: Ensure equal representation of classes
- [ ] **Character-Specific Training**: Fine-tune on examples from same character

### **2. Model-Level Optimizations**

#### A. BDH-Specific Advantages
- [ ] **Leverage Sparsity**: Use sparse activations for interpretability
- [ ] **Attention Analysis**: Visualize what BDH attends to
- [ ] **Layer-wise Representations**: Use multiple layer outputs, not just final

#### B. Hybrid Approaches
- [ ] **BDH + RAG**: Use BDH for encoding, external knowledge for reasoning
- [ ] **BDH + Rule-Based**: Combine neural with symbolic checks
- [ ] **Multi-Model Ensemble**: BDH + small Transformer + baseline classifier

### **3. Training Optimizations**

#### A. Efficient Training
- [ ] **Gradient Checkpointing**: Save memory for longer sequences
- [ ] **Dynamic Batching**: Adjust batch size based on sequence length
- [ ] **Distributed Training**: Multi-GPU if available

#### B. Advanced Techniques
- [ ] **Label Smoothing**: Prevent overconfidence
- [ ] **Focal Loss**: Focus on hard examples
- [ ] **Progressive Training**: Start with short sequences, gradually increase

### **4. Inference Optimizations**

#### A. Speed Improvements
- [ ] **Model Quantization**: Reduce model size (INT8)
- [ ] **Pruning**: Remove unimportant neurons
- [ ] **Caching**: Cache novel embeddings, only recompute backstory

#### B. Accuracy Improvements
- [ ] **Test-Time Augmentation**: Predict on multiple views, average
- [ ] **Confidence Calibration**: Ensure confidence scores are meaningful
- [ ] **Re-ranking**: Use separate model to verify predictions

---

## 📊 Competition-Winning Strategy

### **Week 1: Foundation**
1. **Days 1-2**: Understand data, set up pipeline
2. **Days 3-4**: Get baseline BDH running
3. **Days 5-7**: Train first model, achieve >60% accuracy

### **Week 2: Optimization**
1. **Days 8-10**: Implement long-context handling
2. **Days 11-12**: Fine-tune, ensemble, optimize
3. **Days 13-14**: Final evaluation, error analysis, submission prep

### **Key Success Factors**

#### 1. **Novelty** (Track B advantage)
- ✅ Use BDH's unique properties (sparsity, interpretability)
- ✅ Show how BDH differs from Transformers on this task
- ✅ Visualize attention patterns, neuron activations

#### 2. **Robustness**
- ✅ Handle edge cases gracefully
- ✅ Consistent performance across different novels
- ✅ Good performance on both "consistent" and "contradict" classes

#### 3. **Evidence & Interpretability**
- ✅ Provide clear rationale for predictions
- ✅ Highlight relevant novel passages
- ✅ Show attention visualizations (BDH advantage!)

#### 4. **Technical Depth**
- ✅ Experiment with different BDH configurations
- ✅ Compare pretrained vs non-pretrained
- ✅ Analyze layer-wise contributions

#### 5. **Presentation**
- ✅ Clear, concise report (max 10 pages)
- ✅ Visualizations (attention, sparsity, embeddings)
- ✅ Honest discussion of limitations

---

## 📝 Implementation Checklist

### **Core Components**
- [ ] `data_loader.py`: Load novels and backstories
- [ ] `train_bdh.py`: Training script for consistency task
- [ ] `predict.py`: Inference script
- [ ] `generate_results.py`: Batch prediction for test set
- [ ] `models/bdh_classifier.py`: BDH + classification head

### **Analysis & Evaluation**
- [ ] `evaluate.py`: Validation and metrics
- [ ] `analyze_errors.py`: Error case analysis
- [ ] `visualize_attention.py`: Attention pattern visualization
- [ ] `extract_evidence.py`: Evidence extraction for rationale

### **Utilities**
- [ ] `utils/text_processing.py`: Tokenization, chunking
- [ ] `utils/training_utils.py`: Training helpers
- [ ] `config.py`: Configuration management
- [ ] `requirements.txt`: Dependencies

### **Documentation**
- [ ] `README.md`: This file (updated)
- [ ] `REPORT.md`: Competition report (10 pages max)
- [ ] `SUBMISSION_GUIDE.md`: How to reproduce results

---

## 🎯 Quick Start Guide

### **Step 1: Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Verify data files
ls files/*.txt  # Should see novel files
head files/train.csv
```

### **Step 2: Train Baseline Model**
```bash
# Train BDH on narrative corpus
python train_on_narratives.py

# Fine-tune for consistency task
python train_consistency.py --pretrained_checkpoint=checkpoints/pretrained.pt
```

### **Step 3: Evaluate**
```bash
# Validate on training set
python evaluate.py --model=checkpoints/best_model.pt --data=files/train.csv

# Generate predictions for test set
python generate_results.py --model=checkpoints/best_model.pt --output=results.csv
```

### **Step 4: Analyze & Improve**
```bash
# Error analysis
python analyze_errors.py --predictions=results.csv --ground_truth=files/train.csv

# Visualize attention
python visualize_attention.py --example_id=1
```

---

## 🔥 Pro Tips to Win

1. **Start Simple**: Get baseline working first, then optimize
2. **Iterate Fast**: Train small models first, validate approach, then scale
3. **Visualize Early**: Use BDH's interpretability to understand failures
4. **Focus on Edge Cases**: Hard examples are where you differentiate
5. **Document Everything**: Track experiments, hyperparameters, results
6. **Ensemble Late**: Only after individual models are good
7. **Test Thoroughly**: Ensure reproducibility on clean environment
8. **Tell a Story**: Report should explain *why* your approach works

---

## 📚 Resources

### **BDH Paper Sections to Study**
- Section 2: Architecture details
- Section 3: GPU formulation (implementation)
- Section 6: Interpretability (for visualizations)
- Section 7: Experimental validation

### **Community Projects to Reference**
- [adamskrodzki/bdh](https://github.com/adamskrodzki/bdh): Dynamic vocabulary, stateful attention
- [GrahLnn/bdh](https://github.com/GrahLnn/bdh): Educational fork with visualizations

### **Pathway Resources**
- [BDH Repository](https://github.com/pathwaycom/bdh)
- [Pathway Framework Docs](https://pathway.com/developers)
- [LLM Integration Guide](https://pathway.com/developers/user-guide/llm-xpack/llm-xpack-overview)

---

## 📋 Submission Requirements

### **Deliverables**
1. **Code** (reproducible)
   - Runnable end-to-end
   - Reads inputs, generates predictions
   - No manual steps required

2. **Report** (max 10 pages, excluding appendix)
   - Overall approach
   - Long context handling
   - How you distinguish causal signals from noise
   - Key limitations/failure cases
   - **Clarity > Length**

3. **Results CSV**
   ```csv
   Story ID,Prediction,Rationale
   1,1,"Earlier economic shock makes outcome necessary"
   2,0,"Proposed backstory contradicts later actions"
   ```

### **ZIP Structure**
```
<TEAMNAME>_KDSH_2026.zip
├── code/
│   ├── bdh.py
│   ├── train_consistency.py
│   ├── predict.py
│   ├── generate_results.py
│   └── ...
├── checkpoints/
│   └── best_model.pt
├── results.csv
├── REPORT.pdf
└── README.md
```

---

**Good luck! 🚀 Use BDH's unique properties to build something innovative!**

<!--For visualization and interpretability analysis, explore the example notebooks in `notebooks/`.-->



## Learn and Discuss

- Watch the *SuperDataScience podcast* [▶️ *Dragon Hatchling: The Missing Link Between Transformers and the Brain*](https://www.youtube.com/watch?v=mfV44-mtg7c) (72 min.) featuring Adrian Kosowski in conversation with Jon Krohn, unpacking BDH’s neuron-level architecture and sparse reasoning dynamics.

- Read about BDH in
[*Forbes*](https://www.forbes.com/sites/victordey/2025/10/08/can-ai-learn-and-evolve-like-a-brain-pathways-bold-research-thinks-so/),
[*Semafor*](https://www.semafor.com/article/10/01/2025/new-ai-research-claims-to-be-getting-closer-to-modeling-human-brain),
[*The Turing Post*](https://www.turingpost.com/p/fod-121-300-million-to-start-a-big-promise-for-science#the-freshest-research-papers-catego),
[*Quantum Zeitgeist*](https://quantumzeitgeist.com/palo-alto-ai-firm-pathway-unveils-post-transformer-architecture-for-autonomous-ai/),
[*Golem*](https://www.golem.de/news/neue-ki-architektur-was-ist-baby-dragon-hatchling-2510-201047-2.html),
and elsewhere in the media.

- Discuss and share the BDH paper on:
[*Hugging Face Papers*](https://huggingface.co/papers/2509.26507), 
[*Alphaxiv*](https://alphaxiv.org/abs/2509.26507),
and [*EmergentMind*](https://emergentmind.com/papers/2509.26507).

## Community Projects

- [adamskrodzki/bdh](https://github.com/adamskrodzki/bdh): dynamic vocabulary, stateful attention
- [mosure/burn_dragon_hatchling](https://github.com/mosure/burn_dragon_hatchling): Burn port
- [severian42/bdh](https://github.com/severian42/bdh): MLX port
- [Git-Faisal/bdh](https://github.com/Git-Faisal/bdh)
- [GrahLnn/bdh](https://github.com/GrahLnn/bdh)

## Acknowledgements
We thank Andrej Karpathy for the [nanoGPT](https://github.com/karpathy/nanoGPT/) code and the tiny Shapespeare dataset used in this demonstration.

BDH research stands at the intersection of **AI architecture**, **biological learning models**, and **theoretical computer science**—an effort to map the *equations of reasoning* between artificial and biological intelligence.

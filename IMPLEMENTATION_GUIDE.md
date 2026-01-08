# 🚀 Quick Implementation Guide - KDSH 2026

## Current Status ✅

- ✅ BDH architecture implemented (`bdh.py`)
- ✅ Basic inference script running (`inference.py`)
- ✅ Baseline counterfactual comparison working
- ✅ Data files present (`files/train.csv`, `files/test.csv`)
- ✅ Novel texts available (`files/backstory.txt`, `files/novel.txt`)

## Immediate Next Steps (Priority Order)

### 🔴 CRITICAL: Days 1-3

#### 1. Create Data Loading Pipeline
**File**: `data_loader.py`

```python
import pandas as pd
import os

class NarrativeDataset:
    def __init__(self, csv_path, data_dir):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.novel_cache = {}
    
    def load_novel(self, book_name):
        # Map book names to actual files
        # "In Search of the Castaways" → files/backstory.txt or files/novel.txt
        # "The Count of Monte Cristo" → find corresponding file
        pass
    
    def get_example(self, idx):
        row = self.df.iloc[idx]
        novel = self.load_novel(row['book_name'])
        backstory = row['content']
        label = row.get('label', None)  # None for test set
        return novel, backstory, label
```

**Tasks**:
- [ ] Map book names to actual text files
- [ ] Create efficient text loading (cache if needed)
- [ ] Handle character encoding

#### 2. Create BDH Classifier Model
**File**: `models/bdh_classifier.py`

```python
import torch.nn as nn
import bdh

class BDHConsistencyClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bdh = bdh.BDH(config)
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, novel_tokens, backstory_tokens):
        # Option 1: Concatenate backstory + novel
        combined = torch.cat([backstory_tokens, novel_tokens], dim=1)
        logits, _ = self.bdh(combined)
        
        # Pool over sequence (mean of last 10 tokens)
        pooled = logits[:, -10:].mean(dim=1)
        
        # Classify
        prediction = self.classifier(pooled)
        return prediction
```

**Tasks**:
- [ ] Implement classification head
- [ ] Decide on pooling strategy (mean, attention, last token)
- [ ] Test forward pass

#### 3. Create Training Script
**File**: `train_consistency.py`

```python
import torch
import torch.nn as nn
from data_loader import NarrativeDataset
from models.bdh_classifier import BDHConsistencyClassifier

def train():
    # Load data
    train_data = NarrativeDataset('files/train.csv', 'files')
    
    # Model
    config = bdh.BDHConfig()
    model = BDHConsistencyClassifier(config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(10):
        for idx in range(len(train_data)):
            novel, backstory, label = train_data.get_example(idx)
            
            # Tokenize
            novel_tokens = tokenize(novel)
            backstory_tokens = tokenize(backstory)
            
            # Forward
            prediction = model(novel_tokens, backstory_tokens)
            target = torch.tensor(1.0 if label == 'consistent' else 0.0)
            
            # Loss
            loss = criterion(prediction, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

**Tasks**:
- [ ] Implement training loop
- [ ] Add validation split
- [ ] Add checkpointing
- [ ] Add logging

### 🟡 HIGH PRIORITY: Days 4-7

#### 4. Long-Context Handling
**File**: `utils/chunking.py`

```python
def process_long_novel(model, novel_tokens, backstory_tokens, chunk_size=512):
    # Strategy: Process novel in chunks, maintain state
    representations = []
    
    for i in range(0, len(novel_tokens), chunk_size):
        chunk = novel_tokens[i:i+chunk_size]
        
        # If first chunk, include backstory
        if i == 0:
            combined = torch.cat([backstory_tokens, chunk], dim=1)
        else:
            combined = chunk
        
        # Process through BDH
        logits, _ = model.bdh(combined)
        representations.append(logits[:, -1])  # Last token
    
    # Aggregate representations (mean pooling)
    final_rep = torch.stack(representations).mean(dim=0)
    
    return final_rep
```

**Tasks**:
- [ ] Implement chunking strategy
- [ ] Test on full novels
- [ ] Optimize memory usage

#### 5. Prediction Script
**File**: `predict.py`

```python
def predict_consistency(novel_path, backstory_text, model_path):
    # Load model
    model = BDHConsistencyClassifier.load(model_path)
    model.eval()
    
    # Load novel
    novel = load_text(novel_path)
    
    # Tokenize
    novel_tokens = tokenize(novel)
    backstory_tokens = tokenize(backstory_text)
    
    # Predict
    with torch.no_grad():
        prediction = model(novel_tokens, backstory_tokens)
    
    # Return
    return int(prediction > 0.5), prediction.item()
```

**Tasks**:
- [ ] Implement prediction function
- [ ] Add batch prediction
- [ ] Generate results.csv

### 🟢 MEDIUM PRIORITY: Days 8-12

#### 6. Evidence Extraction (Optional for Track B)
**File**: `extract_evidence.py`

```python
def extract_supporting_evidence(model, novel, backstory):
    # Run model
    logits, attention = model.bdh.get_attention(novel, backstory)
    
    # Find high-attention passages
    top_indices = torch.topk(attention, k=5).indices
    
    # Extract text passages
    evidence = [novel[i-50:i+50] for i in top_indices]
    
    return evidence
```

#### 7. Visualization Tools
**File**: `visualize_attention.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(model, novel, backstory):
    attention = model.bdh.get_attention_weights(novel, backstory)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention.cpu().numpy(), cmap='YlOrRd')
    plt.title('BDH Attention Patterns')
    plt.savefig('attention_heatmap.png')
```

### 🔵 OPTIMIZATION: Days 10-14

#### 8. Model Improvements
- [ ] Increase model size (if compute allows)
- [ ] Pretraining on narrative corpus
- [ ] Ensemble multiple models
- [ ] Fine-tune hyperparameters

#### 9. Advanced Features
- [ ] Contrastive learning
- [ ] Multi-task learning
- [ ] Test-time augmentation

## File Structure to Create

```
KDSH/
├── bdh.py (✅ existing)
├── train.py (✅ existing, modify for new task)
├── inference.py (✅ existing)
├── data_loader.py (🆕 CREATE)
├── models/
│   └── bdh_classifier.py (🆕 CREATE)
├── train_consistency.py (🆕 CREATE)
├── predict.py (🆕 CREATE)
├── generate_results.py (🆕 CREATE)
├── evaluate.py (🆕 CREATE)
├── utils/
│   ├── chunking.py (🆕 CREATE)
│   ├── text_processing.py (🆕 CREATE)
│   └── training_utils.py (🆕 CREATE)
├── visualize_attention.py (🆕 CREATE - optional)
└── extract_evidence.py (🆕 CREATE - optional)
```

## Quick Wins (First 24 Hours)

1. **Get baseline accuracy >50%**
   - Use existing inference.py
   - Add simple threshold-based classifier
   - Train on train.csv

2. **Visualize attention patterns**
   - Modify inference.py to save attention
   - Create simple heatmap
   - Use for report

3. **Document current approach**
   - Write first draft of report
   - Describe BDH architecture
   - Explain counterfactual comparison

## Testing Checklist

Before submission, ensure:

- [ ] Code runs on clean environment
- [ ] No manual intervention required
- [ ] Generates results.csv correctly
- [ ] Format matches specification
- [ ] Reproducible results (set seeds)
- [ ] Documentation complete
- [ ] Report within 10 pages
- [ ] Visualizations clear and informative

## Common Pitfalls to Avoid

1. ❌ **Overfitting to train set** - Use proper validation
2. ❌ **Ignoring long-context** - Must handle full novels
3. ❌ **No interpretability** - Track B values this!
4. ❌ **Unclear report** - Judges value clarity
5. ❌ **Non-reproducible** - Test on clean environment
6. ❌ **Missing edge cases** - Test on different novels
7. ❌ **No evidence** - Provide rationale even if optional

## Success Metrics

**Minimum Viable Submission**:
- ✅ Accuracy > 60%
- ✅ Handles long contexts
- ✅ Clear report (even if short)
- ✅ Reproducible code

**Winning Submission**:
- ✅ Accuracy > 80%
- ✅ Innovative BDH usage
- ✅ Clear evidence/rationale
- ✅ Beautiful visualizations
- ✅ Honest limitations discussion

---

**Start with Step 1 (Data Loader) and work through sequentially!** 🚀

# Chapter 3: Semantic Turn Detection via Text Normalization

## Objective

Retrain all models (General, Domain, Agent, Customer) on **normalized text** to eliminate punctuation dependency and force models to learn semantic patterns of turn completion.

---

## Text Normalization Strategy

### Normalization Function
```python
def normalize_text(text: str) -> str:
    """Remove all punctuation and lowercase."""
    # Remove all punctuation: .,!?;:'"()-
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text
```

### Application Points
1. **Data Processing:** Normalize all training/validation/test text BEFORE tokenization
2. **Inference:** Accept both normalized and raw text (normalize internally)

---

## Model Variants

All Chapter 1 models remain intact. Chapter 3 adds normalized variants:

```
models/
├── general/                 # Chapter 1: Baseline with punctuation
├── domain/                  # Chapter 1: Domain-specific with punctuation  
├── agent/                   # Chapter 1: Agent-specific with punctuation
├── customer/                # Chapter 1: Customer-specific with punctuation
├── general_normalized/      # Chapter 3: NEW - Baseline without punctuation
├── domain_normalized/       # Chapter 3: NEW - Domain without punctuation
├── agent_normalized/        # Chapter 3: NEW - Agent without punctuation
└── customer_normalized/     # Chapter 3: NEW - Customer without punctuation
```

**Naming Convention:**
- Suffix `_normalized` for all Chapter 3 models
- Clear separation in directory structure

---

## Data Processing Changes

### Update `src/data_processor.py`

Add new method to `OneShotDataProcessor`:

```python
def process_datasets_normalized(self):
    """
    Process all datasets with text normalization.
    Creates normalized versions alongside original datasets.
    """
    # Load original data (reuse existing methods)
    general_ds = self.load_easy_turn_dataset(max_examples=20000)
    call_center_ds, agent_ds, customer_ds, metadata = self.load_call_center_dataset(max_examples=20000)
    
    # Apply normalization to all text fields
    def normalize_dataset(dataset):
        def normalize_example(example):
            example['text'] = normalize_text(example['text'])
            return example
        return dataset.map(normalize_example)
    
    general_normalized = normalize_dataset(general_ds)
    call_center_normalized = normalize_dataset(call_center_ds)
    agent_normalized = normalize_dataset(agent_ds)
    customer_normalized = normalize_dataset(customer_ds)
    
    # Save to separate directories
    general_normalized.save_to_disk("data/processed/general_normalized")
    call_center_normalized.save_to_disk("data/processed/call_center_normalized")
    agent_normalized.save_to_disk("data/processed/agent_normalized")
    customer_normalized.save_to_disk("data/processed/customer_normalized")
```

### Datasets Directory Structure
```
data/processed/
├── general/                    # Original (with punctuation)
├── call_center/                # Original
├── agent/                      # Original
├── customer/                   # Original
├── general_normalized/         # Chapter 3: Normalized
├── call_center_normalized/     # Chapter 3: Normalized
├── agent_normalized/           # Chapter 3: Normalized
└── customer_normalized/        # Chapter 3: Normalized
```

---

## Training Script

### Create `chapter_03_normalized/run_experiment.py`

Structure similar to Chapter 1, but:
1. Load normalized datasets
2. Train to `models/*_normalized/` directories
3. Evaluate on BOTH normalized and original test sets (cross-robustness test)

```python
# Load normalized datasets
general_normalized = load_from_disk("../data/processed/general_normalized")
call_center_normalized = load_from_disk("../data/processed/call_center_normalized")
agent_normalized = load_from_disk("../data/processed/agent_normalized")
customer_normalized = load_from_disk("../data/processed/customer_normalized")

# Train models
trainer_general = TurnDetectionTrainer(...)
trainer_general.train(
    general_normalized['train'],
    general_normalized['validation'],
    output_dir="../models/general_normalized"
)

# Similar for domain_normalized, agent_normalized, customer_normalized
```

---

## Evaluation Strategy

### Cross-Robustness Testing

For each normalized model, evaluate on:
1. **Normalized test set** (primary accuracy metric)
2. **Original test set** (robustness - can it handle punctuation it never saw?)

For each original model (Chapter 1), evaluate on:
1. **Original test set** (already done in Chapter 1)
2. **Normalized test set** (robustness - can it handle missing punctuation?)

**Full Evaluation Matrix:**

| Model | Test Set | Expected Performance |
|-------|----------|---------------------|
| General | General (original) | High (Chapter 1 baseline) |
| General | General (normalized) | Poor (punctuation-dependent) |
| General Normalized | General (normalized) | High (trained on this) |
| General Normalized | General (original) | ? (extra punctuation - should ignore) |
| Domain | CallCenter (original) | 100% (Chapter 1) |
| Domain | CallCenter (normalized) | Poor (66.7% flip in Chapter 2) |
| Domain Normalized | CallCenter (normalized) | High (trained on this) |
| Domain Normalized | CallCenter (original) | ? (robustness test) |

---

## Visualization Updates

### New Plots for Chapter 3

1. **Robustness Comparison Matrix**
   - Heatmap: Model x Test Set → Accuracy
   - Diagonal = trained domain, off-diagonal = generalization

2. **Punctuation Sensitivity Before/After**
   - Bar chart: % predictions changed when punctuation removed
   - Original models vs. Normalized models

3. **Confidence Stability**
   - Box plots: Confidence distributions across test sets
   - Compare fluctuation in Chapter 1 vs. Chapter 3 models

---

## Expected Findings

### Hypothesis 1: Normalized Models Are Punctuation-Agnostic
- Normalized models should maintain predictions on both normalized and original text
- Punctuation becomes "noise" to ignore, not a signal

### Hypothesis 2: Accuracy Trade-off
- Normalized models may have LOWER accuracy on normalized test sets
- Removing punctuation removes a strong signal - forces harder semantic learning
- But predictions should be MORE STABLE and ROBUST

### Hypothesis 3: Domain Advantage Persists
- Domain-normalized should still outperform General-normalized on call center data
- Domain-specific vocabulary/patterns exist beyond punctuation

---

## Chapter 3 Deliverables

1. **`chapter_03_normalized/README.md`**
   - Motivation: Chapter 2 findings
   - Methodology: Text normalization approach
   - Results: Robustness comparison
   - Analysis: Semantic learning vs. punctuation shortcuts

2. **`chapter_03_normalized/run_experiment.py`**
   - Automated training pipeline for all normalized models

3. **`chapter_03_normalized/results/`**
   - Metrics for normalized models on both test sets
   - Robustness comparison figures

4. **Updated Root `README.md`**
   - Add Chapter 3 section with key visual

---

## Implementation Checklist

- [ ] Add `normalize_text()` function to `src/data_processor.py`
- [ ] Create `process_datasets_normalized()` method
- [ ] Run data processing to generate normalized datasets
- [ ] Create `chapter_03_normalized/` directory
- [ ] Implement `chapter_03_normalized/run_experiment.py`
- [ ] Update `src/visualize.py` with robustness comparison plots
- [ ] Train all 4 normalized models
- [ ] Evaluate on full cross-robustness matrix
- [ ] Generate visualizations
- [ ] Write Chapter 3 README
- [ ] Update root README with Chapter 3 summary
- [ ] Commit and push

---

## File Naming Conventions

**Models:**
- `models/general_normalized/`
- `models/domain_normalized/`
- `models/agent_normalized/`
- `models/customer_normalized/`

**Datasets:**
- `data/processed/general_normalized/`
- `data/processed/call_center_normalized/`
- `data/processed/agent_normalized/`
- `data/processed/customer_normalized/`

**Results:**
- `chapter_03_normalized/results/metrics/general_normalized_results.json`
- `chapter_03_normalized/results/figures/robustness_matrix.png`

**Code Imports:**
All scripts use the centralized `src/` directory - no code duplication.

---

## Testing Strategy

Before full run:
1. Process 1000 examples to verify normalization works
2. Train one model (e.g., `general_normalized`) for 1 epoch
3. Test inference on both normalized and original text
4. Verify results make sense before full experiment

---

**Next Steps:** User approval to proceed with implementation.


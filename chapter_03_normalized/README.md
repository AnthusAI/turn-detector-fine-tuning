# Chapter 3: Semantic Turn Detection via Text Normalization

> **Can models learn turn detection without punctuation cues, or is punctuation the only learnable signal?**

---

## The Question

Chapter 2 revealed that our models were **punctuation detectors**, not turn detectors. They learned simple rules:
- Ends with `.` or `?` → "Complete"
- No punctuation → "Incomplete"

This makes them redundant when ASR provides punctuation (just echoing the ASR's decision) and useless when it doesn't.

**Chapter 3 tests:** Can we force models to learn semantic patterns by training on **normalized text** (all punctuation removed, lowercased)?

---

## The Experiment

### Text Normalization

```python
def normalize_text(text: str) -> str:
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase
    text = text.lower()
    return text
```

**Examples:**
- Original: `"Is it a 250 or 350?"`
- Normalized: `"is it a 250 or 350"`

- Original: `"Yeah, that's fine."`
- Normalized: `"yeah thats fine"`

### Training

We retrained all 4 models on normalized text:
- **General Normalized:** PersonaChat + TURNS-2K (normalized)
- **Domain Normalized:** CallCenter EN (normalized)
- **Agent Normalized:** CallCenter agent utterances (normalized)
- **Customer Normalized:** CallCenter customer utterances (normalized)

### Evaluation: Cross-Robustness Testing

Each normalized model was tested on:
1. **Normalized test set** (no punctuation) - measures semantic learning
2. **Original test set** (with punctuation) - measures robustness

---

## Results

### Accuracy Comparison

| Model | Test Set | Accuracy | Notes |
|-------|----------|----------|-------|
| **General Normalized** | General (normalized) | **84.0%** | Semantic learning without punctuation |
| **General Normalized** | General (original) | **85.9%** | Nearly identical - punctuation-agnostic! |
| **General Normalized** | CallCenter (normalized) | **72.2%** | Lower on harder call center data |
| **Domain Normalized** | CallCenter (normalized) | **77.9%** | +5.7pp improvement from domain training |
| **Domain Normalized** | CallCenter (original) | **86.4%** | Benefits from punctuation, but still works without |
| **Agent Normalized** | Agent (normalized) | **76.4%** | Similar to domain |
| **Customer Normalized** | Customer (normalized) | **76.9%** | Similar to domain |

### Comparison to Chapter 1 (Punctuation-Dependent Models)

| Model | Chapter 1 (with punct.) | Chapter 3 (without punct.) | Accuracy Cost |
|-------|------------------------|---------------------------|---------------|
| General | 88.6% | 84.0% | **-4.6pp** |
| Domain | 100%* | 77.9% | **-22.1pp** |

*The 100% in Chapter 1 was achieved by learning punctuation shortcuts, not semantic understanding.

---

## Key Findings

### 1. Models CAN Learn Semantic Patterns

The normalized models achieve reasonable accuracy (77-84%) without any punctuation cues. They learned:
- **Trailing phrases:** "kind of", "you know", "or" signal incompleteness
- **Syntactic patterns:** Incomplete clauses, missing subjects/objects
- **Domain vocabulary:** Call center ending phrases vs. mid-utterance patterns

This suggests that **text-based turn detection is possible** without relying on ASR punctuation.

### 2. Punctuation-Agnostic Models Are Robust

The General Normalized model performs nearly identically on normalized and original text:
- 84.0% (no punctuation) vs 85.9% (with punctuation)

The model treats punctuation as **noise to ignore**, not a critical signal. This is exactly what we want for real-world deployment.

### 3. The Accuracy-Robustness Trade-off

**Chapter 1 models:** High accuracy (100%) by learning punctuation shortcuts → Brittle, redundant
**Chapter 3 models:** Lower accuracy (77-84%) by learning semantics → Robust, independent value

For production systems, the trade-off depends on:
- ASR punctuation reliability (is it always correct?)
- Value of independent signal (do we want to catch ASR mistakes?)
- Tolerance for lower accuracy (is 84% vs 100% acceptable?)

### 4. Domain Training Still Helps (But Less Than With Punctuation)

**On CallCenter test set:**
- General Normalized: 72.2% accuracy
- Domain Normalized: 77.9% accuracy (+5.7pp)

Domain-specific training improves performance on call center data by **5.7 percentage points**, even without punctuation cues. This validates the core hypothesis.

**Comparison across chapters:**
- Chapter 1 (with punctuation): Domain improved accuracy by +12.5pp (87.5% → 100%)
- Chapter 3 (without punctuation): Domain improves accuracy by +5.7pp (72.2% → 77.9%)

**Why is the improvement smaller without punctuation?**

1. **Call center data is harder without punctuation**
   - More disfluencies ("um", "uh", "like") that are harder to interpret semantically
   - More interruptions and incomplete fragments
   - Punctuation helps disambiguate these cases

2. **General data trains more robust semantic features**
   - Clean conversational data with clear turn boundaries
   - General model achieves 84% on general test (vs 77.9% on harder call center test)
   - Domain-specific patterns matter less without punctuation shortcuts

The domain model still wins on its target data, but the advantage is smaller when relying on semantics alone.

### 5. The 100% Accuracy Was a Red Flag

Chapter 1's perfect 100% accuracy on call center data should have been suspicious. No real-world model achieves perfection.

When we remove the punctuation shortcut, the "real" accuracy is revealed: **77.9%**

The 22-point gap (100% → 77.9%) shows how much the Chapter 1 model was relying on punctuation rather than semantic understanding.

---

## Implications

### When to Use Normalized Models

**Use normalized (Chapter 3) models when:**
- ASR doesn't provide reliable punctuation
- You need independent turn detection signal
- Robustness > peak accuracy
- You want to catch ASR punctuation mistakes

**Use punctuation-dependent (Chapter 1) models when:**
- ASR always provides accurate punctuation
- You need maximum accuracy (and can tolerate redundancy)
- You're okay with the model just confirming ASR decisions

### Model Selection for Production

**For call center applications:**
- Domain Normalized model: 77.9% on call center data
- General Normalized model: 72.2% on call center data
- **Domain training provides 5.7pp improvement** - use the domain model

**For general conversation applications:**
- General Normalized model: 84.0% on general data
- Achieves high accuracy across diverse conversational patterns
- More robust to varied language styles

---

## Reproducing This Experiment

```bash
# 1. Create normalized datasets
python -c "from src.data_processor import create_normalized_datasets; create_normalized_datasets()"

# 2. Train normalized models
cd chapter_03_normalized
python run_experiment.py

# 3. Results saved to:
#    results/metrics/all_normalized_results.json
```

---

## Conclusion

**These results suggest that models can learn semantic turn detection from text alone.**

The accuracy cost is significant (72-84% vs 88-100%), but the gain in robustness and independence may be worthwhile for systems that can't rely on ASR punctuation.

**Key findings:**
1. Domain-specific training helps even without punctuation (+5.7pp on call center data)
2. The improvement is smaller than with punctuation (+5.7pp vs +12.5pp)
3. Punctuation-agnostic models are robust to ASR variations
4. The 100% accuracy in Chapter 1 was achieved by learning punctuation shortcuts, not semantic understanding

**Trade-off summary:**
- Chapter 1: High accuracy, low robustness (punctuation shortcuts)
- Chapter 3: Lower accuracy, high robustness (semantic patterns)

The choice depends on your production constraints and whether you value peak accuracy or reliable, independent turn detection.


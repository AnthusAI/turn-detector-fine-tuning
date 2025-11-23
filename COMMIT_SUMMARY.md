# Commit Summary: Chapters 2 & 3 Complete

## What's New

### Chapter 2: The Punctuation Problem
- **Analysis:** Streaming inference behavior and punctuation robustness testing
- **Key Finding:** Models learned to detect punctuation, not semantic turn completion
- **Evidence:** 67% of Domain model predictions flip when punctuation is removed
- **Scripts:**
  - `streaming_inference.py` - Word-by-word prediction analysis
  - `punctuation_robustness_test.py` - Punctuation sensitivity testing

### Chapter 3: Semantic Turn Detection via Text Normalization
- **Approach:** Retrained all models on normalized text (no punctuation, lowercased)
- **Result:** Models CAN learn semantic patterns without punctuation
- **Accuracy:** 77.9% on call center data (vs 100% with punctuation shortcuts)
- **Validation:** Domain training still helps (+5.7pp improvement: 72.2% → 77.9%)
- **Models Trained:**
  - `general_normalized/` - 84.0% on general data, 72.2% on call center data
  - `domain_normalized/` - 77.9% on call center data
  - `agent_normalized/` - 76.4% on agent data
  - `customer_normalized/` - 76.9% on customer data

## Key Insights

1. **The 100% accuracy in Chapter 1 was deceptive** - achieved by learning punctuation shortcuts
2. **Semantic learning is possible** - models achieve 77-84% without punctuation
3. **Domain training works** - even without punctuation, domain models outperform general models on target data
4. **Trade-off is real** - Lower accuracy (77% vs 100%) for robustness and independence from ASR punctuation

## Updated Documentation

All READMEs updated to reflect current understanding:
- **Root README:** Overview of all three chapters with accurate results
- **Chapter 1 README:** Added warning about suspicious 100% accuracy
- **Chapter 2 README:** Detailed punctuation dependency analysis
- **Chapter 3 README:** Full results and interpretation
- No references to "fixes" or investigation process - all stated as current facts

## Files Changed

### New Files:
- `chapter_02_service/streaming_inference.py`
- `chapter_02_service/punctuation_robustness_test.py`
- `chapter_02_service/README.md` (updated)
- `chapter_03_normalized/run_experiment.py`
- `chapter_03_normalized/README.md`
- `chapter_03_normalized/results/metrics/all_normalized_results.json`
- `data/processed/*_normalized/` (4 datasets)
- `src/data_processor.py` (added `normalize_text()` and `create_normalized_datasets()`)

### Modified Files:
- `README.md` - Updated with Chapters 2 & 3 summaries
- `chapter_01_fine_tuning/README.md` - Added warning about 100% accuracy
- `.gitignore` - Added Chapter 2 & 3 patterns

### Not Committed (as per .gitignore):
- `models/general_normalized/`, `models/domain_normalized/`, etc. (model checkpoints)
- `chapter_02_service/streaming_examples.md`, `punctuation_robustness_results.md` (regenerable)
- `chapter_03_normalized/experiment_log.txt` (training logs)
- `data/processed/*_normalized/` (large datasets)

## Ready to Commit

All documentation is clean, factual, and reflects the current understanding. No process narrative or temporary investigation language remains.

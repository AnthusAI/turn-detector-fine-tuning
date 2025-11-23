#!/usr/bin/env python
"""
Chapter 4: Conversation-Level Evaluation

Evaluates all models on multi-turn conversations using conversation-level metrics.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conversation_processor import ConversationProcessor
from src.conversation_evaluator import ConversationEvaluator
import json
from datasets import load_from_disk

print("="*80)
print("CHAPTER 4: CONVERSATION-LEVEL EVALUATION")
print("="*80)

# Step 1: Extract conversations (or load if already exists)
processor = ConversationProcessor()

call_center_conv_path = Path("data/call_center_conversations.json")
general_conv_path = Path("data/general_conversations.json")

if not call_center_conv_path.exists():
    print("\n" + "="*80)
    print("STEP 1: EXTRACTING CONVERSATIONS")
    print("="*80)
    
    # Extract call center conversations
    print("\nCall Center Conversations:")
    call_center_test = load_from_disk("../data/processed/call_center")['test']
    call_center_convs = processor.extract_call_center_conversations(
        call_center_test,
        num_conversations=100,
        min_turns=5,
        max_turns=20
    )
    processor.save_conversations(call_center_convs, call_center_conv_path)
    processor.get_conversation_statistics(call_center_convs)
    
    # Extract general conversations
    print("\nGeneral Conversations:")
    general_test = load_from_disk("../data/processed/general")['test']
    general_convs = processor.extract_general_conversations(
        general_test,
        num_conversations=100,
        min_turns=5,
        max_turns=15
    )
    processor.save_conversations(general_convs, general_conv_path)
    processor.get_conversation_statistics(general_convs)
else:
    print("\n✓ Conversations already extracted, loading from disk...")
    call_center_convs = processor.load_conversations(call_center_conv_path)
    general_convs = processor.load_conversations(general_conv_path)

# Step 2: Evaluate all models
print("\n" + "="*80)
print("STEP 2: EVALUATING MODELS ON CONVERSATIONS")
print("="*80)

# Define all models to evaluate
models_to_evaluate = [
    # Chapter 1 models (with punctuation)
    ("General (Ch1)", "../models/general"),
    ("Domain (Ch1)", "../models/domain"),
    
    # Chapter 3 models (normalized)
    ("General Normalized (Ch3)", "../models/general_normalized"),
    ("Domain Normalized (Ch3)", "../models/domain_normalized"),
]

all_results = {}

for model_name, model_path in models_to_evaluate:
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    evaluator = ConversationEvaluator(model_path, model_name)
    
    # Evaluate on call center conversations
    print(f"\n--- {model_name} on Call Center Conversations ---")
    cc_results = evaluator.evaluate_conversations(call_center_convs)
    
    # Evaluate on general conversations
    print(f"\n--- {model_name} on General Conversations ---")
    gen_results = evaluator.evaluate_conversations(general_convs)
    
    # Store results
    all_results[model_name] = {
        'call_center': cc_results,
        'general': gen_results
    }
    
    # Print summary
    print(f"\n{model_name} - Call Center Results:")
    cc_metrics = cc_results['metrics']
    print(f"  Interruption Rate: {cc_metrics['interruption_rate']:.1%}")
    print(f"  Missed Turn Rate: {cc_metrics['missed_turn_rate']:.1%}")
    print(f"  F1 Score: {cc_metrics['f1_score']:.3f}")
    print(f"  Perfect Conversation Rate: {cc_metrics['perfect_conversation_rate']:.1%}")
    print(f"  Mean Words to Detection: {cc_metrics['mean_words_to_detection']:.2f}")
    
    print(f"\n{model_name} - General Results:")
    gen_metrics = gen_results['metrics']
    print(f"  Interruption Rate: {gen_metrics['interruption_rate']:.1%}")
    print(f"  Missed Turn Rate: {gen_metrics['missed_turn_rate']:.1%}")
    print(f"  F1 Score: {gen_metrics['f1_score']:.3f}")
    print(f"  Perfect Conversation Rate: {gen_metrics['perfect_conversation_rate']:.1%}")
    print(f"  Mean Words to Detection: {gen_metrics['mean_words_to_detection']:.2f}")

# Step 3: Save results
print("\n" + "="*80)
print("STEP 3: SAVING RESULTS")
print("="*80)

results_path = Path("results/metrics/conversation_metrics.json")
results_path.parent.mkdir(parents=True, exist_ok=True)

with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"✓ Results saved to {results_path}")

# Step 4: Print comparison table
print("\n" + "="*80)
print("CONVERSATION-LEVEL METRICS COMPARISON")
print("="*80)

print("\n=== Call Center Conversations ===\n")
print(f"{'Model':<30} {'IR ↓':<8} {'MTR ↓':<8} {'F1 ↑':<8} {'PCR ↑':<8} {'MWD ↓':<8}")
print("-" * 80)

for model_name in all_results.keys():
    metrics = all_results[model_name]['call_center']['metrics']
    print(f"{model_name:<30} "
          f"{metrics['interruption_rate']:>6.1%}  "
          f"{metrics['missed_turn_rate']:>6.1%}  "
          f"{metrics['f1_score']:>6.3f}  "
          f"{metrics['perfect_conversation_rate']:>6.1%}  "
          f"{metrics['mean_words_to_detection']:>6.2f}")

print("\n=== General Conversations ===\n")
print(f"{'Model':<30} {'IR ↓':<8} {'MTR ↓':<8} {'F1 ↑':<8} {'PCR ↑':<8} {'MWD ↓':<8}")
print("-" * 80)

for model_name in all_results.keys():
    metrics = all_results[model_name]['general']['metrics']
    print(f"{model_name:<30} "
          f"{metrics['interruption_rate']:>6.1%}  "
          f"{metrics['missed_turn_rate']:>6.1%}  "
          f"{metrics['f1_score']:>6.3f}  "
          f"{metrics['perfect_conversation_rate']:>6.1%}  "
          f"{metrics['mean_words_to_detection']:>6.2f}")

print("\n" + "="*80)
print("CHAPTER 4 EVALUATION COMPLETE")
print("="*80)
print(f"\nResults saved to: {results_path}")
print("Next: Generate visualizations and write Chapter 4 README")


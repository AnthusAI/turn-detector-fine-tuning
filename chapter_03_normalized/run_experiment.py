#!/usr/bin/env python
"""
Chapter 3: Semantic Turn Detection via Text Normalization

Trains models on normalized text (no punctuation, lowercased) to force
semantic learning and eliminate punctuation dependency.

This script trains 4 normalized models and evaluates them on BOTH
normalized and original test sets to measure robustness.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import TurnDetectionTrainer
from src.evaluate import ModelEvaluator
from datasets import load_from_disk
import json

print("="*80)
print("CHAPTER 3: NORMALIZED TEXT TRAINING")
print("="*80)

# Load normalized datasets
print("\nLoading normalized datasets...")
general_normalized = load_from_disk("../data/processed/general_normalized")
call_center_normalized = load_from_disk("../data/processed/call_center_normalized")
agent_normalized = load_from_disk("../data/processed/agent_normalized")
customer_normalized = load_from_disk("../data/processed/customer_normalized")

# Also load original datasets for cross-robustness testing
print("Loading original datasets for cross-testing...")
general_original = load_from_disk("../data/processed/general")
call_center_original = load_from_disk("../data/processed/call_center")

print(f"\n✓ Datasets loaded:")
print(f"  General (normalized):     {len(general_normalized['train'])} train, {len(general_normalized['test'])} test")
print(f"  Call Center (normalized): {len(call_center_normalized['train'])} train, {len(call_center_normalized['test'])} test")
print(f"  Agent (normalized):       {len(agent_normalized['train'])} train, {len(agent_normalized['test'])} test")
print(f"  Customer (normalized):    {len(customer_normalized['train'])} train, {len(customer_normalized['test'])} test")

# Train Model 1: General (normalized)
print("\n" + "="*80)
print("TRAINING MODEL 1: General (Normalized)")
print("="*80)
trainer_general = TurnDetectionTrainer(output_dir="../models/general_normalized")
trainer_general.train(
    train_dataset=general_normalized['train'],
    eval_dataset=general_normalized['validation'],
    num_epochs=3,
    batch_size=16
)

# Train Model 2: Domain (normalized)
print("\n" + "="*80)
print("TRAINING MODEL 2: Domain (Normalized)")
print("="*80)
trainer_domain = TurnDetectionTrainer(output_dir="../models/domain_normalized")
trainer_domain.train(
    train_dataset=call_center_normalized['train'],
    eval_dataset=call_center_normalized['validation'],
    num_epochs=3,
    batch_size=16
)

# Train Model 3: Agent (normalized)
print("\n" + "="*80)
print("TRAINING MODEL 3: Agent (Normalized)")
print("="*80)
trainer_agent = TurnDetectionTrainer(output_dir="../models/agent_normalized")
trainer_agent.train(
    train_dataset=agent_normalized['train'],
    eval_dataset=agent_normalized['validation'],
    num_epochs=3,
    batch_size=16
)

# Train Model 4: Customer (normalized)
print("\n" + "="*80)
print("TRAINING MODEL 4: Customer (Normalized)")
print("="*80)
trainer_customer = TurnDetectionTrainer(output_dir="../models/customer_normalized")
trainer_customer.train(
    train_dataset=customer_normalized['train'],
    eval_dataset=customer_normalized['validation'],
    num_epochs=3,
    batch_size=16
)

# Evaluation: Cross-Robustness Testing
print("\n" + "="*80)
print("EVALUATION: CROSS-ROBUSTNESS TESTING")
print("="*80)

all_results = {}

# Define evaluation matrix
evaluations = [
    # Normalized models on normalized test sets (primary metric)
    ("general_normalized", "../models/general_normalized", general_normalized['test'], "General Normalized Test"),
    ("domain_normalized", "../models/domain_normalized", call_center_normalized['test'], "CallCenter Normalized Test"),
    
    # Normalized models on ORIGINAL test sets (robustness: can they handle punctuation?)
    ("general_normalized", "../models/general_normalized", general_original['test'], "General Original Test"),
    ("domain_normalized", "../models/domain_normalized", call_center_original['test'], "CallCenter Original Test"),
    
    # Agent/Customer normalized models
    ("agent_normalized", "../models/agent_normalized", agent_normalized['test'], "Agent Normalized Test"),
    ("customer_normalized", "../models/customer_normalized", customer_normalized['test'], "Customer Normalized Test"),
]

for model_name, model_path, test_set, test_name in evaluations:
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} on {test_name}")
    print(f"{'='*80}")
    
    evaluator = ModelEvaluator(model_path)
    metrics = evaluator.evaluate(test_set)
    
    if model_name not in all_results:
        all_results[model_name] = {}
    all_results[model_name][test_name] = metrics
    
    print(f"\nResults for {model_name} on {test_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")

# Save combined results
result_file = Path("results/metrics/all_normalized_results.json")
result_file.parent.mkdir(parents=True, exist_ok=True)
with open(result_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✓ Results saved to {result_file}")

# Generate visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# We'll create new visualization functions in src/visualize.py for Chapter 3
# For now, just save the results

print("\n" + "="*80)
print("CHAPTER 3 EXPERIMENT COMPLETE")
print("="*80)
print(f"\nResults saved:")
print(f"  - results/metrics/all_normalized_results.json")


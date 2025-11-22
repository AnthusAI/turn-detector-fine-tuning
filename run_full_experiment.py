#!/usr/bin/env python
"""
Full Turn Detection Experiment
Trains all models and generates comprehensive results.
"""
import sys
sys.path.insert(0, '/Users/ryan.porter/turn-detector-fine-tuning')

from src.train import TurnDetectionTrainer
from src.evaluate import ModelEvaluator
from src.visualize import generate_all_visualizations
from datasets import load_from_disk
from pathlib import Path
import json

print("="*80)
print("TURN DETECTION EXPERIMENT: Full Run")
print("="*80)

# Ensure results directory exists
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)

#  ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
#  ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 1: TRAINING")
print("="*80)

# Load datasets
general_ds = load_from_disk("data/processed/general")
call_center_ds = load_from_disk("data/processed/call_center")
agent_ds = load_from_disk("data/processed/agent")
customer_ds = load_from_disk("data/processed/customer")

# Model 1: General
print("\n[1/4] Training Model_General...")
trainer_general = TurnDetectionTrainer(output_dir="models/general")
trainer_general.train(
    train_dataset=general_ds['train'],
    eval_dataset=general_ds['validation'],
    num_epochs=3,
    batch_size=16,
    logging_steps=200
)

# Model 2: Domain (Call Center)
print("\n[2/4] Training Model_Domain...")
trainer_domain = TurnDetectionTrainer(output_dir="models/domain")
trainer_domain.train(
    train_dataset=call_center_ds['train'],
    eval_dataset=call_center_ds['validation'],
    num_epochs=3,
    batch_size=16,
    logging_steps=200
)

# Model 3: Agent
print("\n[3/4] Training Model_Agent...")
trainer_agent = TurnDetectionTrainer(output_dir="models/agent")
trainer_agent.train(
    train_dataset=agent_ds['train'],
    eval_dataset=agent_ds['validation'],
    num_epochs=3,
    batch_size=16,
    logging_steps=200
)

# Model 4: Customer
print("\n[4/4] Training Model_Customer...")
trainer_customer = TurnDetectionTrainer(output_dir="models/customer")
trainer_customer.train(
    train_dataset=customer_ds['train'],
    eval_dataset=customer_ds['validation'],
    num_epochs=3,
    batch_size=16,
    logging_steps=200
)

#  ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
#  ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 2: EVALUATION")
print("="*80)

all_results = {}

models_to_eval = [
    ("Model_General", "models/general"),
    ("Model_Domain", "models/domain"),
    ("Model_Agent", "models/agent"),
    ("Model_Customer", "models/customer"),
]

test_sets = [
    ("General Test", general_ds['test']),
    ("CallCenter Test", call_center_ds['test']),
    ("Agent Test", agent_ds['test']),
    ("Customer Test", customer_ds['test']),
]

for model_name, model_path in models_to_eval:
    print(f"\n{'─'*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'─'*80}")
    
    evaluator = ModelEvaluator(model_path)
    all_results[model_name] = {}
    
    for test_name, test_ds in test_sets:
        print(f"\n  {test_name}...")
        metrics = evaluator.evaluate(test_ds)
        all_results[model_name][test_name] = metrics
        
        # Save individual result
        result_file = f"results/metrics/{model_name}_{test_name.replace(' ', '_')}.json"
        with open(result_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    F1: {metrics['f1']:.3f}")

# Save combined results
with open("results/metrics/all_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*80)
print("✓ All models trained and evaluated!")
print("="*80)

#  ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
#  ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("PHASE 3: GENERATING VISUALIZATIONS")
print("="*80)

generate_all_visualizations()

print("\n" + "="*80)
print("✓✓✓ EXPERIMENT COMPLETE! ✓✓✓")
print("="*80)
print("\nResults saved to:")
print("  - results/metrics/")
print("  - results/figures/")
print("\nNext step: Review results and update README.md")


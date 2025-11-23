#!/usr/bin/env python
"""
Threshold Optimization for Turn Detection Models

Find optimal decision thresholds to balance interruptions vs. missed turns.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.threshold_optimizer import ThresholdOptimizer
from src.conversation_processor import ConversationProcessor
import json
import numpy as np

print("="*80)
print("THRESHOLD OPTIMIZATION FOR TURN DETECTION")
print("="*80)

# Load conversations
processor = ConversationProcessor()
call_center_convs = processor.load_conversations(Path("data/call_center_conversations.json"))
general_convs = processor.load_conversations(Path("data/general_conversations.json"))

# Models to optimize
models_to_optimize = [
    ("Domain Normalized (Ch3)", "../models/domain_normalized"),
    ("General Normalized (Ch3)", "../models/general_normalized"),
]

all_optimization_results = {}

for model_name, model_path in models_to_optimize:
    print(f"\n{'='*80}")
    print(f"Optimizing: {model_name}")
    print(f"{'='*80}")
    
    optimizer = ThresholdOptimizer(model_path, model_name)
    
    # Collect predictions on call center data
    print("\n--- Call Center Data ---")
    cc_predictions = optimizer.collect_predictions(call_center_convs)
    
    # Find optimal threshold with different strategies
    print("\n1. Optimizing for F1 Score (balanced)")
    f1_optimal = optimizer.find_optimal_threshold(
        cc_predictions,
        target_metric='f1_score'
    )
    
    print(f"\n✓ F1-Optimal Threshold: {f1_optimal['optimal_threshold']:.3f}")
    print(f"  Interruption Rate: {f1_optimal['optimal_metrics']['interruption_rate']:.1%}")
    print(f"  Missed Turn Rate: {f1_optimal['optimal_metrics']['missed_turn_rate']:.1%}")
    print(f"  F1 Score: {f1_optimal['optimal_metrics']['f1_score']:.3f}")
    
    print("\n2. Optimizing for Minimal Interruptions (5x weight)")
    weighted_optimal = optimizer.find_optimal_threshold(
        cc_predictions,
        target_metric='weighted_error',
        interruption_weight=5.0,  # Interrupting customers is 5x worse
        missed_turn_weight=1.0
    )
    
    print(f"\n✓ Interruption-Minimizing Threshold: {weighted_optimal['optimal_threshold']:.3f}")
    print(f"  Interruption Rate: {weighted_optimal['optimal_metrics']['interruption_rate']:.1%}")
    print(f"  Missed Turn Rate: {weighted_optimal['optimal_metrics']['missed_turn_rate']:.1%}")
    print(f"  F1 Score: {weighted_optimal['optimal_metrics']['f1_score']:.3f}")
    print(f"  Weighted Error: {weighted_optimal['optimal_metrics']['weighted_error']:.3f}")
    
    # Compare with default threshold (0.5)
    default_metrics = optimizer.evaluate_at_threshold(cc_predictions, 0.5)
    print(f"\n3. Default Threshold (0.50) for comparison:")
    print(f"  Interruption Rate: {default_metrics['interruption_rate']:.1%}")
    print(f"  Missed Turn Rate: {default_metrics['missed_turn_rate']:.1%}")
    print(f"  F1 Score: {default_metrics['f1_score']:.3f}")
    
    # Plot threshold curves
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    optimizer.plot_threshold_curves(
        weighted_optimal,
        figures_dir / f"threshold_optimization_{safe_name}_callcenter.png",
        title=f"{model_name} - Call Center Data\nOptimal Threshold: {weighted_optimal['optimal_threshold']:.3f}"
    )
    
    # Store results
    all_optimization_results[model_name] = {
        'call_center': {
            'f1_optimal': f1_optimal['optimal_threshold'],
            'f1_optimal_metrics': f1_optimal['optimal_metrics'],
            'weighted_optimal': weighted_optimal['optimal_threshold'],
            'weighted_optimal_metrics': weighted_optimal['optimal_metrics'],
            'default_metrics': default_metrics,
            'all_thresholds': weighted_optimal['all_results']
        }
    }

# Save optimization results
results_path = Path("results/metrics/threshold_optimization.json")
with open(results_path, 'w') as f:
    json.dump(all_optimization_results, f, indent=2)

print(f"\n✓ Optimization results saved to {results_path}")

# Print summary comparison
print("\n" + "="*80)
print("THRESHOLD OPTIMIZATION SUMMARY")
print("="*80)

print("\n=== Call Center - Default (0.50) vs Optimized Thresholds ===\n")
print(f"{'Model':<30} {'Threshold':<12} {'IR ↓':<10} {'MTR ↓':<10} {'F1 ↑':<10}")
print("-" * 80)

for model_name in all_optimization_results.keys():
    results = all_optimization_results[model_name]['call_center']
    
    # Default
    default = results['default_metrics']
    print(f"{model_name:<30} {'0.500 (def)':<12} {default['interruption_rate']:>8.1%} {default['missed_turn_rate']:>8.1%} {default['f1_score']:>8.3f}")
    
    # Weighted optimal (minimize interruptions)
    opt = results['weighted_optimal_metrics']
    opt_thresh = results['weighted_optimal']
    print(f"{'':<30} {f'{opt_thresh:.3f} (opt)':<12} {opt['interruption_rate']:>8.1%} {opt['missed_turn_rate']:>8.1%} {opt['f1_score']:>8.3f}")
    
    # Show improvement
    ir_improvement = (default['interruption_rate'] - opt['interruption_rate']) / default['interruption_rate'] * 100
    print(f"{'':<30} {'Improvement:':<12} {ir_improvement:>7.1f}%")
    print()

print("\n" + "="*80)
print("KEY INSIGHT: Adjusting threshold can significantly reduce interruptions!")
print("="*80)


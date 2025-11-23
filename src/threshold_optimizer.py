"""
Threshold optimization for turn detection models.

Instead of using default 0.5 threshold for "Complete" prediction,
find optimal threshold to minimize interruptions while maintaining
acceptable missed turn rate.
"""
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from transformers import AutoTokenizer
from src.train import MobileBERTForSequenceClassificationNormalized
from safetensors.torch import load_file
from transformers import AutoConfig
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class ThresholdOptimizer:
    """Find optimal decision threshold for turn detection."""
    
    def __init__(self, model_path: str, model_name: str):
        """Load model and tokenizer."""
        print(f"Loading {model_name} from {model_path}...")
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 2
        self.model = MobileBERTForSequenceClassificationNormalized(config)
        state_dict = load_file(f"{model_path}/model.safetensors")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        self.model_name = model_name
    
    def predict_with_threshold(self, text: str, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Predict if turn is complete using custom threshold.
        
        Returns:
            (is_complete: bool, complete_probability: float)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits'][0]
            probs = torch.softmax(logits, dim=0)
            
            complete_prob = probs[1].item()  # Probability of "Complete"
            is_complete = (complete_prob >= threshold)
        
        return is_complete, complete_prob
    
    def collect_predictions(self, conversations: List[Dict]) -> List[Dict]:
        """
        Collect probability scores for all utterances.
        
        Returns list of dicts with:
            - text: utterance text
            - true_complete: ground truth label
            - complete_prob: model's probability for "Complete"
        """
        print(f"\nCollecting predictions for {len(conversations)} conversations...")
        
        predictions = []
        for conv in tqdm(conversations, desc="Collecting predictions"):
            for turn in conv['turns']:
                _, complete_prob = self.predict_with_threshold(turn['text'], threshold=0.5)
                
                predictions.append({
                    'text': turn['text'],
                    'true_complete': turn['complete'],
                    'complete_prob': complete_prob
                })
        
        return predictions
    
    def evaluate_at_threshold(
        self,
        predictions: List[Dict],
        threshold: float
    ) -> Dict:
        """
        Evaluate metrics at a specific threshold.
        
        Returns:
            - interruption_rate: FP / total utterances
            - missed_turn_rate: FN / total complete utterances
            - f1_score: harmonic mean of precision and recall
            - accuracy: overall accuracy
        """
        total = len(predictions)
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        complete_count = sum(1 for p in predictions if p['true_complete'])
        
        for pred in predictions:
            predicted_complete = (pred['complete_prob'] >= threshold)
            
            if pred['true_complete'] and predicted_complete:
                true_positives += 1
            elif pred['true_complete'] and not predicted_complete:
                false_negatives += 1
            elif not pred['true_complete'] and predicted_complete:
                false_positives += 1
            else:
                true_negatives += 1
        
        # Metrics
        interruption_rate = false_positives / total if total > 0 else 0
        missed_turn_rate = false_negatives / complete_count if complete_count > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        return {
            'threshold': threshold,
            'interruption_rate': interruption_rate,
            'missed_turn_rate': missed_turn_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    
    def find_optimal_threshold(
        self,
        predictions: List[Dict],
        target_metric: str = 'f1_score',
        interruption_weight: float = 1.0,
        missed_turn_weight: float = 1.0,
        thresholds: np.ndarray = None
    ) -> Dict:
        """
        Find optimal threshold by sweeping through possible values.
        
        Args:
            predictions: List of predictions with probabilities
            target_metric: 'f1_score', 'accuracy', or 'weighted_error'
            interruption_weight: How much to penalize interruptions (for weighted_error)
            missed_turn_weight: How much to penalize missed turns (for weighted_error)
            thresholds: Array of thresholds to try (default: 0.1 to 0.9 in 0.01 steps)
        
        Returns:
            Dict with optimal threshold and metrics at all thresholds
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.01)
        
        print(f"\nSearching for optimal threshold...")
        print(f"  Target metric: {target_metric}")
        if target_metric == 'weighted_error':
            print(f"  Interruption weight: {interruption_weight}")
            print(f"  Missed turn weight: {missed_turn_weight}")
        
        results = []
        for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
            metrics = self.evaluate_at_threshold(predictions, threshold)
            
            # Calculate weighted error if that's the target
            if target_metric == 'weighted_error':
                weighted_error = (
                    metrics['interruption_rate'] * interruption_weight +
                    metrics['missed_turn_rate'] * missed_turn_weight
                )
                metrics['weighted_error'] = weighted_error
            
            results.append(metrics)
        
        # Find best threshold
        if target_metric == 'weighted_error':
            best_result = min(results, key=lambda x: x['weighted_error'])
        elif target_metric == 'f1_score':
            best_result = max(results, key=lambda x: x['f1_score'])
        elif target_metric == 'accuracy':
            best_result = max(results, key=lambda x: x['accuracy'])
        else:
            raise ValueError(f"Unknown target metric: {target_metric}")
        
        return {
            'optimal_threshold': best_result['threshold'],
            'optimal_metrics': best_result,
            'all_results': results
        }
    
    def plot_threshold_curves(
        self,
        optimization_results: Dict,
        output_path: Path,
        title: str = None
    ):
        """
        Plot metrics vs. threshold to visualize trade-offs.
        
        Creates:
        1. Interruption Rate vs. Threshold
        2. Missed Turn Rate vs. Threshold
        3. F1 Score vs. Threshold
        4. ROC-like curve: IR vs MTR
        """
        results = optimization_results['all_results']
        optimal = optimization_results['optimal_threshold']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        thresholds = [r['threshold'] for r in results]
        irs = [r['interruption_rate'] for r in results]
        mtrs = [r['missed_turn_rate'] for r in results]
        f1s = [r['f1_score'] for r in results]
        
        # Plot 1: Interruption Rate vs Threshold
        axes[0, 0].plot(thresholds, irs, 'b-', linewidth=2)
        axes[0, 0].axvline(optimal, color='r', linestyle='--', label=f'Optimal: {optimal:.2f}')
        axes[0, 0].set_xlabel('Threshold', fontsize=12)
        axes[0, 0].set_ylabel('Interruption Rate', fontsize=12)
        axes[0, 0].set_title('Interruption Rate vs. Threshold', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Missed Turn Rate vs Threshold
        axes[0, 1].plot(thresholds, mtrs, 'g-', linewidth=2)
        axes[0, 1].axvline(optimal, color='r', linestyle='--', label=f'Optimal: {optimal:.2f}')
        axes[0, 1].set_xlabel('Threshold', fontsize=12)
        axes[0, 1].set_ylabel('Missed Turn Rate', fontsize=12)
        axes[0, 1].set_title('Missed Turn Rate vs. Threshold', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: F1 Score vs Threshold
        axes[1, 0].plot(thresholds, f1s, 'purple', linewidth=2)
        axes[1, 0].axvline(optimal, color='r', linestyle='--', label=f'Optimal: {optimal:.2f}')
        axes[1, 0].set_xlabel('Threshold', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_title('F1 Score vs. Threshold', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Trade-off curve (IR vs MTR)
        axes[1, 1].plot(irs, mtrs, 'o-', markersize=3, linewidth=1, alpha=0.6)
        
        # Mark optimal point
        optimal_metrics = optimization_results['optimal_metrics']
        axes[1, 1].plot(
            optimal_metrics['interruption_rate'],
            optimal_metrics['missed_turn_rate'],
            'r*', markersize=20, label=f'Optimal (θ={optimal:.2f})'
        )
        
        # Mark default threshold (0.5)
        default_result = [r for r in results if abs(r['threshold'] - 0.5) < 0.01][0]
        axes[1, 1].plot(
            default_result['interruption_rate'],
            default_result['missed_turn_rate'],
            'bs', markersize=10, label='Default (θ=0.50)'
        )
        
        axes[1, 1].set_xlabel('Interruption Rate', fontsize=12)
        axes[1, 1].set_ylabel('Missed Turn Rate', fontsize=12)
        axes[1, 1].set_title('Error Trade-off Curve', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved threshold curves to {output_path}")


"""
Comprehensive evaluation script for turn detection models.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
from safetensors.torch import load_file

from src.utils import save_metrics, print_metrics_summary, measure_inference_latency
from src.train import MobileBERTForSequenceClassificationNormalized


class ModelEvaluator:
    """Evaluator for turn detection models."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load custom MobileBERT model
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 2
        self.model = MobileBERTForSequenceClassificationNormalized(config)
        
        # Load saved weights
        state_dict = load_file(f"{model_path}/model.safetensors")
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(device)
        self.model.eval()
    
    def predict(self, texts: List[str], return_probs: bool = False) -> Tuple:
        """
        Make predictions on a list of texts.
        
        Returns:
            predictions, probabilities (if return_probs=True)
        """
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Predicting"):
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)
                
                predictions.append(pred.item())
                if return_probs:
                    # Get probability for class 1 (Complete)
                    probabilities.append(probs[0][1].item())
        
        if return_probs:
            return np.array(predictions), np.array(probabilities)
        return np.array(predictions)
    
    def evaluate(self, dataset, dataset_name: str = "test") -> Dict:
        """
        Evaluate model on a dataset.
        
        Returns:
            Dictionary with metrics
        """
        print(f"\nEvaluating on {dataset_name} set ({len(dataset)} examples)...")
        
        texts = dataset['text']
        labels = np.array(dataset['label'])
        
        # Get predictions with probabilities
        predictions, probabilities = self.predict(texts, return_probs=True)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        # Also get per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(labels, predictions, average=None)
        
        cm = confusion_matrix(labels, predictions)
        
        # Calculate false positive and false negative rates
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics = {
            'dataset': dataset_name,
            'num_examples': len(dataset),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'precision_incomplete': float(precision_per_class[0]),
            'precision_complete': float(precision_per_class[1]),
            'recall_incomplete': float(recall_per_class[0]),
            'recall_complete': float(recall_per_class[1]),
            'f1_incomplete': float(f1_per_class[0]),
            'f1_complete': float(f1_per_class[1]),
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'support_incomplete': int(support_per_class[0]),
            'support_complete': int(support_per_class[1]),
            # Store raw data for visualization
            'y_true': labels.tolist(),
            'y_pred': predictions.tolist(),
            'y_pred_proba': probabilities.tolist()
        }
        
        return metrics
    
    def measure_latency(self, texts: List[str], n_runs: int = 100) -> Dict:
        """Measure inference latency on CPU."""
        print(f"\nMeasuring inference latency ({n_runs} runs)...")
        
        return measure_inference_latency(
            self.model,
            self.tokenizer,
            texts,
            device=self.device,
            n_runs=n_runs
        )


def evaluate_all_models():
    """Evaluate all trained models on both test sets."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Define models to evaluate
    models = {
        'General': 'models/general',
        'Domain': 'models/domain',
        'Agent': 'models/agent',
        'Customer': 'models/customer'
    }
    
    # Define test sets
    test_sets = {
        'General': 'data/processed/general',
        'CallCenter': 'data/processed/call_center',
        'Agent': 'data/processed/agent',
        'Customer': 'data/processed/customer'
    }
    
    all_results = {}
    
    for model_name, model_path in models.items():
        if not Path(model_path).exists():
            print(f"\n⚠ Skipping {model_name} - model not found at {model_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name} Model")
        print(f"{'='*70}")
        
        try:
            evaluator = ModelEvaluator(model_path, device="cpu")
            
            model_results = {}
            
            # Evaluate on all available test sets
            for test_name, test_path in test_sets.items():
                if not Path(test_path).exists():
                    continue
                
                try:
                    dataset = load_from_disk(test_path)
                    test_data = dataset['test']
                    
                    # Evaluate
                    metrics = evaluator.evaluate(test_data, dataset_name=test_name)
                    
                    # Measure latency (use sample texts from test set)
                    sample_texts = test_data['text'][:100]
                    latency_metrics = evaluator.measure_latency(sample_texts, n_runs=100)
                    
                    # Combine metrics
                    result_key = f"{model_name}_on_{test_name}"
                    model_results[result_key] = {
                        **metrics,
                        'latency': latency_metrics
                    }
                    
                    # Print summary
                    print(f"\n{test_name} Test Set:")
                    print(f"  Accuracy: {metrics['accuracy']:.4f}")
                    print(f"  F1 Score: {metrics['f1']:.4f}")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall: {metrics['recall']:.4f}")
                    print(f"  FP Rate: {metrics['false_positive_rate']:.4f}")
                    print(f"  FN Rate: {metrics['false_negative_rate']:.4f}")
                    print(f"  Latency: {latency_metrics['mean_ms']:.2f}ms (±{latency_metrics['std_ms']:.2f}ms)")
                    
                except Exception as e:
                    print(f"  Error evaluating on {test_name}: {e}")
            
            # Save model results
            if model_results:
                # Save primary result (model evaluated on its intended test set)
                primary_test = 'General' if model_name == 'General' else 'CallCenter'
                if model_name in ['Agent', 'Customer']:
                    primary_test = model_name
                
                primary_key = f"{model_name}_on_{primary_test}"
                if primary_key in model_results:
                    all_results[model_name] = model_results[primary_key]
                    save_metrics(
                        model_results[primary_key],
                        f"{model_name.lower()}_metrics.json"
                    )
                
                # Save all cross-evaluations
                save_metrics(
                    model_results,
                    f"{model_name.lower()}_all_evaluations.json"
                )
        
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    
    # Save combined results
    if all_results:
        save_metrics(all_results, "combined_results.json")
        
        # Print final comparison
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        
        print(f"\n{'Model':<15} {'Accuracy':<12} {'F1 Score':<12} {'Latency (ms)':<15}")
        print("-" * 70)
        
        for model_name, results in all_results.items():
            acc = results['accuracy']
            f1 = results['f1']
            lat = results['latency']['mean_ms']
            print(f"{model_name:<15} {acc:<12.4f} {f1:<12.4f} {lat:<15.2f}")
        
        print("="*70)
    
    return all_results


def analyze_errors(model_path: str, dataset_path: str, num_examples: int = 20):
    """Analyze model errors and extract examples."""
    
    print(f"\n{'='*70}")
    print(f"Error Analysis: {Path(model_path).name}")
    print(f"{'='*70}")
    
    evaluator = ModelEvaluator(model_path, device="cpu")
    dataset = load_from_disk(dataset_path)
    test_data = dataset['test']
    
    texts = test_data['text']
    labels = np.array(test_data['label'])
    predictions = evaluator.predict(texts)
    
    # Find errors
    errors = []
    for i, (text, label, pred) in enumerate(zip(texts, labels, predictions)):
        if label != pred:
            error_type = "False Positive" if pred == 1 else "False Negative"
            errors.append({
                'text': text,
                'true_label': 'Complete' if label == 1 else 'Incomplete',
                'predicted_label': 'Complete' if pred == 1 else 'Incomplete',
                'error_type': error_type
            })
    
    print(f"\nTotal errors: {len(errors)} / {len(texts)} ({len(errors)/len(texts)*100:.2f}%)")
    
    # Sample errors
    if errors:
        print(f"\nSample errors (showing {min(num_examples, len(errors))}):")
        print("-" * 70)
        
        for i, error in enumerate(errors[:num_examples]):
            print(f"\n{i+1}. {error['error_type']}:")
            print(f"   Text: {error['text']}")
            print(f"   True: {error['true_label']} | Predicted: {error['predicted_label']}")
    
    # Save error analysis
    error_analysis = {
        'total_examples': len(texts),
        'total_errors': len(errors),
        'error_rate': len(errors) / len(texts),
        'sample_errors': errors[:100]  # Save first 100 errors
    }
    
    model_name = Path(model_path).name
    save_metrics(error_analysis, f"{model_name}_error_analysis.json")
    
    return error_analysis


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate turn detection models")
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        help='Model to evaluate (or "all" for all models)'
    )
    parser.add_argument(
        '--error-analysis',
        action='store_true',
        help='Perform error analysis'
    )
    
    args = parser.parse_args()
    
    # Run comprehensive evaluation
    results = evaluate_all_models()
    
    # Error analysis if requested
    if args.error_analysis:
        print("\n" + "="*70)
        print("PERFORMING ERROR ANALYSIS")
        print("="*70)
        
        models = {
            'general': 'data/processed/general',
            'domain': 'data/processed/call_center',
            'agent': 'data/processed/agent',
            'customer': 'data/processed/customer'
        }
        
        for model_name, dataset_path in models.items():
            model_path = f"models/{model_name}"
            
            if Path(model_path).exists() and Path(dataset_path).exists():
                analyze_errors(model_path, dataset_path, num_examples=10)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()



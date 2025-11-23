#!/usr/bin/env python
"""
Punctuation Robustness Test for Turn Detection Models

Tests whether our models rely on punctuation cues or have learned 
semantic understanding of turn completion.

Compares predictions on:
1. Original text (with punctuation)
2. Normalized text (punctuation removed, lowercased)
"""
import sys
sys.path.insert(0, '../')

from src.train import MobileBERTForSequenceClassificationNormalized
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import torch
from datasets import load_from_disk
import re
from pathlib import Path
from typing import Dict, List, Tuple

def normalize_text(text: str) -> str:
    """Remove all punctuation and lowercase."""
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

class TurnDetectorTester:
    """Test turn detector with and without punctuation."""
    
    def __init__(self, model_path: str, model_name: str):
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 2
        self.model = MobileBERTForSequenceClassificationNormalized(config)
        state_dict = load_file(f"{model_path}/model.safetensors")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        self.name = model_name
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Run prediction and return results.
        Returns: {prediction: str, confidence: float, logits: list}
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
            
            prediction = torch.argmax(logits).item()
            confidence = probs[prediction].item()
            pred_text = "Complete" if prediction == 1 else "Incomplete"
        
        return {
            'prediction': pred_text,
            'confidence': confidence,
            'logits': logits.tolist(),
            'complete_prob': probs[1].item(),
            'incomplete_prob': probs[0].item()
        }
    
    def test_robustness(self, text: str, true_label: str) -> Dict:
        """Test same text with and without punctuation."""
        original_result = self.predict(text)
        normalized_result = self.predict(normalize_text(text))
        
        return {
            'original_text': text,
            'normalized_text': normalize_text(text),
            'true_label': true_label,
            'original': original_result,
            'normalized': normalized_result,
            'prediction_changed': original_result['prediction'] != normalized_result['prediction'],
            'confidence_delta': abs(original_result['confidence'] - normalized_result['confidence'])
        }

def format_result_table(example: Dict, general: Dict, domain: Dict) -> str:
    """Format a single example's results as markdown."""
    
    md = f"\n### Example: \"{example['text']}\"\n\n"
    md += f"**True Label: {example['label']}**\n\n"
    
    # Table comparing original vs normalized for both models
    md += "| Text Version | General Prediction | General Confidence | Domain Prediction | Domain Confidence |\n"
    md += "|--------------|-------------------|-------------------|-------------------|-------------------|\n"
    
    # Original text row
    md += f"| **Original** (with punctuation) | "
    md += f"{general['original']['prediction']} | "
    md += f"{general['original']['confidence']:.1%} | "
    md += f"{domain['original']['prediction']} | "
    md += f"{domain['original']['confidence']:.1%} |\n"
    
    # Normalized text row
    md += f"| **Normalized** (no punctuation) | "
    md += f"{general['normalized']['prediction']} | "
    md += f"{general['normalized']['confidence']:.1%} | "
    md += f"{domain['normalized']['prediction']} | "
    md += f"{domain['normalized']['confidence']:.1%} |\n"
    
    md += "\n"
    
    # Show the actual normalized text
    md += f"**Normalized text:** `{general['normalized_text']}`\n\n"
    
    # Analysis
    if general['prediction_changed'] or domain['prediction_changed']:
        md += "⚠️ **Prediction changed when punctuation removed!**\n"
        if general['prediction_changed']:
            md += f"- General model flipped from {general['original']['prediction']} → {general['normalized']['prediction']}\n"
        if domain['prediction_changed']:
            md += f"- Domain model flipped from {domain['original']['prediction']} → {domain['normalized']['prediction']}\n"
    else:
        md += "✓ Both models maintained the same prediction\n"
    
    # Confidence changes
    if general['confidence_delta'] > 0.1 or domain['confidence_delta'] > 0.1:
        md += f"\n**Confidence Impact:**\n"
        if general['confidence_delta'] > 0.1:
            md += f"- General: {general['confidence_delta']:.1%} change\n"
        if domain['confidence_delta'] > 0.1:
            md += f"- Domain: {domain['confidence_delta']:.1%} change\n"
    
    md += "\n---\n"
    return md

def main():
    print("="*80)
    print("PUNCTUATION ROBUSTNESS TEST")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    tester_general = TurnDetectorTester("../models/general", "General")
    tester_domain = TurnDetectorTester("../models/domain", "Domain")
    
    # Load test data
    print("Loading test data...")
    call_center_test = load_from_disk("../data/processed/call_center")['test']
    
    # Select examples that have punctuation
    examples_with_punctuation = []
    for item in call_center_test:
        text = item['text']
        # Only include examples that have punctuation to test
        if any(char in text for char in '.,!?;:'):
            label = "Complete" if item['label'] == 1 else "Incomplete"
            examples_with_punctuation.append({
                'text': text,
                'label': label
            })
    
    print(f"Found {len(examples_with_punctuation)} examples with punctuation")
    
    # Test a diverse set: some complete, some incomplete, various punctuation
    test_examples = []
    
    # Find examples with different punctuation types
    for example in examples_with_punctuation[:100]:  # Check first 100
        text = example['text']
        if '?' in text and len(test_examples) < 3:
            test_examples.append(example)
        elif '.' in text and example['label'] == 'Complete' and len([e for e in test_examples if '.' in e['text']]) < 3:
            test_examples.append(example)
        elif example['label'] == 'Incomplete' and len([e for e in test_examples if e['label'] == 'Incomplete']) < 2:
            test_examples.append(example)
        
        if len(test_examples) >= 6:
            break
    
    # Run tests
    print(f"\nTesting {len(test_examples)} examples...")
    
    output_md = "# Punctuation Robustness Test Results\n\n"
    output_md += "This test investigates whether our turn detection models rely on punctuation cues "
    output_md += "or have learned semantic understanding of turn completion.\n\n"
    output_md += "For each example, we compare predictions on:\n"
    output_md += "1. **Original text** - as it appears in the dataset (with punctuation)\n"
    output_md += "2. **Normalized text** - all punctuation removed, lowercased\n\n"
    
    # Track statistics
    general_flips = 0
    domain_flips = 0
    general_confidence_drops = []
    domain_confidence_drops = []
    
    for example in test_examples:
        print(f"Testing: {example['text']}")
        
        general_results = tester_general.test_robustness(example['text'], example['label'])
        domain_results = tester_domain.test_robustness(example['text'], example['label'])
        
        output_md += format_result_table(example, general_results, domain_results)
        
        # Track stats
        if general_results['prediction_changed']:
            general_flips += 1
        if domain_results['prediction_changed']:
            domain_flips += 1
        
        general_confidence_drops.append(general_results['confidence_delta'])
        domain_confidence_drops.append(domain_results['confidence_delta'])
    
    # Summary statistics
    output_md += "\n## Summary\n\n"
    output_md += f"**Prediction Changes (when punctuation removed):**\n"
    output_md += f"- General Model: {general_flips}/{len(test_examples)} examples changed ({general_flips/len(test_examples)*100:.1f}%)\n"
    output_md += f"- Domain Model: {domain_flips}/{len(test_examples)} examples changed ({domain_flips/len(test_examples)*100:.1f}%)\n\n"
    
    avg_general_drop = sum(general_confidence_drops) / len(general_confidence_drops)
    avg_domain_drop = sum(domain_confidence_drops) / len(domain_confidence_drops)
    
    output_md += f"**Average Confidence Change:**\n"
    output_md += f"- General Model: {avg_general_drop:.1%}\n"
    output_md += f"- Domain Model: {avg_domain_drop:.1%}\n\n"
    
    # Interpretation
    output_md += "## Interpretation\n\n"
    
    if general_flips > len(test_examples) * 0.3 or domain_flips > len(test_examples) * 0.3:
        output_md += "⚠️ **High Punctuation Dependency Detected**\n\n"
        output_md += "The models show significant reliance on punctuation for making predictions. "
        output_md += "When punctuation is removed, predictions change frequently, indicating that the models "
        output_md += "have learned punctuation patterns rather than semantic understanding of turn completion.\n\n"
        output_md += "**Implication:** These models may perform poorly with ASR systems that don't reliably "
        output_md += "predict punctuation, or in streaming scenarios where punctuation arrives late.\n"
    else:
        output_md += "✓ **Strong Semantic Understanding**\n\n"
        output_md += "The models maintain stable predictions even without punctuation, suggesting they have "
        output_md += "learned semantic patterns of turn completion beyond simple punctuation cues.\n\n"
        output_md += "**Implication:** These models should be robust to ASR variations and streaming scenarios "
        output_md += "where punctuation may not be available.\n"
    
    # Save results
    output_path = Path("punctuation_robustness_results.md")
    output_path.write_text(output_md)
    print(f"\n✓ Results saved to {output_path}")
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"General Model: {general_flips}/{len(test_examples)} predictions changed")
    print(f"Domain Model: {domain_flips}/{len(test_examples)} predictions changed")
    print(f"Average confidence drops: General={avg_general_drop:.1%}, Domain={avg_domain_drop:.1%}")

if __name__ == "__main__":
    main()


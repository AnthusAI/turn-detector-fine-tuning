#!/usr/bin/env python
"""
Streaming Inference Analysis for Turn Detection
Demonstrates how models behave in a real-time streaming scenario.
"""
import sys
sys.path.insert(0, '../')

from src.train import MobileBERTForSequenceClassificationNormalized
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file
import torch
from datasets import load_from_disk
import json
from pathlib import Path
from typing import List, Tuple

class StreamingTurnDetector:
    """Run predictions on a sentence as words stream in."""
    
    def __init__(self, model_path: str):
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 2
        self.model = MobileBERTForSequenceClassificationNormalized(config)
        state_dict = load_file(f"{model_path}/model.safetensors")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    
    def predict_streaming(self, text: str) -> List[Tuple[str, float, str]]:
        """
        Predict as words stream in.
        Returns: [(prefix, confidence, prediction), ...]
        """
        words = text.split()
        results = []
        
        for i in range(1, len(words) + 1):
            prefix = " ".join(words[:i])
            
            inputs = self.tokenizer(
                prefix,
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
            
            results.append((prefix, confidence, pred_text))
        
        return results

def generate_markdown_table(example_text: str, detector_general, detector_domain) -> str:
    """Generate a markdown table showing streaming inference."""
    
    print(f"\nProcessing: '{example_text}'")
    
    general_results = detector_general.predict_streaming(example_text)
    domain_results = detector_domain.predict_streaming(example_text)
    
    # Build markdown table
    md = f"""
### Example: "{example_text}"

| Word(s) | General Model | General Confidence | Domain Model | Domain Confidence |
|---------|---------------|-------------------|--------------|-------------------|
"""
    
    words = example_text.split()
    for i in range(len(words)):
        prefix = " ".join(words[:i+1])
        gen_pred = general_results[i][2]
        gen_conf = general_results[i][1]
        dom_pred = domain_results[i][2]
        dom_conf = domain_results[i][1]
        
        # Show full prefix - no truncation
        md += f"| {prefix} | {gen_pred} | {gen_conf:.1%} | {dom_pred} | {dom_conf:.1%} |\n"
    
    return md

def main():
    print("="*80)
    print("CHAPTER 2: STREAMING TURN DETECTION ANALYSIS")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    detector_general = StreamingTurnDetector("../models/general")
    detector_domain = StreamingTurnDetector("../models/domain")
    
    # Load evaluation data to sample from
    print("Loading evaluation data...")
    call_center_test = load_from_disk("../data/processed/call_center")['test']
    
    # Sample a few interesting examples
    examples = []
    for i in [0, 5, 10, 15]:  # Sample a few
        if i < len(call_center_test):
            text = call_center_test[i]['text']
            label = call_center_test[i]['label']
            examples.append((text, "Complete" if label == 1 else "Incomplete"))
    
    # Generate markdown tables
    output_md = "# Streaming Inference Examples\n\n"
    output_md += "These examples show how the General and Domain-specific models make predictions as words stream in.\n\n"
    
    for text, true_label in examples:
        output_md += f"**True Label: {true_label}**\n"
        output_md += generate_markdown_table(text, detector_general, detector_domain)
        output_md += "\n---\n"
    
    # Save output
    output_path = Path("streaming_examples.md")
    output_path.write_text(output_md)
    print(f"\nResults saved to {output_path}")
    print(output_md)

if __name__ == "__main__":
    main()


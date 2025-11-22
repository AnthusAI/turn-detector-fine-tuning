"""
Turn Detection Inference Class

A simple interface for using trained turn detection models.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import time
from typing import Dict, Optional, Literal


class TurnDetector:
    """
    Turn detection model with channel-aware routing.
    
    Usage:
        # General model
        detector = TurnDetector(model_type='general')
        result = detector.predict("How can I help you today")
        
        # Domain-specific model
        detector = TurnDetector(model_type='domain')
        result = detector.predict("Please hold while I transfer you")
        
        # Channel-specific model
        detector = TurnDetector(model_type='channel_specific', channel='agent')
        result = detector.predict("Thank you for calling")
    """
    
    def __init__(
        self,
        model_type: Literal['general', 'domain', 'channel_specific'] = 'domain',
        channel: Optional[Literal['agent', 'customer']] = None,
        device: str = "cpu",
        models_dir: str = "models"
    ):
        """
        Initialize the turn detector.
        
        Args:
            model_type: Type of model to use
                - 'general': Trained on general conversation data
                - 'domain': Fine-tuned on call center data (recommended)
                - 'channel_specific': Use agent or customer-specific model
            channel: Required if model_type='channel_specific'
                - 'agent': Use agent-specific model
                - 'customer': Use customer-specific model
            device: Device to run inference on ('cpu' or 'cuda')
            models_dir: Base directory containing model checkpoints
        """
        self.model_type = model_type
        self.channel = channel
        self.device = device
        self.models_dir = Path(models_dir)
        
        # Determine model path
        if model_type == 'channel_specific':
            if channel not in ['agent', 'customer']:
                raise ValueError(
                    f"channel must be 'agent' or 'customer' when model_type='channel_specific', "
                    f"got: {channel}"
                )
            model_path = self.models_dir / channel
        else:
            model_path = self.models_dir / model_type
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please train the model first using src/train.py"
            )
        
        # Load model and tokenizer
        print(f"Loading {model_type} model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ Turn detector ready (model_type={model_type}, channel={channel})")
    
    def predict(self, text: str, return_confidence: bool = True) -> Dict:
        """
        Predict whether an utterance is complete or incomplete.
        
        Args:
            text: The utterance text to classify
            return_confidence: Whether to include confidence scores
            
        Returns:
            Dictionary with:
                - is_complete: Boolean indicating if turn is complete
                - label: String label ('Complete' or 'Incomplete')
                - confidence: Confidence score (if return_confidence=True)
                - inference_time_ms: Inference time in milliseconds
        """
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        result = {
            'is_complete': bool(prediction == 1),
            'label': 'Complete' if prediction == 1 else 'Incomplete',
            'inference_time_ms': inference_time_ms
        }
        
        if return_confidence:
            result['confidence'] = confidence
            result['confidence_complete'] = probs[0][1].item()
            result['confidence_incomplete'] = probs[0][0].item()
        
        return result
    
    def predict_batch(self, texts: list[str]) -> list[Dict]:
        """
        Predict on a batch of texts.
        
        Args:
            texts: List of utterance texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def demo():
    """Demonstration of the TurnDetector class."""
    
    print("\n" + "="*70)
    print("Turn Detector Demo")
    print("="*70)
    
    # Example utterances
    test_utterances = [
        "How can I help you today",  # Incomplete (agent greeting)
        "I need to speak with a manager",  # Complete (customer request)
        "Thank you for",  # Incomplete (truncated)
        "Your account has been updated successfully",  # Complete
        "Please hold while I",  # Incomplete (truncated)
        "I would like to cancel my subscription",  # Complete
    ]
    
    # Try different models
    models_to_try = [
        ('general', None),
        ('domain', None),
    ]
    
    # Add channel-specific if available
    if (Path("models/agent").exists()):
        models_to_try.append(('channel_specific', 'agent'))
    if (Path("models/customer").exists()):
        models_to_try.append(('channel_specific', 'customer'))
    
    for model_type, channel in models_to_try:
        try:
            print(f"\n{'='*70}")
            if channel:
                print(f"Model: {model_type} ({channel})")
            else:
                print(f"Model: {model_type}")
            print(f"{'='*70}")
            
            detector = TurnDetector(model_type=model_type, channel=channel)
            
            for utterance in test_utterances:
                result = detector.predict(utterance)
                print(f"\nText: {utterance}")
                print(f"  Prediction: {result['label']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Latency: {result['inference_time_ms']:.2f}ms")
        
        except FileNotFoundError:
            print(f"\n⚠ Skipping {model_type} - model not found")
            continue
    
    print("\n" + "="*70)


if __name__ == "__main__":
    demo()



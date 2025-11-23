"""
Conversation-level evaluation metrics for turn detection models.
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


class ConversationEvaluator:
    """Evaluate turn detection models on conversation sequences."""
    
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
    
    def predict_turn(self, text: str) -> Tuple[bool, float]:
        """
        Predict if a turn is complete.
        
        Returns:
            (is_complete: bool, confidence: float)
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
            
            is_complete = (prediction == 1)
        
        return is_complete, confidence
    
    def simulate_streaming_detection(self, text: str, true_complete: bool) -> Dict:
        """
        Simulate word-by-word streaming and detect when model first predicts 'Complete'.
        
        Returns dict with:
            - detection_point: word index when first predicted Complete (None if never)
            - total_words: total words in utterance
            - correct: did final prediction match true label?
            - premature: did it predict Complete before end?
            - detection_latency: words after completion before detection (0 if correct, negative if premature)
        """
        words = text.split()
        total_words = len(words)
        detection_point = None
        
        # Feed words incrementally
        for i in range(1, len(words) + 1):
            partial_text = " ".join(words[:i])
            is_complete, confidence = self.predict_turn(partial_text)
            
            if is_complete and detection_point is None:
                detection_point = i
                break  # First time it predicts Complete
        
        # Final prediction (all words)
        final_complete, final_conf = self.predict_turn(text)
        
        if true_complete:
            # Utterance is actually complete
            if detection_point is None:
                # Never predicted Complete (False Negative)
                return {
                    'detection_point': None,
                    'total_words': total_words,
                    'correct': False,
                    'premature': False,
                    'detection_latency': None,  # Missed turn
                    'error_type': 'false_negative'
                }
            elif detection_point == total_words:
                # Predicted Complete at the end (True Positive, no latency)
                return {
                    'detection_point': detection_point,
                    'total_words': total_words,
                    'correct': True,
                    'premature': False,
                    'detection_latency': 0,
                    'error_type': None
                }
            else:
                # Predicted Complete before the end (True Positive, but early)
                return {
                    'detection_point': detection_point,
                    'total_words': total_words,
                    'correct': True,
                    'premature': True,
                    'detection_latency': detection_point - total_words,  # Negative = early
                    'error_type': None
                }
        else:
            # Utterance is actually incomplete
            if detection_point is None:
                # Never predicted Complete (True Negative)
                return {
                    'detection_point': None,
                    'total_words': total_words,
                    'correct': True,
                    'premature': False,
                    'detection_latency': 0,
                    'error_type': None
                }
            else:
                # Predicted Complete but shouldn't have (False Positive = Interruption)
                return {
                    'detection_point': detection_point,
                    'total_words': total_words,
                    'correct': False,
                    'premature': True,
                    'detection_latency': detection_point - total_words,  # How early was the interruption
                    'error_type': 'false_positive'
                }
    
    def evaluate_conversation(self, conversation: Dict) -> Dict:
        """
        Evaluate model on a single conversation.
        
        Returns metrics for this conversation:
            - interruptions: count of false positives
            - missed_turns: count of false negatives
            - total_turns: total utterances
            - correct_predictions: count of correct predictions
            - turn_results: list of per-turn results
        """
        turn_results = []
        interruptions = 0
        missed_turns = 0
        correct_predictions = 0
        
        for turn in conversation['turns']:
            result = self.simulate_streaming_detection(
                turn['text'],
                turn['complete']
            )
            result['true_label'] = 'Complete' if turn['complete'] else 'Incomplete'
            result['text'] = turn['text']
            turn_results.append(result)
            
            if result['error_type'] == 'false_positive':
                interruptions += 1
            elif result['error_type'] == 'false_negative':
                missed_turns += 1
            else:
                correct_predictions += 1
        
        return {
            'conversation_id': conversation['id'],
            'domain': conversation['domain'],
            'total_turns': len(turn_results),
            'interruptions': interruptions,
            'missed_turns': missed_turns,
            'correct_predictions': correct_predictions,
            'turn_results': turn_results
        }
    
    def evaluate_conversations(self, conversations: List[Dict]) -> Dict:
        """
        Evaluate model on multiple conversations.
        
        Returns aggregated metrics across all conversations.
        """
        print(f"\nEvaluating {self.model_name} on {len(conversations)} conversations...")
        
        conversation_results = []
        for conv in tqdm(conversations, desc=f"Evaluating {self.model_name}"):
            result = self.evaluate_conversation(conv)
            conversation_results.append(result)
        
        # Aggregate metrics
        metrics = self.compute_metrics(conversation_results)
        
        return {
            'model_name': self.model_name,
            'num_conversations': len(conversations),
            'metrics': metrics,
            'conversation_results': conversation_results
        }
    
    def compute_metrics(self, conversation_results: List[Dict]) -> Dict:
        """
        Compute conversation-level metrics from individual conversation results.
        """
        # Aggregate counts
        total_conversations = len(conversation_results)
        total_turns = sum(r['total_turns'] for r in conversation_results)
        total_interruptions = sum(r['interruptions'] for r in conversation_results)
        total_missed_turns = sum(r['missed_turns'] for r in conversation_results)
        total_correct = sum(r['correct_predictions'] for r in conversation_results)
        
        # Count complete vs incomplete turns
        complete_turns = 0
        incomplete_turns = 0
        for conv_result in conversation_results:
            for turn_result in conv_result['turn_results']:
                if turn_result['true_label'] == 'Complete':
                    complete_turns += 1
                else:
                    incomplete_turns += 1
        
        # 1. Interruption Rate (IR)
        interruption_rate = total_interruptions / total_turns if total_turns > 0 else 0
        
        # 2. Missed Turn Rate (MTR)
        missed_turn_rate = total_missed_turns / complete_turns if complete_turns > 0 else 0
        
        # 3. Conversation F1
        # Precision = TP / (TP + FP), Recall = TP / (TP + FN)
        true_positives = total_correct - sum(
            1 for r in conversation_results
            for t in r['turn_results']
            if t['error_type'] is None and t['true_label'] == 'Incomplete'
        )
        false_positives = total_interruptions
        false_negatives = total_missed_turns
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 4. Perfect Conversation Rate (PCR)
        perfect_conversations = sum(
            1 for r in conversation_results
            if r['interruptions'] == 0 and r['missed_turns'] == 0
        )
        perfect_conversation_rate = perfect_conversations / total_conversations if total_conversations > 0 else 0
        
        # 5. Mean Words to Detection (MWD) - for correctly detected complete turns
        detection_latencies = []
        for conv_result in conversation_results:
            for turn_result in conv_result['turn_results']:
                if turn_result['true_label'] == 'Complete' and turn_result['detection_latency'] is not None:
                    if turn_result['detection_latency'] >= 0:  # Only positive latencies (late detection)
                        detection_latencies.append(turn_result['detection_latency'])
        
        mean_words_to_detection = np.mean(detection_latencies) if detection_latencies else 0
        
        # 6. Premature Detection Depth (PDD) - for interruptions
        premature_depths = []
        for conv_result in conversation_results:
            for turn_result in conv_result['turn_results']:
                if turn_result['error_type'] == 'false_positive' and turn_result['detection_point']:
                    depth = turn_result['detection_point'] / turn_result['total_words']
                    premature_depths.append(depth)
        
        premature_detection_depth = np.mean(premature_depths) if premature_depths else 0
        
        return {
            'total_conversations': total_conversations,
            'total_turns': total_turns,
            'complete_turns': complete_turns,
            'incomplete_turns': incomplete_turns,
            'interruption_rate': interruption_rate,
            'missed_turn_rate': missed_turn_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'perfect_conversation_rate': perfect_conversation_rate,
            'mean_words_to_detection': mean_words_to_detection,
            'premature_detection_depth': premature_detection_depth,
            'total_interruptions': total_interruptions,
            'total_missed_turns': total_missed_turns,
            'total_correct': total_correct
        }


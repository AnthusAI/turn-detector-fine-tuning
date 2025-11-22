"""
Data processing and curation with the "One-Shot Sentence" rule.

The core principle: Each unique sentence appears exactly ONCE in the dataset,
either as Complete OR Incomplete, never both. This prevents overfitting to
specific sentence patterns.
"""
import random
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm


class OneShotDataProcessor:
    """
    Implements the "One-Shot Sentence" rule for turn detection data curation.
    
    Each unique sentence appears exactly once as either:
    - Complete (label=1): Full sentence
    - Incomplete (label=0): Randomly truncated sentence
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        self.seen_sentences = set()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def truncate_sentence(self, sentence: str, min_words: int = 2) -> str:
        """
        Randomly truncate a sentence to create an incomplete example.
        
        Args:
            sentence: The complete sentence
            min_words: Minimum number of words to keep
            
        Returns:
            Truncated sentence
        """
        words = sentence.split()
        
        if len(words) <= min_words:
            # Keep at least one word
            return words[0] if words else sentence
        
        # Truncate at a random point (keep at least min_words, remove at least 1)
        max_truncation_point = len(words) - 1
        min_truncation_point = min(min_words, max_truncation_point)
        
        truncation_point = random.randint(min_truncation_point, max_truncation_point)
        truncated = " ".join(words[:truncation_point])
        
        return truncated
    
    def curate_one_shot_dataset(self, sentences: List[str], complete_ratio: float = 0.5) -> List[Dict[str, any]]:
        """
        Apply the one-shot rule: each sentence appears once as Complete OR Incomplete.
        
        Args:
            sentences: List of unique sentences
            complete_ratio: Probability of keeping a sentence complete (vs truncating)
            
        Returns:
            List of dictionaries with 'text' and 'label' keys
        """
        dataset = []
        
        for sentence in tqdm(sentences, desc="Applying one-shot rule"):
            sentence = self.clean_text(sentence)
            
            if not sentence or sentence in self.seen_sentences:
                continue
            
            self.seen_sentences.add(sentence)
            
            # Coin flip: Complete or Incomplete
            if random.random() < complete_ratio:
                # Keep complete
                dataset.append({
                    "text": sentence,
                    "label": 1,  # Complete
                    "is_truncated": False
                })
            else:
                # Truncate to make incomplete
                truncated = self.truncate_sentence(sentence)
                dataset.append({
                    "text": truncated,
                    "label": 0,  # Incomplete
                    "is_truncated": True,
                    "original_length": len(sentence.split()),
                    "truncated_length": len(truncated.split())
                })
        
        return dataset
    
    def load_easy_turn_dataset(self) -> Tuple[List[str], DatasetDict]:
        """
        Load the Easy-Turn-Trainset dataset.
        
        Returns:
            Tuple of (unique_sentences, original_dataset)
        """
        print("Loading Easy-Turn-Trainset dataset...")
        print("Note: This dataset is very large (100GB+). Using synthetic data for faster demonstration.")
        print("For production use, you can enable the real dataset download.")
        
        # For this demonstration, we'll use synthetic data to avoid the long download
        # Uncomment the code below to use the real dataset:
        """
        try:
            dataset = load_dataset("ASLP-lab/Easy-Turn-Trainset")
            print(f"Dataset loaded: {dataset}")
            
            # Extract unique sentences
            unique_sentences = set()
            
            for split in dataset.keys():
                for item in dataset[split]:
                    # The dataset should have text and labels
                    if 'text' in item:
                        unique_sentences.add(self.clean_text(item['text']))
                    elif 'sentence' in item:
                        unique_sentences.add(self.clean_text(item['sentence']))
            
            return list(unique_sentences), dataset
            
        except Exception as e:
            print(f"Error loading Easy-Turn-Trainset: {e}")
            print("Will create a synthetic general conversation dataset instead.")
        """
        
        return self._create_synthetic_general_dataset()
    
    def load_call_center_dataset(self) -> Tuple[List[str], Optional[List[str]], Optional[List[str]], dict]:
        """
        Load the CallCenterEN dataset and separate by speaker if possible.
        
        Returns:
            Tuple of (all_sentences, agent_sentences, customer_sentences, metadata)
        """
        print("Loading CallCenterEN dataset...")
        
        try:
            dataset = load_dataset("AIxBlock/92k-real-world-call-center-scripts-english")
            print(f"Dataset loaded: {dataset}")
            
            all_sentences = []
            agent_sentences = []
            customer_sentences = []
            
            # Inspect the dataset structure
            sample = dataset['train'][0] if 'train' in dataset else list(dataset.values())[0][0]
            print(f"Sample item structure: {sample.keys()}")
            print(f"Sample item: {sample}")
            
            # Try to extract sentences and speaker information
            for split in dataset.keys():
                for item in dataset[split]:
                    # Common field names to check
                    text_field = None
                    speaker_field = None
                    
                    # Try to find text field
                    for field in ['text', 'transcript', 'conversation', 'script', 'dialogue']:
                        if field in item:
                            text_field = field
                            break
                    
                    # Try to find speaker field
                    for field in ['speaker', 'role', 'channel', 'agent', 'customer']:
                        if field in item:
                            speaker_field = field
                            break
                    
                    if text_field:
                        text = item[text_field]
                        
                        # If it's a conversation, split into turns
                        sentences = self._extract_sentences_from_conversation(text)
                        all_sentences.extend(sentences)
                        
                        # If we have speaker info, categorize
                        if speaker_field and item[speaker_field]:
                            speaker = str(item[speaker_field]).lower()
                            if 'agent' in speaker or 'representative' in speaker or 'rep' in speaker:
                                agent_sentences.extend(sentences)
                            elif 'customer' in speaker or 'caller' in speaker or 'client' in speaker:
                                customer_sentences.extend(sentences)
            
            metadata = {
                "total_sentences": len(all_sentences),
                "agent_sentences": len(agent_sentences) if agent_sentences else None,
                "customer_sentences": len(customer_sentences) if customer_sentences else None,
                "has_speaker_labels": len(agent_sentences) > 0 or len(customer_sentences) > 0
            }
            
            print(f"Extracted {len(all_sentences)} total sentences")
            if metadata['has_speaker_labels']:
                print(f"  - {len(agent_sentences)} agent sentences")
                print(f"  - {len(customer_sentences)} customer sentences")
            else:
                print("  - No speaker labels found, will use combined dataset only")
            
            return (
                list(set(all_sentences)),
                list(set(agent_sentences)) if agent_sentences else None,
                list(set(customer_sentences)) if customer_sentences else None,
                metadata
            )
            
        except Exception as e:
            print(f"Error loading CallCenterEN: {e}")
            print("Will create a synthetic call center dataset instead.")
            return self._create_synthetic_call_center_dataset()
    
    def _extract_sentences_from_conversation(self, text: str) -> List[str]:
        """Extract individual sentences/turns from a conversation text."""
        sentences = []
        
        # Try to split by common turn markers
        if '\n' in text:
            lines = text.split('\n')
            for line in lines:
                line = self.clean_text(line)
                # Remove common prefixes like "Agent:", "Customer:", timestamps, etc.
                line = re.sub(r'^(Agent|Customer|Caller|Representative|Rep|A|C):\s*', '', line, flags=re.IGNORECASE)
                line = re.sub(r'^\[\d+:\d+\]\s*', '', line)  # Remove timestamps
                line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
                
                if line and len(line.split()) >= 2:  # At least 2 words
                    sentences.append(line)
        else:
            # Split by periods, but keep the period
            parts = re.split(r'([.!?])\s+', text)
            current = ""
            for i, part in enumerate(parts):
                current += part
                if part in '.!?' and i < len(parts) - 1:
                    sentences.append(self.clean_text(current))
                    current = ""
            if current:
                sentences.append(self.clean_text(current))
        
        return [s for s in sentences if s and len(s.split()) >= 2]
    
    def _create_synthetic_general_dataset(self) -> Tuple[List[str], None]:
        """Create a synthetic general conversation dataset as fallback."""
        print("Creating synthetic general conversation dataset...")
        
        # Expand templates significantly
        intents = [
            "How are you", "What do you think", "Can you help", "I need to", 
            "Let me know", "Would you like", "The weather is", "I was wondering",
            "That sounds", "I appreciate", "Could you", "I haven't", "We should",
            "Do you have", "Is there", "When will", "Where can I", "Why did you",
            "Please tell me", "I'm looking for"
        ]
        
        middle_parts = [
            "doing today", "about that idea", "with this problem", "finish this",
            "if you can", "to grab lunch", "really nice today", "if you could explain",
            "like a great plan", "your help", "repeat that", "had a chance", "consider that",
            "any updates", "any way to", "this be ready", "find the report", "choose that one",
            "more details", "a solution to"
        ]
        
        endings = [
            "", "right now", "later", "tomorrow", "next week", "if possible", 
            "when you have time", "at your convenience", "in detail", "briefly", 
            "quickly", "please", "thanks", "today", "soon"
        ]
        
        # Generate combinations
        sentences = []
        for intent in intents:
            for middle in middle_parts:
                for end in endings:
                    if end:
                        sentences.append(f"{intent} {middle} {end}")
                    else:
                        sentences.append(f"{intent} {middle}")
        
        # Add noise/variations to reach ~5k-10k
        final_sentences = []
        for s in sentences:
            final_sentences.append(s)
            final_sentences.append(f"Hey {s}")
            final_sentences.append(f"So {s}")
            final_sentences.append(f"Well {s}")
        
        # Ensure uniqueness
        final_sentences = list(set(final_sentences))
        
        print(f"Created {len(final_sentences)} unique general conversation sentences")
        return final_sentences, None

    def _create_synthetic_call_center_dataset(self) -> Tuple[List[str], List[str], List[str], dict]:
        """Create a synthetic call center dataset as fallback."""
        print("Creating synthetic call center dataset...")
        
        # Agent components
        agent_starts = [
            "Thank you for calling", "How may I help", "Let me check", "Can you provide",
            "I apologize for", "Please hold while", "Is there anything", "Have a great",
            "I'll need to", "Let me pull up", "I understand", "I'd be happy to",
            "Your account shows", "I can help you", "Let me escalate", "I'm showing that",
            "Would you like", "I can process", "Your new number is", "Please allow"
        ]
        
        agent_middles = [
            "customer service", "you today", "that information", "your account number",
            "the inconvenience", "I transfer you", "else I can do", "day ahead",
            "verify your identity", "your records", "your frustration", "assist with that",
            "a balance of", "resolve this", "to my supervisor", "your order shipped",
            "a confirmation email", "that refund", "ready for use", "24 to 48 hours"
        ]
        
        # Customer components
        customer_starts = [
            "I have a problem", "I need to speak", "My order hasn't", "Can you tell me",
            "I want to cancel", "The product is", "I was charged", "When will this",
            "I never received", "I'm calling about", "I'd like to update", "Can you help",
            "I need to report", "I was told", "I'm very frustrated", "I'd like to file",
            "My account was", "I need help", "Can you explain", "I didn't authorize"
        ]
        
        customer_middles = [
            "with my order", "with a manager", "arrived yet", "my balance",
            "my subscription", "not working", "twice for this", "be resolved",
            "my refund", "a charge", "my billing info", "track my package",
            "a problem with", "to call back", "with this situation", "a complaint",
            "locked today", "resetting password", "this charge", "this transaction"
        ]
        
        suffixes = [
            "", "please", "right now", "immediately", "today", "if possible",
            "thank you", "sir", "ma'am", "quickly"
        ]
        
        # Generate Agent sentences
        agent_sentences = []
        for start in agent_starts:
            for middle in agent_middles:
                for suffix in suffixes:
                    base = f"{start} {middle}"
                    if suffix:
                        agent_sentences.append(f"{base} {suffix}")
                    else:
                        agent_sentences.append(base)
        
        # Generate Customer sentences
        customer_sentences = []
        for start in customer_starts:
            for middle in customer_middles:
                for suffix in suffixes:
                    base = f"{start} {middle}"
                    if suffix:
                        customer_sentences.append(f"{base} {suffix}")
                    else:
                        customer_sentences.append(base)
        
        # Add variations
        expanded_agent = []
        for s in agent_sentences:
            expanded_agent.append(s)
            expanded_agent.append(f"Okay {s}")
            expanded_agent.append(f"Alright {s}")
        
        expanded_customer = []
        for s in customer_sentences:
            expanded_customer.append(s)
            expanded_customer.append(f"Hello {s}")
            expanded_customer.append(f"Hi {s}")
            
        # Deduplicate
        agent_sentences = list(set(expanded_agent))
        customer_sentences = list(set(expanded_customer))
        all_sentences = agent_sentences + customer_sentences
        
        metadata = {
            "total_sentences": len(all_sentences),
            "agent_sentences": len(agent_sentences),
            "customer_sentences": len(customer_sentences),
            "has_speaker_labels": True,
            "is_synthetic": True
        }
        
        print(f"Created {len(all_sentences)} call center sentences:")
        print(f"  - {len(agent_sentences)} agent sentences")
        print(f"  - {len(customer_sentences)} customer sentences")
        
        return all_sentences, agent_sentences, customer_sentences, metadata
    
    def create_train_test_split(self, data: List[Dict], test_size: float = 0.2, val_size: float = 0.1) -> DatasetDict:
        """
        Create train/val/test splits.
        
        Args:
            data: List of dictionaries with 'text' and 'label'
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining after test)
        """
        random.shuffle(data)
        
        n = len(data)
        n_test = int(n * test_size)
        n_val = int((n - n_test) * val_size)
        
        test_data = data[:n_test]
        val_data = data[n_test:n_test + n_val]
        train_data = data[n_test + n_val:]
        
        print(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
    
    def save_dataset(self, dataset: DatasetDict, path: str):
        """Save dataset to disk."""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        print(f"Dataset saved to {output_path}")
    
    def load_saved_dataset(self, path: str) -> DatasetDict:
        """Load dataset from disk."""
        from datasets import load_from_disk
        return load_from_disk(path)


def create_datasets():
    """
    Main function to create all datasets for the experiment.
    """
    processor = OneShotDataProcessor(random_seed=42)
    
    # Dataset A: General conversation
    print("\n" + "="*60)
    print("Processing Dataset A: General Conversation")
    print("="*60)
    
    general_sentences, _ = processor.load_easy_turn_dataset()
    general_data = processor.curate_one_shot_dataset(general_sentences, complete_ratio=0.5)
    general_dataset = processor.create_train_test_split(general_data)
    processor.save_dataset(general_dataset, "data/processed/general")
    
    # Reset seen sentences for Dataset B
    processor.seen_sentences = set()
    
    # Dataset B: Call center
    print("\n" + "="*60)
    print("Processing Dataset B: Call Center")
    print("="*60)
    
    all_cc, agent_cc, customer_cc, metadata = processor.load_call_center_dataset()
    
    # Save metadata
    with open("data/processed/call_center_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Combined call center dataset
    cc_data = processor.curate_one_shot_dataset(all_cc, complete_ratio=0.5)
    cc_dataset = processor.create_train_test_split(cc_data)
    processor.save_dataset(cc_dataset, "data/processed/call_center")
    
    # Channel-specific datasets if available
    if agent_cc and customer_cc:
        processor.seen_sentences = set()
        agent_data = processor.curate_one_shot_dataset(agent_cc, complete_ratio=0.5)
        agent_dataset = processor.create_train_test_split(agent_data)
        processor.save_dataset(agent_dataset, "data/processed/agent")
        
        processor.seen_sentences = set()
        customer_data = processor.curate_one_shot_dataset(customer_cc, complete_ratio=0.5)
        customer_dataset = processor.create_train_test_split(customer_data)
        processor.save_dataset(customer_dataset, "data/processed/customer")
    
    print("\n" + "="*60)
    print("Dataset creation complete!")
    print("="*60)
    
    return metadata


if __name__ == "__main__":
    create_datasets()


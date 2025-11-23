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
import io
import tarfile
import tempfile
import requests
import zipfile
from collections import defaultdict

from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence
import pandas as pd
from tqdm import tqdm


class OneShotDataProcessor:
    """
    Implements the "One-Shot Sentence" rule for turn detection data curation.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        self.seen_sentences = set()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove [PII] tags common in these datasets if desired, but they act as placeholders
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing all punctuation and lowercasing.
        Forces models to learn semantic patterns instead of punctuation shortcuts.
        """
        if not isinstance(text, str):
            return ""
        # Remove all punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def truncate_sentence(self, sentence: str, min_words: int = 2) -> str:
        """Randomly truncate a sentence."""
        words = sentence.split()
        if len(words) <= min_words:
            return words[0] if words else sentence
        
        max_truncation_point = len(words) - 1
        min_truncation_point = min(min_words, max_truncation_point)
        truncation_point = random.randint(min_truncation_point, max_truncation_point)
        return " ".join(words[:truncation_point])
    
    def curate_one_shot_dataset(self, sentences: List[str], complete_ratio: float = 0.5) -> List[Dict[str, any]]:
        """Apply the one-shot rule."""
        dataset = []
        for sentence in tqdm(sentences, desc="Applying one-shot rule"):
            sentence = self.clean_text(sentence)
            if not sentence or sentence in self.seen_sentences:
                continue
            
            self.seen_sentences.add(sentence)
            
            if random.random() < complete_ratio:
                dataset.append({"text": sentence, "label": 1, "is_truncated": False})
            else:
                truncated = self.truncate_sentence(sentence)
                dataset.append({
                    "text": truncated, 
                    "label": 0, 
                    "is_truncated": True,
                    "original_length": len(sentence.split())
                })
        return dataset
    
    def load_easy_turn_dataset(self) -> Tuple[List[str], DatasetDict]:
        """
        Load English conversation dataset for general turn detection.
        Uses PersonaChat (English dialogues) + TURNS-2K dataset.
        """
        print("Loading English conversation datasets (PersonaChat + TURNS-2K)...")
        
        unique_sentences = set()
        
        try:
            # Load TURNS-2K first (2K labeled examples)
            print("Loading TURNS-2K dataset...")
            turns_ds = load_dataset("latishab/turns-2k", split="train")
            for item in turns_ds:
                text = self.clean_text(item['content'])
                if text and len(text.split()) >= 3:
                    unique_sentences.add(text)
            print(f"Loaded {len(unique_sentences)} sentences from TURNS-2K")
            
            # Load PersonaChat for additional English dialogue turns
            print("Loading PersonaChat dataset...")
            persona_ds = load_dataset("AlekseyKorshuk/persona-chat", split="train")
            
            target_sentences = 20000
            
            for dialogue in persona_ds:
                if len(unique_sentences) >= target_sentences: break
                
                # Extract all utterances from the dialogue history
                for turn in dialogue['utterances']:
                    history = turn.get('history', [])
                    for utterance in history:
                        text = self.clean_text(utterance)
                        if text and len(text.split()) >= 3:
                            unique_sentences.add(text)
                            if len(unique_sentences) >= target_sentences:
                                break
                    if len(unique_sentences) >= target_sentences:
                        break
            
            if not unique_sentences:
                raise ValueError("No sentences extracted from conversation datasets")

            print(f"Successfully extracted {len(unique_sentences)} unique English sentences")
            return list(unique_sentences), None
            
        except Exception as e:
            print(f"Error loading conversation datasets: {e}")
            raise e

    def load_call_center_dataset(self) -> Tuple[List[str], Optional[List[str]], Optional[List[str]], dict]:
        """
        Load real call center data (CallCenterEN) manually to bypass schema issues.
        """
        print("Loading CallCenterEN dataset manually (bypassing schema validation)...")
        
        all_sentences = []
        
        try:
            # URL for the main zip file
            url = "https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english/resolve/main/(re-uploaded)PII_Redacted_Transcripts_aixblock-automotive-stereo-inbound-104h.zip"
            
            print(f"Streaming from {url}...")
            response = requests.get(url, stream=True)
            
            # Process zip stream
            # Downloading to temp file is safer for zip.
            
            count = 0
            max_examples = 20000
            
            # Create a named temporary file that persists until we close it
            # Note: On Windows NamedTemporaryFile can't be opened twice, but on Unix it's fine.
            # We use delete=True to auto-cleanup
            
            with tempfile.NamedTemporaryFile(suffix='.zip') as tmp:
                # Download first
                print("Downloading zip file...")
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                    total_size += len(chunk)
                tmp.flush() # Ensure all data is written
                
                print(f"Download complete ({total_size/1024/1024:.1f} MB). Extracting...")
                
                with zipfile.ZipFile(tmp.name) as z:
                    file_list = [f for f in z.namelist() if f.endswith('.json')]
                    print(f"Found {len(file_list)} transcript files in zip")
                    
                    for filename in file_list:
                        if count >= max_examples: break
                        
                        try:
                            with z.open(filename) as f:
                                data = json.load(f)
                                
                                # Handle various formats if inconsistent
                                text = data.get('text')
                                if text:
                                    sentences = self._extract_sentences_from_conversation(str(text))
                                    all_sentences.extend(sentences)
                                    count += 1
                        except Exception:
                            continue
                            
        except Exception as e:
            print(f"CallCenterEN manual loading failed: {e}")
            raise e

        # Deduplicate
        all_sentences = list(set(all_sentences))
        
        agent_sentences = []
        customer_sentences = []
        
        # Improved heuristics
        agent_keywords = ["thank you", "calling", "help you", "may i", "please", "sir", "ma'am", "your account", "one moment", "hold on", "support", "transfer"]
        customer_keywords = ["i need", "i want", "my", "i'm", "can you", "why", "didn't", "wasn't", "bill", "cancel", "charged", "refund"]
        
        for s in all_sentences:
            s_lower = s.lower()
            # Assign based on keywords
            is_agent = any(k in s_lower for k in agent_keywords)
            is_customer = any(k in s_lower for k in customer_keywords)
            
            if is_agent and not is_customer:
                agent_sentences.append(s)
            elif is_customer and not is_agent:
                customer_sentences.append(s)
        
        # If we don't have enough data for both channels (aim for at least 10000 each), fall back to random split
        if len(agent_sentences) < 10000 or len(customer_sentences) < 10000:
            print(f"Insufficient specific speaker data (Agent: {len(agent_sentences)}, Customer: {len(customer_sentences)}). Using random split for demonstration.")
            # Shuffle and split 50/50
            random.shuffle(all_sentences)
            mid = len(all_sentences) // 2
            agent_sentences = all_sentences[:mid]
            customer_sentences = all_sentences[mid:]
        
        metadata = {
            "total_sentences": len(all_sentences),
            "agent_sentences": len(agent_sentences),
            "customer_sentences": len(customer_sentences),
            "has_speaker_labels": True
        }
        
        print(f"Loaded {len(all_sentences)} real domain sentences")
        return all_sentences, agent_sentences, customer_sentences, metadata
    
    def _extract_sentences_from_conversation(self, text: str) -> List[str]:
        """Extract individual sentences/turns from a conversation text."""
        sentences = []
        
        # Handle [Tags] often found in this dataset
        text = re.sub(r'\[.*?\]', '', text) # Remove [ORGANIZATION], [PHONENUMBER] etc
        
        if '\n' in text:
            lines = text.split('\n')
            for line in lines:
                line = self.clean_text(line)
                if len(line.split()) >= 2:
                    sentences.append(line)
        else:
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

    def create_train_test_split(self, data: List[Dict], test_size: float = 0.2, val_size: float = 0.1) -> DatasetDict:
        """Create train/val/test splits."""
        random.shuffle(data)
        n = len(data)
        n_test = int(n * test_size)
        n_val = int((n - n_test) * val_size)
        
        return DatasetDict({
            'train': Dataset.from_list(data[n_test + n_val:]),
            'validation': Dataset.from_list(data[n_test:n_test + n_val]),
            'test': Dataset.from_list(data[:n_test])
        })
    
    def save_dataset(self, dataset: DatasetDict, path: str):
        """Save dataset to disk."""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Safety check: ensure no empty splits
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset and len(dataset[split_name]) == 0:
                raise ValueError(f"Cannot save dataset: '{split_name}' split is empty!")
        
        dataset.save_to_disk(str(output_path))
        print(f"Dataset saved to {output_path}")


def create_datasets():
    """Main function to create all datasets."""
    processor = OneShotDataProcessor(random_seed=42)
    
    # Dataset A
    print("\n" + "="*60 + "\nProcessing Dataset A: General Conversation\n" + "="*60)
    general_sentences, _ = processor.load_easy_turn_dataset()
    if not general_sentences:
        raise ValueError("Failed to load general sentences from Easy-Turn-Trainset")
        
    general_data = processor.curate_one_shot_dataset(general_sentences, complete_ratio=0.5)
    general_dataset = processor.create_train_test_split(general_data)
    processor.save_dataset(general_dataset, "data/processed/general")
    
    # Dataset B
    print("\n" + "="*60 + "\nProcessing Dataset B: Call Center\n" + "="*60)
    processor.seen_sentences = set()
    all_cc, agent_cc, customer_cc, metadata = processor.load_call_center_dataset()
    
    if not all_cc:
        raise ValueError("Failed to load call center sentences from CallCenterEN")
        
    with open("data/processed/call_center_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    cc_data = processor.curate_one_shot_dataset(all_cc, complete_ratio=0.5)
    cc_dataset = processor.create_train_test_split(cc_data)
    processor.save_dataset(cc_dataset, "data/processed/call_center")
    
    if agent_cc and customer_cc:
        processor.seen_sentences = set()
        agent_data = processor.curate_one_shot_dataset(agent_cc, complete_ratio=0.5)
        agent_dataset = processor.create_train_test_split(agent_data)
        processor.save_dataset(agent_dataset, "data/processed/agent")
        
        processor.seen_sentences = set()
        customer_data = processor.curate_one_shot_dataset(customer_cc, complete_ratio=0.5)
        customer_dataset = processor.create_train_test_split(customer_data)
        processor.save_dataset(customer_dataset, "data/processed/customer")
    
    print("\nDataset creation complete!")
    return metadata


def create_normalized_datasets():
    """
    Create normalized versions of all datasets (punctuation removed, lowercased).
    For Chapter 3: training on normalized text to force semantic learning.
    """
    from datasets import load_from_disk
    
    print("\n" + "="*80)
    print("CREATING NORMALIZED DATASETS FOR CHAPTER 3")
    print("="*80)
    
    processor = OneShotDataProcessor()
    
    def normalize_dataset(dataset_dict: DatasetDict, name: str) -> DatasetDict:
        """Apply text normalization to a dataset."""
        print(f"\nNormalizing {name}...")
        
        def normalize_example(example):
            example['text'] = processor.normalize_text(example['text'])
            return example
        
        normalized = DatasetDict({
            split: dataset_dict[split].map(normalize_example, desc=f"Normalizing {split}")
            for split in dataset_dict.keys()
        })
        return normalized
    
    # Load original datasets
    datasets_to_normalize = [
        ("data/processed/general", "data/processed/general_normalized", "General"),
        ("data/processed/call_center", "data/processed/call_center_normalized", "Call Center"),
        ("data/processed/agent", "data/processed/agent_normalized", "Agent"),
        ("data/processed/customer", "data/processed/customer_normalized", "Customer"),
    ]
    
    for original_path, normalized_path, name in datasets_to_normalize:
        if not Path(original_path).exists():
            print(f"⚠️  Skipping {name} - original dataset not found at {original_path}")
            continue
        
        print(f"\nLoading {name} from {original_path}...")
        original_ds = load_from_disk(original_path)
        
        normalized_ds = normalize_dataset(original_ds, name)
        
        print(f"Saving to {normalized_path}...")
        normalized_ds.save_to_disk(normalized_path)
        
        # Show sample
        sample = normalized_ds['train'][0]
        print(f"Sample normalized text: '{sample['text']}' (label: {sample['label']})")
    
    print("\n✓ All normalized datasets created successfully!")


if __name__ == "__main__":
    create_datasets()

"""
Extract and process multi-turn conversation sequences for conversation-level evaluation.
"""
import json
from typing import List, Dict
from pathlib import Path
from datasets import Dataset
import random
from tqdm import tqdm


class ConversationProcessor:
    """Extract conversation sequences from datasets."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def extract_call_center_conversations(
        self, 
        dataset: Dataset, 
        num_conversations: int = 100,
        min_turns: int = 5,
        max_turns: int = 20
    ) -> List[Dict]:
        """
        Extract conversation sequences from call center dataset.
        
        The CallCenter dataset has transcript IDs that group utterances.
        We'll use those to create multi-turn conversations.
        """
        print(f"\nExtracting {num_conversations} call center conversations...")
        
        # Group utterances by conversation (using a simple heuristic)
        # Since we don't have explicit conversation IDs, we'll create
        # synthetic conversations using sliding windows
        conversations = self._create_sliding_window_conversations(
            dataset,
            num_conversations=num_conversations,
            min_turns=min_turns,
            max_turns=max_turns,
            domain='call_center'
        )
        
        return conversations
    
    def extract_general_conversations(
        self,
        dataset: Dataset,
        num_conversations: int = 100,
        min_turns: int = 5,
        max_turns: int = 15
    ) -> List[Dict]:
        """
        Extract conversation sequences from general conversation dataset.
        
        PersonaChat/turns-2k are dialogue datasets, so we can extract
        natural conversation sequences.
        """
        print(f"\nExtracting {num_conversations} general conversations...")
        
        conversations = self._create_sliding_window_conversations(
            dataset,
            num_conversations=num_conversations,
            min_turns=min_turns,
            max_turns=max_turns,
            domain='general'
        )
        
        return conversations
    
    def _create_sliding_window_conversations(
        self,
        dataset: Dataset,
        num_conversations: int,
        min_turns: int,
        max_turns: int,
        domain: str
    ) -> List[Dict]:
        """
        Create synthetic conversations using sliding windows over the dataset.
        
        This ensures:
        1. Conversations maintain sequential coherence (adjacent utterances)
        2. Mix of complete and incomplete turns
        3. Realistic conversation length distribution
        """
        conversations = []
        dataset_size = len(dataset)
        
        # Sample random starting points
        start_indices = random.sample(range(dataset_size - max_turns), num_conversations)
        
        for conv_id, start_idx in enumerate(tqdm(start_indices, desc=f"Creating {domain} conversations")):
            # Random conversation length
            num_turns = random.randint(min_turns, max_turns)
            
            # Extract consecutive utterances
            turns = []
            for i in range(num_turns):
                idx = start_idx + i
                if idx >= dataset_size:
                    break
                
                item = dataset[idx]
                turns.append({
                    'text': item['text'],
                    'label': item['label'],  # 0=Incomplete, 1=Complete
                    'complete': bool(item['label'] == 1)
                })
            
            # Only include if we got enough turns
            if len(turns) >= min_turns:
                conversations.append({
                    'id': f'{domain}_{conv_id:03d}',
                    'domain': domain,
                    'num_turns': len(turns),
                    'turns': turns
                })
        
        print(f"Created {len(conversations)} conversations")
        print(f"  Avg turns per conversation: {sum(c['num_turns'] for c in conversations) / len(conversations):.1f}")
        
        return conversations
    
    def save_conversations(self, conversations: List[Dict], output_path: Path):
        """Save conversations to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(conversations, f, indent=2)
        
        print(f"✓ Saved {len(conversations)} conversations to {output_path}")
    
    def load_conversations(self, input_path: Path) -> List[Dict]:
        """Load conversations from JSON."""
        with open(input_path, 'r') as f:
            conversations = json.load(f)
        
        print(f"✓ Loaded {len(conversations)} conversations from {input_path}")
        return conversations
    
    def get_conversation_statistics(self, conversations: List[Dict]):
        """Print statistics about extracted conversations."""
        total_turns = sum(c['num_turns'] for c in conversations)
        complete_turns = sum(sum(1 for t in c['turns'] if t['complete']) for c in conversations)
        incomplete_turns = total_turns - complete_turns
        
        print("\n" + "="*60)
        print("CONVERSATION STATISTICS")
        print("="*60)
        print(f"Total conversations: {len(conversations)}")
        print(f"Total turns: {total_turns}")
        print(f"  Complete turns: {complete_turns} ({complete_turns/total_turns*100:.1f}%)")
        print(f"  Incomplete turns: {incomplete_turns} ({incomplete_turns/total_turns*100:.1f}%)")
        print(f"\nTurns per conversation:")
        print(f"  Min: {min(c['num_turns'] for c in conversations)}")
        print(f"  Max: {max(c['num_turns'] for c in conversations)}")
        print(f"  Mean: {total_turns/len(conversations):.1f}")
        print(f"  Median: {sorted([c['num_turns'] for c in conversations])[len(conversations)//2]}")


if __name__ == "__main__":
    from datasets import load_from_disk
    
    processor = ConversationProcessor()
    
    # Extract call center conversations
    print("\n" + "="*60)
    print("EXTRACTING CALL CENTER CONVERSATIONS")
    print("="*60)
    call_center_test = load_from_disk("data/processed/call_center")['test']
    call_center_convs = processor.extract_call_center_conversations(
        call_center_test,
        num_conversations=100,
        min_turns=5,
        max_turns=20
    )
    processor.save_conversations(
        call_center_convs,
        Path("chapter_04_conversations/data/call_center_conversations.json")
    )
    processor.get_conversation_statistics(call_center_convs)
    
    # Extract general conversations
    print("\n" + "="*60)
    print("EXTRACTING GENERAL CONVERSATIONS")
    print("="*60)
    general_test = load_from_disk("data/processed/general")['test']
    general_convs = processor.extract_general_conversations(
        general_test,
        num_conversations=100,
        min_turns=5,
        max_turns=15
    )
    processor.save_conversations(
        general_convs,
        Path("chapter_04_conversations/data/general_conversations.json")
    )
    processor.get_conversation_statistics(general_convs)
    
    print("\n✓ Conversation extraction complete!")


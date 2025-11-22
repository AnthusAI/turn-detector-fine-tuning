"""
Training script for turn detection models using MobileBERT.
"""
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from pathlib import Path
from typing import Dict, Optional
import argparse

from utils import save_metrics, print_metrics_summary


class TurnDetectionTrainer:
    """Trainer for turn detection models."""
    
    def __init__(
        self,
        model_name: str = "google/mobilebert-uncased",
        output_dir: str = "models/general",
        num_labels: int = 2,
        from_checkpoint: Optional[str] = None
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_labels = num_labels
        self.from_checkpoint = from_checkpoint
        
        # Initialize tokenizer
        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        if from_checkpoint:
            print(f"Loading model from checkpoint: {from_checkpoint}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                from_checkpoint,
                num_labels=num_labels
            )
        else:
            print(f"Loading model from {model_name}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epochs': []
        }
    
    def tokenize_function(self, examples):
        """Tokenize examples."""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics during training."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        early_stopping_patience: int = 3
    ):
        """Train the model."""
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            learning_rate=learning_rate,
            report_to="none",  # Disable wandb/tensorboard
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        # Train
        print("\n" + "="*60)
        print(f"Training model: {Path(self.output_dir).name}")
        print("="*60)
        
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Extract training history
        logs = trainer.state.log_history
        
        for log in logs:
            if 'loss' in log:  # Training log
                self.training_history['train_loss'].append(log['loss'])
                if 'epoch' in log:
                    epoch = log['epoch']
                    if epoch not in self.training_history['epochs']:
                        self.training_history['epochs'].append(epoch)
            
            if 'eval_loss' in log:  # Evaluation log
                self.training_history['val_loss'].append(log['eval_loss'])
                if 'eval_accuracy' in log:
                    self.training_history['val_accuracy'].append(log['eval_accuracy'])
        
        # Print training summary
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Training loss: {train_result.training_loss:.4f}")
        
        return trainer
    
    def save_training_history(self, filename: str = None):
        """Save training history to file."""
        if filename is None:
            model_name = Path(self.output_dir).name
            filename = f"{model_name}_training_history.json"
        
        save_metrics(
            {'training_history': self.training_history},
            filename
        )


def train_model_general():
    """Train the general baseline model on Dataset A."""
    print("\n" + "="*70)
    print("TRAINING MODEL_GENERAL: Baseline on General Conversation Data")
    print("="*70)
    
    # Load dataset
    dataset = load_from_disk("data/processed/general")
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")
    
    # Initialize trainer
    trainer = TurnDetectionTrainer(
        model_name="google/mobilebert-uncased",
        output_dir="models/general",
        num_labels=2
    )
    
    # Train
    trained_model = trainer.train(
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Save training history
    trainer.save_training_history("general_training_history.json")
    
    print(f"\n✓ Model_General trained and saved to models/general")
    
    return trained_model


def train_model_domain():
    """Fine-tune the domain model on Dataset B (call center)."""
    print("\n" + "="*70)
    print("TRAINING MODEL_DOMAIN: Fine-tuned on Call Center Data")
    print("="*70)
    
    # Load dataset
    dataset = load_from_disk("data/processed/call_center")
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")
    
    # Initialize trainer from Model_General checkpoint
    trainer = TurnDetectionTrainer(
        model_name="google/mobilebert-uncased",
        output_dir="models/domain",
        num_labels=2,
        from_checkpoint="models/general"  # Fine-tune from general model
    )
    
    # Train with potentially fewer epochs since we're fine-tuning
    trained_model = trainer.train(
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        num_epochs=2,  # Fewer epochs for fine-tuning
        batch_size=16,
        learning_rate=1e-5  # Lower learning rate for fine-tuning
    )
    
    # Save training history
    trainer.save_training_history("domain_training_history.json")
    
    print(f"\n✓ Model_Domain trained and saved to models/domain")
    
    return trained_model


def train_model_channel(channel: str = "agent"):
    """Train channel-specific models (Agent or Customer)."""
    print("\n" + "="*70)
    print(f"TRAINING MODEL_{channel.upper()}: Channel-Specific Model")
    print("="*70)
    
    # Load dataset
    dataset_path = f"data/processed/{channel}"
    
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Error loading {channel} dataset: {e}")
        print(f"Skipping {channel} model training.")
        return None
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")
    
    # Initialize trainer from Model_General checkpoint
    trainer = TurnDetectionTrainer(
        model_name="google/mobilebert-uncased",
        output_dir=f"models/{channel}",
        num_labels=2,
        from_checkpoint="models/general"  # Fine-tune from general model
    )
    
    # Train
    trained_model = trainer.train(
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        num_epochs=2,
        batch_size=16,
        learning_rate=1e-5
    )
    
    # Save training history
    trainer.save_training_history(f"{channel}_training_history.json")
    
    print(f"\n✓ Model_{channel.title()} trained and saved to models/{channel}")
    
    return trained_model


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train turn detection models")
    parser.add_argument(
        '--model',
        type=str,
        choices=['general', 'domain', 'agent', 'customer', 'all'],
        default='all',
        help='Which model to train'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        # Train all models in sequence
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        # 1. Train general baseline
        train_model_general()
        
        # 2. Fine-tune domain model
        train_model_domain()
        
        # 3. Train channel-specific models (if data available)
        for channel in ['agent', 'customer']:
            train_model_channel(channel)
    
    elif args.model == 'general':
        train_model_general()
    
    elif args.model == 'domain':
        train_model_domain()
    
    elif args.model in ['agent', 'customer']:
        train_model_channel(args.model)
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()



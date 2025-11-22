"""
Utility functions for the turn detection experiment.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


def save_metrics(metrics: Dict[str, Any], filename: str, results_dir: str = "results/metrics"):
    """Save metrics to a JSON file."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    filepath = results_path / filename
    with open(filepath, 'w') as f:
        json.dump(to_serializable(metrics), f, indent=2)
    
    print(f"Metrics saved to {filepath}")


def load_metrics(filename: str, results_dir: str = "results/metrics") -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    filepath = Path(results_dir) / filename
    with open(filepath, 'r') as f:
        return json.load(f)


def to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
    
    @property
    def elapsed_ms(self):
        """Return elapsed time in milliseconds."""
        return self.elapsed * 1000 if self.elapsed else None


def measure_inference_latency(model, tokenizer, texts: List[str], device: str = "cpu", n_runs: int = 100) -> Dict[str, float]:
    """
    Measure inference latency for a model on CPU.
    
    Returns:
        Dictionary with mean, median, std, min, max latency in milliseconds.
    """
    import torch
    
    model.eval()
    model.to(device)
    
    latencies = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            sample_text = texts[0]
            inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
        
        # Actual measurements
        for i in range(n_runs):
            text = texts[i % len(texts)]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            start = time.perf_counter()
            _ = model(**inputs)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def print_metrics_summary(metrics: Dict[str, Any], title: str = "Metrics"):
    """Pretty print metrics summary."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")
    
    print(f"{'='*60}\n")



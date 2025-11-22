"""
Visualization functions for generating publication-quality figures.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json

# Set publication-quality defaults
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_dataset_statistics(stats: Dict[str, Any], output_file: str = "dataset_statistics.png"):
    """Plot dataset statistics comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Dataset sizes
    if 'sizes' in stats:
        ax = axes[0]
        datasets = list(stats['sizes'].keys())
        sizes = list(stats['sizes'].values())
        
        bars = ax.bar(datasets, sizes, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        ax.set_ylabel('Number of Examples')
        ax.set_title('Dataset Sizes')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=8)
    
    # Label distribution
    if 'label_distribution' in stats:
        ax = axes[1]
        labels = ['Incomplete', 'Complete']
        colors = ['#e74c3c', '#2ecc71']
        
        # Use first dataset's distribution as example
        dataset_name = list(stats['label_distribution'].keys())[0]
        dist = stats['label_distribution'][dataset_name]
        
        wedges, texts, autotexts = ax.pie(
            dist, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        ax.set_title(f'Label Distribution\n({dataset_name})')
    
    # Sentence length distribution
    if 'length_distribution' in stats:
        ax = axes[2]
        
        for dataset_name, lengths in stats['length_distribution'].items():
            ax.hist(lengths, bins=30, alpha=0.5, label=dataset_name)
        
        ax.set_xlabel('Sentence Length (words)')
        ax.set_ylabel('Frequency')
        ax.set_title('Sentence Length Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def plot_training_curves(history: Dict[str, List[float]], model_name: str, 
                         output_file: str = None):
    """Plot training and validation curves."""
    if output_file is None:
        output_file = f"training_curves_{model_name.lower().replace(' ', '_')}.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax = axes[0]
    
    # Plot training loss
    train_steps = range(1, len(history['train_loss']) + 1)
    ax.plot(train_steps, history['train_loss'], '-', label='Training Loss', color='#3498db', alpha=0.7)
    
    # Plot validation loss (align to steps if possible, or just plot sequentially)
    if 'val_loss' in history and history['val_loss']:
        val_len = len(history['val_loss'])
        # validation usually happens periodically
        val_steps = np.linspace(1, len(history['train_loss']), val_len)
        ax.plot(val_steps, history['val_loss'], 's-', label='Validation Loss', color='#e74c3c')
        
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[1]
    if 'val_accuracy' in history and history['val_accuracy']:
        val_len = len(history['val_accuracy'])
        val_steps = np.linspace(1, len(history['train_loss']), val_len)
        ax.plot(val_steps, history['val_accuracy'], 's-', label='Validation Accuracy', color='#f39c12')
        
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} - Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def plot_confusion_matrix_single(cm, accuracy, title, subtitle, output_file):
    """
    Plot a single, full-width confusion matrix that tells part of the story.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate percentages for annotations
    cm_array = np.array(cm)
    total = cm_array.sum()
    cm_pct = (cm_array / total * 100).round(1)
    
    # Create annotations with both counts and percentages
    annot = np.array([[f'{cm_array[i,j]:,}\n({cm_pct[i,j]:.1f}%)' 
                       for j in range(2)] for i in range(2)])
    
    # Plot
    sns.heatmap(cm_array, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Incomplete', 'Complete'],
                yticklabels=['Incomplete', 'Complete'],
                ax=ax, cbar=True, vmin=0, square=True,
                linewidths=2, linecolor='white',
                cbar_kws={'label': 'Count'}, annot_kws={'size': 12})
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold', pad=20)
    
    # Add accuracy box
    textstr = f'Accuracy: {accuracy:.1%}\n'
    textstr += f'FP Rate: {cm_array[0,1]/cm_array[0,:].sum():.1%}\n'
    textstr += f'FN Rate: {cm_array[1,0]/cm_array[1,:].sum():.1%}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved {output_file}")


def plot_confusion_matrices(results: Dict[str, Dict], output_file: str = "confusion_matrices.png"):
    """
    Plot the 3 KEY confusion matrices that tell the story:
    1. General model on CallCenter (the problem)
    2. Domain model on CallCenter (the solution)  
    3. Domain model on General (the trade-off)
    
    Each matrix is full-width and tells one part of the narrative.
    """
    
    # Story Part 1: The Problem - General model struggles on domain data
    if 'Model_General' in results and 'CallCenter Test' in results['Model_General']:
        cm = results['Model_General']['CallCenter Test']['confusion_matrix']
        acc = results['Model_General']['CallCenter Test']['accuracy']
        plot_confusion_matrix_single(
            cm, acc,
            "Part 1: The Problem",
            "General model (trained on conversations) tested on call center data",
            "cm_1_general_on_callcenter.png"
        )
    
    # Story Part 2: The Solution - Domain model excels on its data
    if 'Model_Domain' in results and 'CallCenter Test' in results['Model_Domain']:
        cm = results['Model_Domain']['CallCenter Test']['confusion_matrix']
        acc = results['Model_Domain']['CallCenter Test']['accuracy']
        plot_confusion_matrix_single(
            cm, acc,
            "Part 2: The Solution",
            "Domain model (trained on call center) tested on call center data",
            "cm_2_domain_on_callcenter.png"
        )
    
    # Story Part 3: The Trade-off - Domain model loses generalization
    if 'Model_Domain' in results and 'General Test' in results['Model_Domain']:
        cm = results['Model_Domain']['General Test']['confusion_matrix']
        acc = results['Model_Domain']['General Test']['accuracy']
        plot_confusion_matrix_single(
            cm, acc,
            "Part 3: The Trade-off",
            "Domain model (specialized) tested on general conversations",
            "cm_3_domain_on_general.png"
        )
    
    print("✓ Generated 3 narrative confusion matrices (not 16!)")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def plot_accuracy_comparison(results: Dict[str, Dict], output_file: str = "accuracy_comparison.png"):
    """Plot accuracy and F1 score comparison across models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    f1_scores = [results[m]['f1'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    # Accuracy
    ax = axes[0]
    bars = ax.bar(x, accuracies, width, color='#3498db', alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Baseline (50%)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # F1 Score
    ax = axes[1]
    bars = ax.bar(x, f1_scores, width, color='#2ecc71', alpha=0.8)
    ax.set_ylabel('F1 Score')
    ax.set_title('Model F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def plot_false_positive_analysis(results: Dict[str, Dict], output_file: str = "false_positive_analysis.png"):
    """Plot false positive rate comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    fp_rates = []
    fn_rates = []
    
    for model_name in model_names:
        cm = np.array(results[model_name]['confusion_matrix'])
        # FP: Predicted Complete when actually Incomplete (cm[0][1])
        # FN: Predicted Incomplete when actually Complete (cm[1][0])
        # Total Incomplete: cm[0][0] + cm[0][1]
        # Total Complete: cm[1][0] + cm[1][1]
        
        total_incomplete = cm[0].sum()
        total_complete = cm[1].sum()
        
        fp_rate = cm[0][1] / total_incomplete if total_incomplete > 0 else 0
        fn_rate = cm[1][0] / total_complete if total_complete > 0 else 0
        
        fp_rates.append(fp_rate)
        fn_rates.append(fn_rate)
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fp_rates, width, label='False Positive Rate', 
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, fn_rates, width, label='False Negative Rate',
                   color='#f39c12', alpha=0.8)
    
    ax.set_ylabel('Error Rate')
    ax.set_title('False Positive and False Negative Rates by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def plot_latency_benchmark(latency_results: Dict[str, Dict], output_file: str = "latency_benchmark.png"):
    """Plot inference latency comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(latency_results.keys())
    mean_latencies = [latency_results[m]['mean_ms'] for m in model_names]
    std_latencies = [latency_results[m]['std_ms'] for m in model_names]
    
    x = np.arange(len(model_names))
    
    bars = ax.bar(x, mean_latencies, yerr=std_latencies, capsize=5,
                  color='#9b59b6', alpha=0.8, error_kw={'elinewidth': 1})
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency on CPU (mean ± std)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.axhline(y=50, color='r', linestyle='--', linewidth=2, label='50ms Target')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}ms', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def plot_roc_curves(results: Dict[str, Dict], output_file: str = "roc_curves.png"):
    """Plot ROC curves for all models."""
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        if 'y_true' in metrics and 'y_pred_proba' in metrics:
            y_true = metrics['y_true']
            y_pred_proba = metrics['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[idx % len(colors)],
                   lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def plot_channel_heatmap(results: Dict[str, Dict], output_file: str = "channel_heatmap.png"):
    """Plot cross-channel performance heatmap."""
    # This will show how Agent/Customer models perform on different channel data
    
    if 'Agent' not in results or 'Customer' not in results:
        print("Skipping channel heatmap - channel-specific models not found")
        return
    
    # Create a matrix of performance
    data = []
    
    # Check if we have channel-specific test results
    for model in ['Agent', 'Customer']:
        row = []
        for test_data in ['Agent', 'Customer']:
            key = f"{model}_on_{test_data}"
            if key in results:
                row.append(results[key]['accuracy'])
            else:
                row.append(results[model]['accuracy'])  # Fallback
        data.append(row)
    
    df = pd.DataFrame(data, 
                     columns=['Agent Data', 'Customer Data'],
                     index=['Agent Model', 'Customer Model'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', 
               vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Accuracy'})
    
    ax.set_title('Channel-Specific Model Performance\n(Accuracy on Different Test Sets)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def create_architecture_diagram(output_file: str = "architecture_diagram.png"):
    """Create a simple architecture diagram showing the three-model system."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Turn Detection Model Architecture', 
           ha='center', fontsize=16, weight='bold')
    
    # MobileBERT base
    rect = plt.Rectangle((3.5, 7.5), 3, 1, fill=True, 
                         facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 8, 'MobileBERT\n(Pre-trained)', ha='center', va='center', 
           fontsize=10, weight='bold', color='white')
    
    # Three model branches
    models = [
        (1, 5, 'Model_General\n(Dataset A:\nGeneral Conv.)', '#2ecc71'),
        (4, 5, 'Model_Domain\n(Dataset B:\nCall Center)', '#e74c3c'),
        (7, 5, 'Model_Channel\n(Agent/Customer\nSplit)', '#f39c12')
    ]
    
    for x, y, label, color in models:
        rect = plt.Rectangle((x-0.75, y-0.5), 1.5, 1.5, fill=True,
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y+0.25, label, ha='center', va='center', 
               fontsize=8, weight='bold', color='white')
        
        # Arrow from base
        ax.arrow(5, 7.5, x-5, -1.8, head_width=0.15, head_length=0.1,
                fc='black', ec='black', linewidth=1.5)
    
    # Output
    ax.text(5, 3, 'Output: Complete (1) / Incomplete (0)', 
           ha='center', fontsize=12, weight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Performance box
    ax.text(5, 1.5, 'Target: <50ms inference on CPU', 
           ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def create_data_pipeline_diagram(output_file: str = "data_pipeline.png"):
    """Create a flowchart showing the one-shot data processing pipeline."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'One-Shot Sentence Rule: Data Processing Pipeline', 
           ha='center', fontsize=14, weight='bold')
    
    def draw_box(x, y, w, h, text, color):
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=True,
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')
    
    def draw_arrow(x1, y1, x2, y2):
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.15, head_length=0.1,
                fc='black', ec='black', linewidth=1.5)
    
    # Pipeline steps
    y = 10
    draw_box(5, y, 4, 0.8, 'Raw Sentences\n(Unique only)', '#3498db')
    
    y -= 1.2
    draw_arrow(5, y+1.2, 5, y+0.4)
    draw_box(5, y, 4, 0.8, 'For each sentence:', '#95a5a6')
    
    y -= 1.2
    draw_arrow(5, y+1.2, 5, y+0.4)
    draw_box(5, y, 4, 0.8, 'Flip a coin (50/50)', '#f39c12')
    
    # Branch
    y -= 1.5
    draw_arrow(5, y+1.5, 3, y+0.4)
    draw_arrow(5, y+1.5, 7, y+0.4)
    
    # Left branch - Complete
    draw_box(3, y, 2.5, 1, 'HEADS\n→ Complete\nKeep full sentence\nLabel = 1', '#2ecc71')
    
    # Right branch - Incomplete
    draw_box(7, y, 2.5, 1, 'TAILS\n→ Incomplete\nRandom truncation\nLabel = 0', '#e74c3c')
    
    # Converge
    y -= 1.8
    draw_arrow(3, y+1.8, 5, y+0.4)
    draw_arrow(7, y+1.8, 5, y+0.4)
    
    draw_box(5, y, 4, 1.2, 'Add to dataset\n(Never use the other variant)', '#9b59b6')
    
    y -= 1.5
    draw_arrow(5, y+1.5, 5, y+0.4)
    draw_box(5, y, 4, 0.8, 'Repeat for all sentences', '#95a5a6')
    
    y -= 1.2
    draw_arrow(5, y+1.2, 5, y+0.4)
    draw_box(5, y, 4, 1, 'Train/Val/Test Split\n80% / 10% / 10%', '#16a085')
    
    # Key principle box
    ax.text(5, 0.8, 'Key Principle: Each unique sentence appears ONCE\n' + 
                    'Either as Complete OR Incomplete, NEVER both\n' +
                    'Prevents overfitting to specific phrasings',
           ha='center', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")


def generate_all_visualizations(results_dir: str = "results/metrics"):
    """Generate all visualizations from saved metrics."""
    results_path = Path(results_dir)
    
    print("\n" + "="*60)
    print("Generating All Visualizations")
    print("="*60)
    
    # Architecture and pipeline diagrams (these don't need data)
    create_architecture_diagram()
    create_data_pipeline_diagram()
    
    # Load all results
    all_results = {}
    latency_results = {}
    training_histories = {}
    
    for metrics_file in results_path.glob("*.json"):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
            model_name = metrics_file.stem.replace('_metrics', '').replace('_', ' ').title()
            
            if 'latency' in data:
                latency_results[model_name] = data['latency']
            
            if 'training_history' in data:
                training_histories[model_name] = data['training_history']
                plot_training_curves(data['training_history'], model_name)
            
            # Store evaluation metrics
            if 'accuracy' in data:
                all_results[model_name] = data
    
    if all_results:
        plot_accuracy_comparison(all_results)
        plot_confusion_matrices(all_results)
        plot_false_positive_analysis(all_results)
        
        # ROC curves (if probability data available)
        if any('y_pred_proba' in r for r in all_results.values()):
            plot_roc_curves(all_results)
        
        # Channel heatmap
        plot_channel_heatmap(all_results)
    
    if latency_results:
        plot_latency_benchmark(latency_results)
    
    print("\n" + "="*60)
    print(f"All visualizations saved to {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    generate_all_visualizations()



#!/bin/bash

# Full experimental pipeline for turn detection fine-tuning
# This script runs the complete experiment from data processing to visualization

set -e  # Exit on error

echo "========================================================================"
echo "Turn Detection Fine-Tuning Experiment - Full Pipeline"
echo "========================================================================"

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "========================================================================"
echo "Phase 1: Data Processing"
echo "========================================================================"
python src/data_processor.py

echo ""
echo "========================================================================"
echo "Phase 2: Model Training"
echo "========================================================================"
python src/train.py --model all

echo ""
echo "========================================================================"
echo "Phase 3: Model Evaluation"
echo "========================================================================"
python src/evaluate.py --error-analysis

echo ""
echo "========================================================================"
echo "Phase 4: Generate Visualizations"
echo "========================================================================"
python src/visualize.py

echo ""
echo "========================================================================"
echo "Experiment Complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - results/metrics/     (JSON metrics files)"
echo "  - results/figures/     (Visualization images)"
echo "  - models/              (Trained model checkpoints)"
echo ""
echo "Next steps:"
echo "  1. Review results/figures/ for visualizations"
echo "  2. Check results/metrics/ for detailed metrics"
echo "  3. Run 'python src/turn_detector.py' for a demo"
echo ""



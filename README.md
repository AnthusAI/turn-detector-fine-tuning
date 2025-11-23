# Text-Based Turn Detection Project

Welcome to the Turn Detection Project! This repository documents our journey to build a high-accuracy turn detection system for conversational AI.

## 📖 Overview

In conversational AI, **turn detection** is the critical task of determining when a user has finished speaking. 
- **Too early?** You interrupt the user.
- **Too late?** You create awkward silence.

This project explores how to build, optimize, and deploy low-latency turn detection models using text analysis.

---

## 📚 Table of Contents

### [Chapter 1: Domain-Specific Fine-Tuning](./chapter_01_fine_tuning/README.md)

Our first experiment asked a simple question: **Does training on domain-specific data actually help?**

We trained MobileBERT models on general conversation data vs. call center transcripts. The results were clear:

![Domain Model Success](./chapter_01_fine_tuning/results/figures/cm_2_domain_on_callcenter.png)

> **Key Finding**: Fine-tuning on call center data improved accuracy from **87.5%** to **100%**, completely eliminating false positives and false negatives on the target domain.

👉 **[Read the full experiment and results](./chapter_01_fine_tuning/README.md)**

---

### [Chapter 2: Building a Turn Detection Service](./chapter_02_service/README.md)

*(Coming Soon)*

Now that we have a highly accurate model, how do we use it?
In this chapter, we will:
- Wrap the model in a high-performance inference service
- Simulate real-time conversation streams
- Measure end-to-end latency and user experience metrics

👉 **[Go to Chapter 2](./chapter_02_service/README.md)**

---

## 🛠️ Project Structure

```
.
├── chapter_01_fine_tuning/  # Experiment: General vs. Domain Fine-Tuning
│   ├── README.md            # Detailed experiment report
│   ├── run_experiment.py    # Experiment reproduction script
│   └── results/             # Charts and metrics
│
├── chapter_02_service/      # Coming Soon: Inference Service
│
├── src/                     # Shared Source Code
│   ├── data_processor.py    # Data loading & curation
│   ├── train.py             # Model training logic
│   ├── evaluate.py          # Evaluation metrics
│   └── ...
│
├── data/                    # Shared Datasets (Gitignored)
├── models/                  # Shared Model Checkpoints (Gitignored)
└── requirements.txt         # Dependencies
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Chapter 1 Experiment**:
   ```bash
   cd chapter_01_fine_tuning
   python run_experiment.py
   ```

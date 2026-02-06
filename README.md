# Fraud Detection on Imbalanced Data and Domain Adaptation

**Big Data Mining | Subject 2**  
Master of Science in Statistics and Computer Science for Data Science  
Université Lumière Lyon 2

**Authors:** Aya Macheri, Maissa Lajimi, Mazilda Zehraoui. 
**Supervisor:** Dr. Guillaume Metzler  
**Date:** February 2026

---

## 📋 Project Overview

This project addresses two critical challenges in machine learning applied to fraud detection:

1. **Part 1: Imbalanced Learning** - Handling severe class imbalance in fraud detection datasets
2. **Part 2: Domain Adaptation** - Transferring knowledge across different data distributions using unsupervised techniques

We work with real anonymized check transaction data from ATOS Worldline, containing approximately 50 features per transaction with fraud rates ranging from 3% to 16% depending on the domain.

---

## 🎯 Key Objectives

### Part 1: Imbalanced Learning
- Compare three approaches for handling class imbalance:
  - Baseline XGBoost (no imbalance handling)
  - SMOTE + Tomek Links (hybrid resampling)
  - Cost-Sensitive XGBoost (algorithmic approach)
- Evaluate using appropriate metrics (Recall, F1-Score, AUC-ROC)
- Identify optimal approach for fraud detection

### Part 2: Domain Adaptation
- Implement self-training with pseudo-labeling
- Adapt models from source domain (16% fraud) to target domain (3% fraud)
- Leverage unlabeled target data to improve performance
- Demonstrate transfer learning in fraud detection context

---

## 📊 Results Summary

### Part 1: Imbalanced Learning (Source Dataset 0)

| Method | Recall | Precision | F1-Score | AUC-ROC | Frauds Missed |
|--------|--------|-----------|----------|---------|---------------|
| Baseline XGBoost | 94.60% | 98.04% | 96.29% | 99.89% | 117 / 2,168 |
| SMOTE + Tomek | 97.23% | 93.07% | 95.10% | 99.86% | 60 / 2,168 |
| **Cost-Sensitive** | **98.75%** | 90.45% | 94.42% | 99.88% | **27 / 2,168** |

**Winner:** Cost-Sensitive XGBoost achieves the highest recall (98.75%), missing only 27 frauds.

### Part 2: Domain Adaptation (Target Dataset 0)

| Method | Recall | F1-Score | AUC-ROC | Frauds Detected |
|--------|--------|----------|---------|-----------------|
| Source-Only | 60.26% | 50.31% | 97.65% | 370 / 614 |
| Self-Training Iter 1 | 64.17% | 49.31% | 97.59% | 394 / 614 |
| **Self-Training Iter 2** | **66.78%** | 50.28% | 97.63% | **410 / 614** |

**Improvement:** +6.52 percentage points in recall, detecting 40 additional frauds using only unlabeled target data.

---

## 📁 Project Structure

```
fraud-detection-project/
│
├── data/                              # Dataset directory (not included in repo)
│   ├── kaggle_source_cate_0_train.npy
│   ├── kaggle_source_cate_0_test.npy
│   ├── kaggle_target_cate_0_train.npy
│   ├── kaggle_target_cate_0_test.npy
│   └── ... (additional datasets)
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_exploration.ipynb          # Data exploration and analysis
│   ├── 02_imbalanced_learning.ipynb  # Part 1: Imbalanced learning experiments
│   └── 03_domain_adaptation.ipynb    # Part 2: Domain adaptation experiments
│
├── results/                           # Generated visualizations
│   ├── class_distribution_source.png
│   ├── class_distribution_target.png
│   ├── feature_distributions_source0.png
│   ├── part1_comparison.png
│   ├── part1_confusion_matrices.png
│   └── part2_domain_adaptation.png
│
├── Report_Big_Data_Mining.pdf        # Final project report
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ZehraouiMazilda/fraud-detection-imbalanced
cd fraud-detection-project
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Python Packages

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 💻 Usage

### Running the Notebooks

**Important:** Place the dataset files in the `../data/` directory relative to the notebooks folder.

#### 1. Data Exploration
```bash
jupyter notebook notebooks/01_exploration.ipynb
```
- Loads all source and target datasets
- Analyzes class distributions and imbalance ratios
- Performs data quality checks (missing values, infinite values)
- Visualizes feature distributions and domain shift

#### 2. Part 1: Imbalanced Learning
```bash
jupyter notebook notebooks/02_imbalanced_learning.ipynb
```
- Compares three imbalanced learning approaches
- Trains models on Source Dataset 0
- Evaluates using Recall, Precision, F1-Score, AUC-ROC
- Generates performance comparison visualizations

#### 3. Part 2: Domain Adaptation
```bash
jupyter notebook notebooks/03_domain_adaptation.ipynb
```
- Implements self-training with pseudo-labeling
- Performs iterative domain adaptation (2 iterations)
- Evaluates on Target Dataset 0
- Compares source-only vs adapted models
  
---

## 🔬 Methodology

### Data Preprocessing

1. **Label Extraction:** Separate first column (3-class labels) from features
2. **Binary Conversion:** Class 2 → Fraud (1), Classes 0+1 → Legitimate (0)
3. **Standardization:** StandardScaler (fit on training data only)

```python
# Example preprocessing pipeline
y_train = (X_train_raw[:, 0] == 2).astype(int)  # Binary labels
X_train = X_train_raw[:, 1:]                     # Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### Model Configuration

**XGBoost Hyperparameters:**
- `n_estimators`: 100
- `max_depth`: 6
- `learning_rate`: 0.1
- `scale_pos_weight`: 5.39 (for cost-sensitive approach)
- `random_state`: 42 (reproducibility)

### Evaluation Metrics

We deliberately avoid accuracy due to class imbalance. Instead, we use:

- **Recall (Sensitivity):** TP / (TP + FN) - Most critical for fraud detection
- **Precision:** TP / (TP + FP) - Minimizes false alarms
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Discrimination ability across all thresholds

---

## 📊 Visualizations

The project generates several visualizations saved in `results/`:

1. **Class Distribution Charts** - Source vs Target domain imbalance
2. **Feature Distribution Plots** - Sample features across datasets
3. **Performance Comparison Bar Charts** - All metrics across approaches
4. **ROC Curves** - Discrimination ability visualization
5. **Confusion Matrices** - Detailed classification breakdown
6. **Domain Adaptation Progress** - Iterative improvement tracking

---

## 🔍 Reproducibility

All experiments use fixed random seeds:
- `random_state=42` for all sklearn and XGBoost models
- `random_state=42` for SMOTE and Tomek Links
- Consistent train/test splits (provided in original datasets)

To reproduce results:
1. Use exact package versions from `requirements.txt`
2. Run notebooks in order (01 → 02 → 03)
3. Results should match report within ±0.5% due to numerical precision

---

## 📚 References

1. Zhu & Zhang (2024). "Enhancing credit card fraud detection: A neural network and SMOTE integrated approach."
2. Bounab & Zarour (2024). "Enhancing medicare fraud detection through machine learning: Addressing class imbalance with SMOTE-ENN."
3. Noviandy & Idroes (2023). "Credit card fraud detection using XGBoost-driven machine learning and data augmentation."
4. Sagiraju & Mohanty (2025). "Hyperparameter tuning of random forest using social group optimization for credit card fraud detection."
5. Cheah & Yang (2023). "Enhancing financial fraud detection through addressing class imbalance using hybrid SMOTE-GAN."
6. Nobel & Sultana (2024). "Unmasking banking fraud: Unleashing the power of machine learning and explainable AI."

---

## 🛠️ Technical Details

### Dataset Characteristics

**Source Domain:**
- 4 datasets (kaggle_source_cate_0 to 3)
- ~41,000 training samples per dataset
- ~14,000 test samples per dataset
- Class imbalance: 5:1 (16% fraud)

**Target Domain:**
- 4 datasets (kaggle_target_cate_0 to 3)
- ~63,000 training samples (unlabeled)
- ~21,000 test samples (labeled)
- Class imbalance: 30:1 (3% fraud)

**Features:**
- 50 anonymized numerical features
- Clean data (0 missing values, 0 infinite values)
- Mix of monetary, behavioral, temporal, and risk indicators

---

For questions about this project:

- **Authors:** Aya Macheri, Maissa Lajimi, Mazilda Zehraoui.
- **Program:** M2 SISE - Data Science
- **University:** Université Lumière Lyon 2
- **Year:** 2025-2026

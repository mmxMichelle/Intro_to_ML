# README.txt

## Submission Information
Course: COMP70050 — Introduction to Machine Learning
Assignment: Decision Tree Coursework (CW1)
Group Members: <<< Your Name >>>
Date: October 2025

## 0. How to Run the Code

All scripts should be run from the project root directory.

0.1. Basic Usage
To run the full pipeline (training, evaluation, pruning):

```bash
python main.py --prune
python main.py --dataset data/noisy_dataset.txt -- prune
```

This will:
1. Load both datasets from the `data/` folder.
2. Train decision trees with k-fold cross-validation (default 10-fold).
3. Evaluate performance metrics (confusion matrix, accuracy, recall, precision, F1-score).
4. Apply pruning and repeat the evaluation.
5. Save results to the `results/` directory.

0.2. Command-Line Arguments

```bash
python main.py --dataset data/clean_dataset.txt --folds 10 --prune True
```

| Argument | Description | Default |
|-----------|-------------|----------|
| `--dataset` | Path to dataset file | `data/clean_dataset.txt` |
| `--folds` | Number of folds in cross-validation | `10` |
| `--prune` | Whether to perform pruning | `False` |
| `--seed` | Random seed for splitting dataset | 42 |
| `--max_depth` | Max depth of tree | 20 |
| `--threshold` | Threshold for Information Gain determining whether a node is leaf node | 0.01 |

---


## 1. Overview
This coursework implements a Decision Tree classifier to predict indoor locations based on WiFi signal strengths collected from mobile devices.
The project follows the structure described in the coursework specification, including:

1. **Loading Data** — from `clean_dataset.txt` and `noisy_dataset.txt`.
2. **Decision Tree Construction** — recursive implementation supporting continuous attributes and multiple labels.
3. **Model Evaluation** — 10-fold cross-validation on both datasets, reporting confusion matrix, accuracy, recall, precision, and F1-measure.
4. **Tree Pruning** — based on validation error reduction, with re-evaluation after pruning.
5. (**Bonus**) **Tree Visualization** — implemented with Matplotlib (optional).

Only **NumPy**, **Matplotlib**, and **standard Python libraries** are used, as required by the coursework.


## 2. Folder Structure

decision_tree_coursework/
├── data/
│   ├── clean_dataset.txt
│   └── noisy_dataset.txt
├── src/
│   ├── decision_tree.py          # Core recursive decision tree implementation
│   ├── pruning.py                # Pruning algorithm (validation-based)
│   ├── evaluation.py             # Cross-validation and metric computation
│   └── visualization.py          # (Bonus) Tree visualization function
├── results/
│   └── <<< confusion_matrices / figures / logs >>>
├── main.py                       # Main execution script
├── requirements.txt              # Dependencies
└── README.txt                    # This file


## 3. Environment Setup

### 3.1. Python Version
The code should run on **Python 3.9–3.11**, consistent with DoC lab machines.

### 3.2. Dependencies
Only the following packages are required:

```
numpy
matplotlib
```

To install dependencies:
```bash
pip install -r requirements.txt
```


## 4. Output Description

Expected printed outputs include:

```
=== CLEAN DATASET ===
Average accuracy before pruning: 0.92
Average accuracy after pruning:  0.90

=== NOISY DATASET ===
Average accuracy before pruning: 0.85
Average accuracy after pruning:  0.88
```

Files generated (depending on implementation):
- Confusion matrices: `results/confusion_matrix_clean.npy`, `results/confusion_matrix_noisy.npy`
- Metrics summary: `results/metrics.csv`
- (Bonus) Tree visualization: `results/tree_visualization.png`

---

## 5. Reproducibility and Notes
- Random seed: <<< specify if you fixed np.random.seed(...) >>>
- Only NumPy, Matplotlib, and built-in Python modules are used.
- The implementation avoids using any external machine learning libraries (e.g., scikit-learn).
- Code tested on DoC lab Linux machines (Python 3 environment).

---

## 6. Known Issues or Limitations
<<< Mention any incomplete or partially implemented parts here, e.g. “Pruning function currently in progress” or “Visualization to be added.” >>>

---


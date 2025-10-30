# README.txt

## Submission Information
Course: COMP70050 — Introduction to Machine Learning
Assignment: Decision Tree Coursework (CW1)
Group Members: Zhongyu Cui (Decision tree construction),
               Mingxuan Ma (Evaluation),
               Yap Yoong Siew (Pruning)
Date: October 2025



## 0. How to Run the Code

All scripts should be run from the project root directory.

0.1. Basic Usage
To run the full pipeline (training, evaluation, pruning, re-evaluation):

```bash
python main.py
python main.py --prune
python main.py --dataset data/noisy_dataset.txt
python main.py --dataset data/noisy_dataset.txt --prune
```

The first and third commands will run the 10-flod cross validation on the clean dataset and noisy dataset respectively. 
This includes:
  (1) data loading and splliting (10 folds)
  (2) 1 fold for testing, 9 folds for training
  (3) train decision tree, plot tree, evaluate tree on testing fold
  (4) until all folds have been used for testing, switch to a new fold for testing, repeat (2) to (3)
  (5) return average evaluation result
The second and forth commands will run the 10-flod nested cross validation on both datasets, giving evaluation results before and after pruning.
This includes:
  (1) data loading and splliting (10 folds)
  (2) 1 fold for testing, 1 fold for validating pruning, 8 folds for training
  (3) train decision tree, plot tree, prune tree by validation fold, evaluate tree on testing fold
  (4) except the testing fold, until all folds have been used for validating, switch to a new fold for validating, repeat (2) to (3)
  (5) until all folds have been used for testing, switch to a new fold for testing, repeat (2) to (4)
  (5) return average evaluation result


0.2. Command-Line Arguments

```bash
python main.py --dataset data/clean_dataset.txt --folds 10 --prune True --seed 42 --max depth 20 --threshold 0.01
```

| Argument | Description | Default |
|-----------|-------------|----------|
| `--dataset` | Path to dataset file | `data/clean_dataset.txt` |
| `--folds` | Number of folds in cross-validation | `10` |
| `--prune` | Whether to perform pruning | `False` |
| `--seed` | Random seed for splitting dataset | `42` |
| `--max_depth` | Max depth of tree | `20` |
| `--threshold` | Threshold for Information Gain determining whether a node is leaf node | `0.01` |



## 1. Overview
This coursework implements a Decision Tree classifier to predict indoor locations based on WiFi signal strengths collected from mobile devices.
The project follows the structure described in the coursework specification, including:

(1) Loading Data — from `clean_dataset.txt` and `noisy_dataset.txt`.
(2) Decision Tree Construction — realization of DecisionTree class and fit the model.
(3) Model Evaluation — 10-fold cross-validation on both datasets, reporting confusion matrix, accuracy, recall, precision, and F1-measure.
(4) Tree Pruning — based on validation error reduction, with re-evaluation after pruning.
(5) (Bonus) Tree Visualization — implemented with Matplotlib.



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
├── result/
│   └── <<< confusion_matrices>>>
│   └── visual_tree/
│       └── <<< visualization of decision trees >>>
├── main.py                       # Main execution script
├── requirements.txt              # Dependencies
└── README.txt                    # This file


## 3. Environment Setup

### 3.1. Python Version
The code should run on **Python 3.9–3.11**, consistent with DoC lab machines.

### 3.2. Dependencies
As shown in requirements.txt, the following packages are required:

```
matplotlib==3.9.2
numpy==2.1.1
```

To install dependencies:
```bash
pip install -r requirements.txt
```


## 4. Output Description

4.1 Example of printed outputs (only for reference)

```
=== Pruning to be applied ===
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_0_pruned_2025-10-29_10-41-18.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_1_pruned_2025-10-29_10-41-19.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_2_pruned_2025-10-29_10-41-20.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_3_pruned_2025-10-29_10-41-20.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_4_pruned_2025-10-29_10-41-21.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_5_pruned_2025-10-29_10-41-22.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_6_pruned_2025-10-29_10-41-23.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_7_pruned_2025-10-29_10-41-23.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_8_pruned_2025-10-29_10-41-24.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_clean_dataset_fold_9_pruned_2025-10-29_10-41-25.png

=== data/clean_dataset.txt | 10-fold CV (pruning experiment) ===
Labels: [1 2 3 4]

-- Before pruning (trained on inner-train, evaluated on outer-test) --
Confusion Matrix (single 4×4):
 [[491   0   4   5]
 [  0 481  19   0]
 [  1  19 477   3]
 [  4   0   1 495]]
Accuracy (from aggregated CM): 0.9720
Accuracy (mean±std over folds): 0.9720 ± 0.0089
Class 1  Precision: 0.9899  Recall: 0.9820  F1: 0.9859
Class 2  Precision: 0.9620  Recall: 0.9620  F1: 0.9620
Class 3  Precision: 0.9521  Recall: 0.9540  F1: 0.9530
Class 4  Precision: 0.9841  Recall: 0.9900  F1: 0.9870
Macro Precision: 0.9720  Macro Recall: 0.9720  Macro F1: 0.9720

-- After pruning (pruned using inner-val, evaluated on outer-test) --
Confusion Matrix (single 4×4):
 [[493   0   4   3]
 [  0 481  19   0]
 [  1  19 477   3]
 [  4   0   1 495]]
Accuracy (from aggregated CM): 0.9730
Accuracy (mean±std over folds): 0.9730 ± 0.0086
Class 1  Precision: 0.9900  Recall: 0.9860  F1: 0.9880
Class 2  Precision: 0.9620  Recall: 0.9620  F1: 0.9620
Class 3  Precision: 0.9521  Recall: 0.9540  F1: 0.9530
Class 4  Precision: 0.9880  Recall: 0.9900  F1: 0.9890
Macro Precision: 0.9730  Macro Recall: 0.9730  Macro F1: 0.9730

✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\Tree_on_whole_clean_dataset_2025-10-29_10-41-26.png
```


```
=== Pruning to be applied ===
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_0_pruned_2025-10-29_10-45-38.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_1_pruned_2025-10-29_10-45-41.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_2_pruned_2025-10-29_10-45-43.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_3_pruned_2025-10-29_10-45-46.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_4_pruned_2025-10-29_10-45-50.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_5_pruned_2025-10-29_10-45-53.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_6_pruned_2025-10-29_10-45-58.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_7_pruned_2025-10-29_10-46-01.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_8_pruned_2025-10-29_10-46-05.png
✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\my_tree_on_noisy_dataset_fold_9_pruned_2025-10-29_10-46-08.png

=== data/noisy_dataset.txt | 10-fold CV (pruning experiment) ===
Labels: [1 2 3 4]

-- Before pruning (trained on inner-train, evaluated on outer-test) --
Confusion Matrix (single 4×4):
 [[393  29  25  43]
 [ 23 410  37  27]
 [ 22  42 420  31]
 [ 37  24  34 403]]
Accuracy (from aggregated CM): 0.8130
Accuracy (mean±std over folds): 0.8130 ± 0.0314
Class 1  Precision: 0.8274  Recall: 0.8020  F1: 0.8145
Class 2  Precision: 0.8119  Recall: 0.8249  F1: 0.8184
Class 3  Precision: 0.8140  Recall: 0.8155  F1: 0.8147
Class 4  Precision: 0.7996  Recall: 0.8092  F1: 0.8044
Macro Precision: 0.8132  Macro Recall: 0.8129  Macro F1: 0.8130

-- After pruning (pruned using inner-val, evaluated on outer-test) --
Confusion Matrix (single 4×4):
 [[397  28  24  41]
 [ 23 410  37  27]
 [ 21  40 423  31]
 [ 37  23  34 404]]
Accuracy (from aggregated CM): 0.8170
Accuracy (mean±std over folds): 0.8170 ± 0.0298
Class 1  Precision: 0.8305  Recall: 0.8102  F1: 0.8202
Class 2  Precision: 0.8184  Recall: 0.8249  F1: 0.8216
Class 3  Precision: 0.8166  Recall: 0.8214  F1: 0.8190
Class 4  Precision: 0.8032  Recall: 0.8112  F1: 0.8072
Macro Precision: 0.8172  Macro Recall: 0.8169  Macro F1: 0.8170

✅ Decision tree image saved to: ...\DecisionTreeCoursework\result\visual_tree\Tree_on_whole_clean_dataset_2025-10-29_10-46-09.png
```

4.2 Files generated (depending on implementation)

- Confusion matrices: `results/confusion_matrix_on_clean_dataset_before.npy`,
                      `results/confusion_matrix_on_noisy_dataset_after.npy`,
                      `results/confusion_matrix_on_clean_dataset_prune_before.npy`,
                      `results/confusion_matrix_on_noisy_dataset_prune_after.npy`
- Plots of tree during cross validation: `results/visual_tree/<<<...>>>.png`
- (Bonus) Tree visualization: `results/visual_tree/Tree_on_whole_clean_dataset_<<<current time>>>.png`



## 5. Reproducibility and Notes
- Random seed: 42
- Only NumPy, Matplotlib, and built-in Python modules are used.
- Code tested on DoC lab Linux machines (Python 3 environment).



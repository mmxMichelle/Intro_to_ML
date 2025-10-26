import numpy as np
from numpy.random import default_rng
from decision_tree import DecisionTree

#-------Evaluation Matrix----------------
def confusion_matrix(y_gold, y_prediction, class_labels=None):
    """ Compute the confusion matrix.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels.
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes.
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))
    
    class_labels=np.asarray(class_labels)
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    # for each correct class (row),
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        gold = y_gold[indices]
        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion

def accuracy_from_confusion(confusion):
    """ Compute the accuracy given the confusion matrix

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions

    Returns:
        float : the accuracy
    """

    if np.sum(confusion) > 0:
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    else:
        return 0
    

def precision(y_gold, y_prediction):
    """ Compute the precision given the confusion matrix

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions

    Returns:
        float : the precision
    """
    confusion = confusion_matrix(y_gold, y_prediction)
    p = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])
    # Compute the macro-averaged precision
    macro_p = 0.
    if len(p) > 0:
        macro_p = np.mean(p)
    return p, macro_p

def recall(y_gold, y_prediction):
    """ Compute the recall given the confusion matrix

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions

    Returns:
        float : the recall
    """
    confusion = confusion_matrix(y_gold, y_prediction)
    r = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])
    # Compute the macro-averaged recall
    macro_r = 0.
    if len(r) > 0:
        macro_r = np.mean(r)
    return r ,macro_r


def f1_score(y_gold, y_prediction):
    """ Compute the F1-score per class given the ground truth and predictions

    Also return the macro-averaged F1-score across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float)
    """

    (precisions, macro_p) = precision(y_gold, y_prediction)
    (recalls, macro_r) = recall(y_gold, y_prediction)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    # Compute the macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)

    return (f, macro_f)

#------------Cross Validation-------------

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a
            numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple)
            with two elements: a numpy array containing the train indices, and another
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        # this solution is fancy and worked for me
        # feel free to use a more verbose solution that's more readable
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds

#====================== Step 3 Core (10-fold CV) ======================

def cross_validate_decision_tree(X, y, DecisionTree,
                                 n_splits=10, seed=27,
                                 max_depth=20, threshold=1e-3):
    """
    10-fold CV on (X, y), with per-fold standardization (train-fold stats only).
    Accumulate a single 4×4 confusion matrix and compute metrics required by Step 3.
    """
    rng = default_rng(seed)
    folds = train_test_k_fold(n_splits, len(y), rng)
    labels_sorted = np.unique(y)  # 固定标签顺序（通常为 4 类）

    C_total = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=int)
    acc_per_fold = []

    # to directly reuse your existing precision/recall/f1 interface, here is a summary of all folded y_true/y_pred
    y_true_all = []   
    y_pred_all = []   

    for train_idx, test_idx in folds:
        X_tr_raw, y_tr = X[train_idx], y[train_idx]
        X_te_raw, y_te = X[test_idx], y[test_idx]

        #  normalization per fold (only training fold statistics)
        mean = X_tr_raw.mean(axis=0)
        std  = X_tr_raw.std(axis=0)
        std[std == 0] = 1.0
        X_tr = (X_tr_raw - mean) / std
        X_te = (X_te_raw - mean) / std

        #Training and prediction (follow your DecisionTree(train_x, train_y, test_x) interface
        clf = DecisionTree(X_tr, y_tr, X_te)
        nodes = clf.fit(max_depth=max_depth, threshold=threshold)
        y_pred = clf.predict(nodes)

        # Cumulative confusion matrix (fixed label order)
        C_fold = confusion_matrix(y_te, y_pred, class_labels=labels_sorted)
        C_total += C_fold

        y_true_all.append(y_te)
        y_pred_all.append(y_pred)

        acc_per_fold.append(accuracy_from_confusion(C_fold))

    # Splice all the folded labels into one column, and then calculate the indicators uniformly 
    # (consistent with the micro caliber of cumulative CM)
    y_true_all = np.concatenate(y_true_all)  
    y_pred_all = np.concatenate(y_pred_all)   

    acc_total = accuracy_from_confusion(C_total)
    p_vec, macro_p = precision(y_true_all, y_pred_all)    
    r_vec, macro_r = recall(y_true_all, y_pred_all)       
    f_vec, macro_f = f1_score(y_true_all, y_pred_all)      

    results = {
        "labels": labels_sorted,
        "confusion_matrix": C_total,                 # 4×4 (rows=true classes, columns=predictions)
        "accuracy_from_cm": float(acc_total),        
        "accuracy_mean_over_folds": float(np.mean(acc_per_fold)),
        "accuracy_std_over_folds": float(np.std(acc_per_fold, ddof=1)) if len(acc_per_fold) > 1 else 0.0,
        "precision_per_class": p_vec,
        "recall_per_class": r_vec,
        "f1_per_class": f_vec,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f)
    }
    return results

#=================== Convenience: run Step3 on a file ===================

def run_step3_for(filepath, DecisionTree,
                  max_depth=20, threshold=1e-3,
                  n_splits=10, seed=27):
    """Load data (last column = label), run 10-fold CV and print Step-3 report."""
    data = np.loadtxt(filepath)
    X = data[:, :-1]
    y = np.round(data[:, -1]).astype(int)

    res = cross_validate_decision_tree(
        X, y, DecisionTree,
        n_splits=n_splits, seed=seed,
        max_depth=max_depth, threshold=threshold
    )

    print(f"\n=== {filepath} | {n_splits}-fold CV ===")
    print("Labels:", res["labels"])
    print("Confusion Matrix (single 4×4):\n", res["confusion_matrix"])
    print(f"Accuracy (from aggregated CM): {res['accuracy_from_cm']:.4f}")
    print(f"Accuracy (mean±std over folds): {res['accuracy_mean_over_folds']:.4f} ± {res['accuracy_std_over_folds']:.4f}")
    for i, lab in enumerate(res["labels"]):
        print(f"Class {lab}  Precision: {res['precision_per_class'][i]:.4f}  "
              f"Recall: {res['recall_per_class'][i]:.4f}  F1: {res['f1_per_class'][i]:.4f}")
    print(f"Macro Precision: {res['macro_precision']:.4f}  "
          f"Macro Recall: {res['macro_recall']:.4f}  Macro F1: {res['macro_f1']:.4f}")
    return res


clean_res = run_step3_for('clean_dataset.txt', DecisionTree, max_depth=20, threshold=1e-3, n_splits=10, seed=27)
noisy_res = run_step3_for('noisy_dataset.txt', DecisionTree, max_depth=20, threshold=1e-3, n_splits=10, seed=27)
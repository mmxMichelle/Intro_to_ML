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

# -------Splitting utilities (outer K-fold + inner hold-out val)------

def k_fold_test_indices(n_splits, n_instances, rng=None):
    """Return list of length n_splits; each item is test indices for an outer fold."""
    # [ADD] safer RNG creation
    if rng is None:
        rng = default_rng()
    shuffled = rng.permutation(n_instances)
    return np.array_split(shuffled, n_splits)

def split_indices_stratified(idx_all, y_full, val_ratio=0.1, rng=None):
    """
    Given a pool of indices `idx_all` (e.g., one outer fold's training set),
    split it into train' and val in a stratified way according to y_full.
    Returns: trainp_idx, val_idx (disjoint subsets of idx_all).
    """
    if rng is None:
        rng = default_rng()
    idx_all = np.asarray(idx_all)
    classes = np.unique(y_full[idx_all])

    val_parts, train_parts = [], []
    for c in classes:
        idx_c = idx_all[y_full[idx_all] == c]
        idx_c = rng.permutation(idx_c)
        n_c_val = int(np.round(len(idx_c) * val_ratio))
        # keep at least 1 sample in train' if possible
        n_c_val = min(n_c_val, len(idx_c) - 1) if len(idx_c) > 1 else 0
        val_parts.append(idx_c[:n_c_val])
        train_parts.append(idx_c[n_c_val:])

    val_idx   = np.concatenate(val_parts)   if len(val_parts)   else np.array([], dtype=int)
    train_idx = np.concatenate(train_parts) if len(train_parts) else np.array([], dtype=int)

    # rare edge guard
    if len(train_idx) == 0 and len(val_idx) > 1:
        train_idx, val_idx = val_idx[1:], val_idx[:1]
    return train_idx, val_idx

# ---------Hyper-parameter grid----------

def hyperparam_grid():
    # minimal grid; adjust if needed
    depths = [6, 10, 14, 20]
    ths    = [1e-4, 5e-4, 1e-3]
    for d in depths:
        for t in ths:
            yield d, t



#====================== Step 3 Core (10-fold CV) ======================
def cross_validate_decision_tree_with_val(
    X, y, DecisionTree,
    n_splits=10, seed=27, val_ratio=0.1
):
    """
    Outer 10-fold CV for final evaluation; inside each outer fold,
    carve a stratified validation from the outer training set to select
    (max_depth, threshold). Per-fold standardization uses only training stats
    to prevent leakage.
    """
    rng = default_rng(seed)
    outer_tests = k_fold_test_indices(n_splits, len(y), rng)
    labels_sorted = np.unique(y)

    C_total = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=int)
    acc_per_fold = []
    y_true_all, y_pred_all = [], []  # [ADD] aggregate labels for macro metrics

    for k in range(n_splits):
        test_idx = outer_tests[k]
        trainval_idx = np.concatenate([outer_tests[i] for i in range(n_splits) if i != k])

        # ---- inner stratified split: train' and val (Lab1-style) ----
        trp_idx, val_idx = split_indices_stratified(trainval_idx, y_full=y, val_ratio=val_ratio, rng=rng)

        # ---- standardize using train' only; apply to train' and val ----
        X_trp_raw, y_trp = X[trp_idx], y[trp_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]
        mean_trp = X_trp_raw.mean(axis=0)
        std_trp  = X_trp_raw.std(axis=0); std_trp[std_trp == 0] = 1.0
        X_trp = (X_trp_raw - mean_trp) / std_trp
        X_val = (X_val_raw - mean_trp) / std_trp

        # ---- hyper-parameter selection on (train', val) ----
        best_acc, best_hp = -1.0, (None, None)
        for max_depth, threshold in hyperparam_grid():
            clf = DecisionTree(X_trp, y_trp, X_val)
            nodes = clf.fit(max_depth=max_depth, threshold=threshold)
            y_val_pred = clf.predict(nodes)
            acc_val = np.mean(y_val_pred == y_val)
            if acc_val > best_acc:
                best_acc, best_hp = acc_val, (max_depth, threshold)

        # ---- retrain on (train'∪val) with best HP; standardize with its stats ----
        tr_full_idx = np.concatenate([trp_idx, val_idx])
        X_tr_full_raw, y_tr_full = X[tr_full_idx], y[tr_full_idx]
        X_te_raw, y_te           = X[test_idx],   y[test_idx]
        mean_full = X_tr_full_raw.mean(axis=0)
        std_full  = X_tr_full_raw.std(axis=0); std_full[std_full == 0] = 1.0
        X_tr_full = (X_tr_full_raw - mean_full) / std_full
        X_te      = (X_te_raw      - mean_full) / std_full

        md, th = best_hp
        clf = DecisionTree(X_tr_full, y_tr_full, X_te)
        nodes = clf.fit(max_depth=md, threshold=th)
        y_te_pred = clf.predict(nodes)

        # ---- accumulate confusion matrix & collect labels ----
        C_fold = confusion_matrix(y_te, y_te_pred, class_labels=labels_sorted)
        C_total += C_fold
        acc_per_fold.append(accuracy_from_confusion(C_fold))
        y_true_all.append(y_te); y_pred_all.append(y_te_pred)

    # ---- final metrics ----
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    acc_total = accuracy_from_confusion(C_total)
    p_vec, macro_p = precision(y_true_all, y_pred_all)
    r_vec, macro_r = recall(y_true_all, y_pred_all)
    f_vec, macro_f = f1_score(y_true_all, y_pred_all)

    return {
        "labels": labels_sorted,
        "confusion_matrix": C_total,                               # single 4×4 CM (rows=GT, cols=Pred)
        "accuracy_from_cm": float(acc_total),                      # micro accuracy from aggregated CM
        "accuracy_mean_over_folds": float(np.mean(acc_per_fold)),  # mean acc across folds
        "accuracy_std_over_folds": float(np.std(acc_per_fold, ddof=1)) if len(acc_per_fold) > 1 else 0.0,
        "precision_per_class": p_vec,
        "recall_per_class": r_vec,
        "f1_per_class": f_vec,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
    }


#=================== Convenience: run Step3 on a file ===================

def run_step3_for(filepath, DecisionTree,
                  n_splits=10, seed=27, val_ratio=0.1):
    """Load data (last column = label), run outer 10-fold CV with inner validation, and print a Step-3 report."""
    data = np.loadtxt(filepath)
    X = data[:, :-1]
    y = np.round(data[:, -1]).astype(int)

    res = cross_validate_decision_tree_with_val(
        X, y, DecisionTree, n_splits=n_splits, seed=seed, val_ratio=val_ratio
    )

    print(f"\n=== {filepath} | {n_splits}-fold (with inner validation, val_ratio={val_ratio}) ===")
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

# =========================
# Example usage 
# =========================
clean_res = run_step3_for('clean_dataset.txt', DecisionTree, n_splits=10, seed=27, val_ratio=0.1)
noisy_res = run_step3_for('noisy_dataset.txt',  DecisionTree, n_splits=10, seed=27, val_ratio=0.1)
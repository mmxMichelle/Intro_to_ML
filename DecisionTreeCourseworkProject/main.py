import numpy as np
from src.decision_tree import DecisionTree
from src.evaluation import cross_validate_decision_tree, cross_validate_pruned_tree
from src.visualization import plot_decision_tree
import argparse

def load_dataset(path):
    """Loads dataset as a NumPy array."""
    return np.loadtxt(path)

def parse_args():
    parser = argparse.ArgumentParser(description="Decision Tree Coursework Runner")
    parser.add_argument("--dataset", type=str, default="data/clean_dataset.txt",
                        help="Path to dataset (txt format)")
    parser.add_argument("--folds", type=int, default=10,
                        help="Number of folds for cross-validation")
    parser.add_argument("--prune", action="store_true",
                        help="Apply post-pruning if set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting dataset")
    parser.add_argument("--max_depth", type=int, default=20,
                        help="Max depth of tree")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Threshold for Information Gain determining whether a node is leaf node")
    return parser.parse_args()

def main():
    args = parse_args()

    filename = args.dataset
    filename = filename[5:-4]

    data = load_dataset(args.dataset)
    X = data[:, :-1]
    y = np.round(data[:, -1]).astype(int)

    # No pruning, train and evaluate
    if not args.prune:
        print("\n=== No pruning to be applied ===")
        res_before = cross_validate_decision_tree(
            X, y, DecisionTree, filename=filename,
            n_splits=args.folds, seed=args.seed,
            max_depth=args.max_depth, threshold=args.threshold
        )
        print(f"\n=== {args.dataset} | {args.folds}-fold CV ===")
        print("Labels:", res_before["labels"])
        print("Confusion Matrix (single 4×4):\n", res_before["confusion_matrix"])
        print(f"Accuracy (from aggregated CM): {res_before['accuracy_from_cm']:.4f}")
        print(
            f"Accuracy (mean±std over folds): {res_before['accuracy_mean_over_folds']:.4f} ± {res_before['accuracy_std_over_folds']:.4f}")
        for i, lab in enumerate(res_before["labels"]):
            print(f"Class {lab}  Precision: {res_before['precision_per_class'][i]:.4f}  "
                  f"Recall: {res_before['recall_per_class'][i]:.4f}  F1: {res_before['f1_per_class'][i]:.4f}")
        print(f"Macro Precision: {res_before['macro_precision']:.4f}  "
              f"Macro Recall: {res_before['macro_recall']:.4f}  Macro F1: {res_before['macro_f1']:.4f}")

        # Save results
        np.save(f"result/confusion_matrix_on_{filename}_before.npy", res_before["confusion_matrix"])
        print()


    # Have pruning, train, evaluate, prune and evaluate
    if args.prune:
        print("\n=== Pruning to be applied ===")
        res_after = cross_validate_pruned_tree(
            X, y, DecisionTree, filename=filename,
            n_splits=args.folds, seed=args.seed,
            max_depth=args.max_depth, threshold=args.threshold
        )
        # The returned results now contain paired before/after metrics for the pruning experiment.
        print(f"\n=== {args.dataset} | {args.folds}-fold CV (pruning experiment) ===")
        print("Labels:", res_after["labels"])

        # Before pruning (apples-to-apples — same inner-trained tree evaluated on outer test)
        print("\n-- Before pruning (trained on inner-train, evaluated on outer-test) --")
        print("Confusion Matrix (single 4×4):\n", res_after["confusion_matrix_before_pruning"])
        print(f"Accuracy (from aggregated CM): {res_after['accuracy_from_cm_before']:.4f}")
        print(f"Accuracy (mean±std over folds): {res_after['accuracy_mean_over_folds_before']:.4f} ± {res_after['accuracy_std_over_folds_before']:.4f}")
        for i, lab in enumerate(res_after["labels"]):
            print(f"Class {lab}  Precision: {res_after['precision_per_class_before'][i]:.4f}  "
                  f"Recall: {res_after['recall_per_class_before'][i]:.4f}  F1: {res_after['f1_per_class_before'][i]:.4f}")
        print(f"Macro Precision: {res_after['macro_precision_before']:.4f}  "
              f"Macro Recall: {res_after['macro_recall_before']:.4f}  Macro F1: {res_after['macro_f1_before']:.4f}")

        # After pruning (pruned tree evaluated on same outer-test)
        print("\n-- After pruning (pruned using inner-val, evaluated on outer-test) --")
        print("Confusion Matrix (single 4×4):\n", res_after["confusion_matrix"])
        print(f"Accuracy (from aggregated CM): {res_after['accuracy_from_cm_after']:.4f}")
        print(f"Accuracy (mean±std over folds): {res_after['accuracy_mean_over_folds_after']:.4f} ± {res_after['accuracy_std_over_folds_after']:.4f}")
        for i, lab in enumerate(res_after["labels"]):
            print(f"Class {lab}  Precision: {res_after['precision_per_class'][i]:.4f}  "
                  f"Recall: {res_after['recall_per_class'][i]:.4f}  F1: {res_after['f1_per_class'][i]:.4f}")
        print(f"Macro Precision: {res_after['macro_precision']:.4f}  "
              f"Macro Recall: {res_after['macro_recall']:.4f}  Macro F1: {res_after['macro_f1']:.4f}")
        print()

        # Save results: save both paired confusion matrices for the pruning experiment
        np.save(f"result/confusion_matrix_on_{filename}_prune_experiment_before.npy", res_after["confusion_matrix_before_pruning"])
        np.save(f"result/confusion_matrix_on_{filename}_prune_experiment_after.npy", res_after["confusion_matrix"])

    # Bonus. Train a decision tree on the whole clean dataset and plot the tree
    # load data
    data = load_dataset('data/clean_dataset.txt')
    X = data[:, :-1]
    y = np.round(data[:, -1]).astype(int)
    # normalization
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / X_std
    # train decision tree and plot tree
    tree_whole_clean_dataset = DecisionTree(X, y, y)
    tree_whole_clean_dataset = tree_whole_clean_dataset.fit(args.max_depth, args.threshold)
    plot_decision_tree(tree_whole_clean_dataset, save_name_prefix='Tree_on_whole_clean_dataset')

if __name__ == "__main__":
    main()

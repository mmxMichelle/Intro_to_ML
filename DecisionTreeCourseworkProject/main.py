import numpy as np
from src.decision_tree import DecisionTree
from src.evaluation import cross_validate_decision_tree, cross_validate_pruned_tree
# from src.pruning import prune_tree
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
    parser.add_argument("--show", action="store_true",
                        help="Show the plot of trees visualized")
    return parser.parse_args()

def main():
    args = parse_args()

    filename = args.dataset
    filename = filename[5:-4]

    data = load_dataset(args.dataset)
    X = data[:, :-1]
    y = np.round(data[:, -1]).astype(int)

    # Before pruning
    print("\n=== Before pruning ===")
    res_before = cross_validate_decision_tree(
        X, y, DecisionTree, filename=filename, show_trees=args.show,
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
    print()


    # After pruning (if enabled)
    if args.prune:
        print("\n=== After pruning ===")
        res_after = cross_validate_pruned_tree(
            X, y, DecisionTree, filename=filename, show_trees=args.show,
            n_splits=args.folds, seed=args.seed,
            max_depth=args.max_depth, threshold=args.threshold
        )
        print(f"\n=== {args.dataset} | {args.folds}-fold CV ===")
        print("Labels:", res_after["labels"])
        print("Confusion Matrix (single 4×4):\n", res_after["confusion_matrix"])
        print(f"Accuracy (from aggregated CM): {res_after['accuracy_from_cm']:.4f}")
        print(
            f"Accuracy (mean±std over folds): {res_after['accuracy_mean_over_folds']:.4f} ± {res_after['accuracy_std_over_folds']:.4f}")
        for i, lab in enumerate(res_after["labels"]):
            print(f"Class {lab}  Precision: {res_after['precision_per_class'][i]:.4f}  "
                  f"Recall: {res_after['recall_per_class'][i]:.4f}  F1: {res_after['f1_per_class'][i]:.4f}")
        print(f"Macro Precision: {res_after['macro_precision']:.4f}  "
              f"Macro Recall: {res_after['macro_recall']:.4f}  Macro F1: {res_after['macro_f1']:.4f}")
        print()

        # Save results
        np.save(f"result/confusion_matrix_on_{filename}_after.npy", res_after["confusion_matrix"])

    # Save results
    np.save(f"result/confusion_matrix_on_{filename}_before.npy", res_before["confusion_matrix"])

if __name__ == "__main__":
    main()

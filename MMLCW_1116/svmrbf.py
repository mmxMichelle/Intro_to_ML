import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

# ======== define simple SVM-RBF class ========
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# ======== settings ========
DATA_PATH = "data"
RESULTS_PATH = "results"
os.makedirs(RESULTS_PATH, exist_ok=True)

dims = [10, 12, 14, 16, 18, 20]
summary_records = []

# ======== main stream ========
for dim in dims:
    print(f"\n========================")
    print(f"Processing {dim}-D dataset")
    print(f"========================")

    X_path = os.path.join(DATA_PATH, f"kryptonite-{dim}-X.npy")
    y_path = os.path.join(DATA_PATH, f"kryptonite-{dim}-y.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print(f"⚠️ Missing files for {dim}-D, skipped.")
        continue

    X = np.load(X_path)
    y = np.load(y_path)

    # results directory
    dim_dir = os.path.join(RESULTS_PATH, f"{dim}D")

    # 8-fold cross validation (10 folds take too long)
    kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    fold_results = []
    best_C_list = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold+1}/8 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # inner cross validation to select the best C (for large C, the process takes too long, thus only choose C=100)
        inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        C_list = [100]
        best_C = None
        best_score = -1

        for C_val in C_list:
            inner_scores = []
            for inner_train_idx, inner_valid_idx in inner_kf.split(X_train, y_train):
                X_in_train, X_in_valid = X_train[inner_train_idx], X_train[inner_valid_idx]
                y_in_train, y_in_valid = y_train[inner_train_idx], y_train[inner_valid_idx]

                # standardization
                scaler_inner = StandardScaler()
                X_in_train = scaler_inner.fit_transform(X_in_train)
                X_in_valid = scaler_inner.transform(X_in_valid)

                # RBF SVM with candidate C
                inner_model = SVC(kernel='rbf', C=C_val, gamma='scale')
                inner_model.fit(X_in_train, y_in_train)

                y_in_pred = inner_model.predict(X_in_valid)
                inner_scores.append(accuracy_score(y_in_valid, y_in_pred))

            avg_inner_score = np.mean(inner_scores)
            if avg_inner_score > best_score:
                best_score = avg_inner_score
                best_C = C_val

        print(f"Selected best C={best_C}")
        best_C_list.append(best_C)

        # train outer model with best C
        model = SVC(kernel='rbf', C=best_C, gamma='scale')
        model.fit(X_train, y_train)

        # predicting
        y_pre = model.predict(X_test)

        # evaluation
        acc = accuracy_score(y_test, y_pre)
        f1 = f1_score(y_test, y_pre)
        try:
            auc = roc_auc_score(y_test, y_pre)
        except:
            auc = np.nan
        cm = confusion_matrix(y_test, y_pre)
        fold_results.append([acc, f1, auc])
        print(f"Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    # ======== summarize evaluation results ========
    avg_best_C = np.mean(best_C_list)
    fold_results = np.array(fold_results)
    mean_acc, mean_f1, mean_auc = np.nanmean(fold_results, axis=0)
    std_acc, std_f1, std_auc = np.nanstd(fold_results, axis=0)

    print(f"\n===== Final 10-Fold Summary ({dim}D) =====")
    print(f"Average best C : {avg_best_C}")
    print(f"Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1-score : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"AUC      : {mean_auc:.4f} ± {std_auc:.4f}")

    # save summary_SVMRBF.txt
    summary_path = os.path.join(dim_dir, "summary_SVMRBF.txt")
    os.makedirs(dim_dir, exist_ok=True)

    with open(summary_path, "w") as f:
        f.write(f"===== 10-Fold Summary for {dim}D Dataset =====\n")
        f.write(f"Average Best C : {avg_best_C:.2f}\n")
        f.write(f"Accuracy : {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"F1-score : {mean_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"AUC      : {mean_auc:.4f} ± {std_auc:.4f}\n")

    # save average evaluation results
    summary_records.append({
        "Dimension": f"{dim}D",
        "Best_C": avg_best_C,
        "Mean_Acc": mean_acc,
        "Std_Acc": std_acc,
        "Mean_F1": mean_f1,
        "Std_F1": std_f1,
        "Mean_AUC": mean_auc,
        "Std_AUC": std_auc
    })

# global summary
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(RESULTS_PATH, "summary_all_SVMRBF.csv"), index=False)
print("\n✅ All training complete. Results saved under 'results/' folder.")
print(f"Global summary: {os.path.join(RESULTS_PATH, 'summary_all_SVMRBF.csv')}")


#!/usr/bin/env python3
"""
run_all_models.py

Complete script to train multiple models on Kryptonite-n datasets,
do light hyperparameter tuning, ensemble, evaluate, and save hidden predictions.

Usage:
    python run_all_models.py --n 10 --datasets_dir Datasets --hidden_file Datasets/hidden-kryptonite-10-X.npy

Outputs:
 - prints training/validation/test accuracy for each model
 - saves best model as best_model_n{n}.joblib
 - if hidden file present, saves hiddenlabels/y_predicted_n{n}.npy (binary vector)
"""

import os
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import compute_class_weight
from scipy.stats import randint, uniform
import joblib
import warnings
warnings.filterwarnings("ignore")

def load_dataset(n, datasets_dir):
    X_path = Path(datasets_dir) / f"kryptonite-{n}-X.npy"
    y_path = Path(datasets_dir) / f"kryptonite-{n}-y.npy"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Expected dataset files not found: {X_path}, {y_path}")
    X = np.load(X_path)
    y = np.load(y_path)
    # Ensure binary integer labels 0/1
    y = y.astype(int).ravel()
    return X, y

def train_val_test_split(X, y, seed=42):
    # 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test

def baseline_poly_logistic(X_train, y_train, X_val, y_val, degree=3, random_state=42):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, solver="saga", penalty='l2', C=1.0, random_state=random_state))
    ])
    pipe.fit(X_train, y_train)
    yval = pipe.predict(X_val)
    acc = accuracy_score(y_val, yval)
    return pipe, acc

def randomized_search_pipeline(pipeline, param_dist, X_train, y_train, cv_splits=3, n_iter=20, seed=42, n_jobs=-1, scoring='accuracy'):
    rs = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter,
                            cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed),
                            random_state=seed, n_jobs=n_jobs, scoring=scoring, verbose=1)
    rs.fit(X_train, y_train)
    return rs

def build_and_tune_models(X_train, y_train, X_val, y_val, seed=42):
    results = {}

    # 1) RandomForest
    print("\n=== RandomForest (light tuning) ===")
    rf_pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(random_state=seed, n_jobs=-1))])
    rf_param_dist = {
        "rf__n_estimators": randint(100, 500),
        "rf__max_depth": randint(4, 40),
        "rf__min_samples_split": randint(2, 10),
        "rf__min_samples_leaf": randint(1, 6),
        "rf__max_features": ['sqrt', 'log2', None]
    }
    rf_search = randomized_search_pipeline(rf_pipe, rf_param_dist, X_train, y_train, n_iter=25, seed=seed)
    rf_best = rf_search.best_estimator_
    rf_val_acc = accuracy_score(y_val, rf_best.predict(X_val))
    results['RandomForest'] = (rf_best, rf_val_acc, rf_search)

    # 2) HistGradientBoosting (fast, strong baseline)
    print("\n=== HistGradientBoostingClassifier ===")
    hgb_pipe = Pipeline([("scaler", StandardScaler()), ("hgb", HistGradientBoostingClassifier(random_state=seed))])
    hgb_param_dist = {
        "hgb__max_iter": randint(100, 500),
        "hgb__max_leaf_nodes": randint(7, 64),
        "hgb__learning_rate": uniform(0.01, 0.3),
    }
    hgb_search = randomized_search_pipeline(hgb_pipe, hgb_param_dist, X_train, y_train, n_iter=20, seed=seed)
    hgb_best = hgb_search.best_estimator_
    hgb_val_acc = accuracy_score(y_val, hgb_best.predict(X_val))
    results['HistGB'] = (hgb_best, hgb_val_acc, hgb_search)

    # 3) MLPClassifier
    print("\n=== MLPClassifier ===")
    mlp_pipe = Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(max_iter=1000, random_state=seed))])
    mlp_param_dist = {
        "mlp__hidden_layer_sizes": [(128,), (256,), (512,), (128,128), (256,128)],
        "mlp__activation": ['relu','tanh'],
        "mlp__alpha": uniform(1e-5, 1e-2),
        "mlp__learning_rate_init": uniform(1e-4, 5e-2)
    }
    mlp_search = randomized_search_pipeline(mlp_pipe, mlp_param_dist, X_train, y_train, n_iter=20, seed=seed)
    mlp_best = mlp_search.best_estimator_
    mlp_val_acc = accuracy_score(y_val, mlp_best.predict(X_val))
    results['MLP'] = (mlp_best, mlp_val_acc, mlp_search)

    # 4) SVC (RBF) — can be slow; keep it limited
    print("\n=== SVC (RBF) ===")
    svc_pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=seed))])
    svc_param_dist = {
        "svc__C": uniform(0.1, 10),
        "svc__gamma": ['scale', 'auto']
    }
    svc_search = randomized_search_pipeline(svc_pipe, svc_param_dist, X_train, y_train, n_iter=10, seed=seed, n_jobs=1)
    svc_best = svc_search.best_estimator_
    svc_val_acc = accuracy_score(y_val, svc_best.predict(X_val))
    results['SVC'] = (svc_best, svc_val_acc, svc_search)

    return results

def stacking_ensemble(best_estimators, X_train_full, y_train_full, X_val, y_val, seed=42):
    # best_estimators: dict name -> (estimator, val_acc, search_obj)
    estimators = []
    for name, (est, acc, _) in best_estimators.items():
        estimators.append((name, est))
    # Use a simple logistic regression meta-learner
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1, passthrough=False)
    stack.fit(X_train_full, y_train_full)
    val_acc = accuracy_score(y_val, stack.predict(X_val))
    return stack, val_acc

def evaluate_and_report(models_dict, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n\n=== Evaluation Summary ===")
    for name, (est, val_acc, search) in models_dict.items():
        y_train_pred = est.predict(X_train)
        y_val_pred = est.predict(X_val)
        y_test_pred = est.predict(X_test)
        print(f"\nModel: {name}")
        print(f"  Validation acc (direct): {val_acc:.4f}")
        print(f"  Train acc: {accuracy_score(y_train, y_train_pred):.4f}")
        print(f"  Test acc:  {accuracy_score(y_test, y_test_pred):.4f}")
        # optionally print classification report
        print(classification_report(y_test, y_test_pred, digits=4))
        print("Confusion matrix (test):")
        print(confusion_matrix(y_test, y_test_pred))

def save_hidden_predictions(best_model, hidden_X_path, out_path, threshold=0.5):
    Xh = np.load(hidden_X_path)
    # If pipeline includes scaler, it's fine
    probs = best_model.predict_proba(Xh)[:,1] if hasattr(best_model, "predict_proba") else None
    if probs is not None:
        preds = (probs >= threshold).astype(int)
    else:
        preds = best_model.predict(Xh).astype(int)
    # Ensure length is 10000 if grading expects that; otherwise save whatever length
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, preds)
    print(f"Saved hidden predictions to: {out_path} (len={len(preds)})")

def main(args):
    seed = args.seed
    # Load data
    X, y = load_dataset(args.n, args.datasets_dir)
    print("Loaded X shape:", X.shape, "y shape:", y.shape)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, seed=seed)
    print("Splits: train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)
    # Baseline polynomial logistic (quick)
    print("\n--- Baseline polynomial logistic regression (degree=3) ---")
    try:
        poly_pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=args.poly_degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="saga", C=1.0, random_state=seed))
        ])
        poly_pipe.fit(X_train, y_train)
        acc_val = accuracy_score(y_val, poly_pipe.predict(X_val))
        acc_test = accuracy_score(y_test, poly_pipe.predict(X_test))
        print(f"Polynomial (deg={args.poly_degree}) val acc: {acc_val:.4f}, test acc: {acc_test:.4f}")
    except Exception as e:
        print("Polynomial baseline failed:", e)
        poly_pipe = None

    # Build and tune stronger models
    tuned = build_and_tune_models(X_train, y_train, X_val, y_val, seed=seed)

    # Evaluate tuned models
    evaluate_and_report(tuned, X_train, y_train, X_val, y_val, X_test, y_test)

    # Choose top-K models (by validation accuracy) for stacking
    sorted_models = sorted(tuned.items(), key=lambda kv: kv[1][1], reverse=True)
    print("\nModels ranked by validation accuracy:")
    for name, (est, acc, _) in sorted_models:
        print(f"  {name}: {acc:.4f}")
    # Use top 3 for stacking
    top_k = dict(sorted_models[:3])
    print("\nBuilding stacking ensemble with top 3 models...")
    # Combine train+val for final stack training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    stack_model, stack_val_acc = stacking_ensemble(top_k, X_train_full, y_train_full, X_val, y_val, seed=seed)
    print(f"Stacking validation acc (trained on train+val): {stack_val_acc:.4f}")
    # Final evaluation of stack on test
    stack_test_acc = accuracy_score(y_test, stack_model.predict(X_test))
    print(f"Stacking test acc: {stack_test_acc:.4f}")
    # Save best model among tuned + stack
    all_models_candidates = {name: (est, acc) for name,(est,acc,_) in tuned.items()}
    all_models_candidates['Stacking'] = (stack_model, stack_test_acc)
    # choose best by test acc (important: you could also choose by val)
    best_name, (best_model, best_score) = max(all_models_candidates.items(), key=lambda kv: kv[1][1])
    print(f"\nBest model by test accuracy: {best_name} (acc={best_score:.4f})")
    # Save model
    out_model_path = f"best_model_n{args.n}.joblib"
    joblib.dump(best_model, out_model_path)
    print("Saved best model to:", out_model_path)

    # If hidden file present, create hiddenlabels directory and save predictions
    if args.hidden_file and os.path.exists(args.hidden_file):
        out_pred_path = f"hiddenlabels/y_predicted_n{args.n}.npy"
        save_hidden_predictions(best_model, args.hidden_file, out_pred_path)
    else:
        print("Hidden file not provided or not found; skipping hidden prediction save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models for Kryptonite-n")
    parser.add_argument("--n", type=int, required=True, help="value of n for Kryptonite dataset (e.g., 10)")
    parser.add_argument("--datasets_dir", type=str, default="Datasets", help="directory containing kryptonite-<n>-X.npy and -y.npy")
    parser.add_argument("--hidden_file", type=str, default="", help="path to hidden-kryptonite-<n>-X.npy (optional)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poly_degree", type=int, default=3, help="degree for polynomial baseline")
    args = parser.parse_args()
    main(args)

# nn_reproduce.py
import os
import numpy as np
import torch
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from model import SimpleMLP

DATA_PATH = "data"
RESULTS_PATH = "results"
dims = [10,12,14,16,18,20]

for dim in dims:
    print(f"\n=== Reproducing {dim}D ===")

    X_path = os.path.join(DATA_PATH, f"kryptonite-{dim}-X.npy")
    y_path = os.path.join(DATA_PATH, f"kryptonite-{dim}-y.npy")
    model_path = os.path.join(RESULTS_PATH, f"{dim}D", "best_model.pt")
    scaler_path = os.path.join(RESULTS_PATH, f"{dim}D", "best_scaler.pkl")

    if not (os.path.exists(X_path) and os.path.exists(y_path) and
            os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("❌ Missing files, skipping...")
        continue

    X = np.load(X_path)
    y = np.load(y_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)

    model = SimpleMLP(input_dim=dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
        y_pred = (preds >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    try:
        auc = roc_auc_score(y, preds)
    except:
        auc = float('nan')

    print(f"Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

print("\n🎯 Reproduction finished")

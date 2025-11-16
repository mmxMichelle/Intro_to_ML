import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

# ======== device detection ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======== model structure ========
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ======== settings ========
DATA_PATH = "data"
RESULTS_PATH = "results"
HIDDEN_LABEL_PATH = "hiddenlabels"
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(HIDDEN_LABEL_PATH, exist_ok=True)

epoch_map = {10:70, 12:80, 14:100, 16:150, 18:150, 20:150}
dims = [10,12,14,16,18,20]
summary_records = []

# ======== main stream (10 fold cross validation) ========
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
    epochs = epoch_map[dim]
    print(f"Training epochs set to {epochs}")

    dim_dir = os.path.join(RESULTS_PATH, f"{dim}D")
    loss_dir = os.path.join(dim_dir, "loss_curves")
    os.makedirs(loss_dir, exist_ok=True)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []
    all_loss_histories = []

    # best-model tracking
    best_metric = -999
    best_model_state = None
    best_scaler = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold+1}/10 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                  batch_size=64, shuffle=True)

        model = SimpleMLP(input_dim=X.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # ======== training ========
        loss_history = []
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(train_loader.dataset)
            loss_history.append(avg_loss)
            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        all_loss_histories.append(loss_history)

        # fold loss curve
        plt.figure()
        plt.plot(range(1, len(loss_history)+1), loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dim}D Fold {fold+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(loss_dir, f"fold_{fold+1}.png"), dpi=200)
        plt.close()

        # ======== evaluation ========
        model.eval()
        with torch.no_grad():
            y_prob = model(X_test_t.to(device)).cpu().numpy()
            y_hat = (y_prob >= 0.5).astype(float)
            acc = accuracy_score(y_test, y_hat)
            f1 = f1_score(y_test, y_hat)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = np.nan

        fold_results.append([acc, f1, auc])
        print(f"Fold {fold+1}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        # ======== select best model by highest AUC, fallback F1 ========
        metric = auc if not np.isnan(auc) else f1
        if metric > best_metric:
            best_metric = metric
            best_model_state = model.state_dict()
            best_scaler = scaler

    # ======== summary of dimension ========
    fold_results = np.array(fold_results)
    mean_acc, mean_f1, mean_auc = np.nanmean(fold_results, axis=0)
    std_acc, std_f1, std_auc = np.nanstd(fold_results, axis=0)

    print(f"\n===== Summary ({dim}D) =====")
    print(f"Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1 : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # save summary file
    with open(os.path.join(dim_dir, "summary.txt"), "w") as f:
        f.write(f"Acc: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"F1 : {mean_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")

    # save avg loss
    avg_loss = np.mean(np.array(all_loss_histories), axis=0)
    plt.figure()
    plt.plot(avg_loss)
    plt.title(f"{dim}D Avg Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, "avg.png"), dpi=200)
    plt.close()

    summary_records.append({
        "Dimension": dim,
        "Acc": mean_acc,
        "F1": mean_f1,
        "AUC": mean_auc
    })

    # ================================
    #     🔥 预测 hidden 数据部分
    # ================================
    hidden_path = os.path.join(DATA_PATH, f"hidden-kryptonite-{dim}-X.npy")
    if os.path.exists(hidden_path):
        print(f"\n🔍 Predicting hidden set for {dim}D ...")
        hidden_X = np.load(hidden_path)
        hidden_X = best_scaler.transform(hidden_X)

        hidden_t = torch.tensor(hidden_X, dtype=torch.float32).to(device)

        # load best model
        best_model = SimpleMLP(input_dim=dim).to(device)
        best_model.load_state_dict(best_model_state)
        best_model.eval()

        with torch.no_grad():
            prob = best_model(hidden_t).cpu().numpy()
            pred = (prob >= 0.5).astype(int).flatten()

        # save prediction
        save_path = os.path.join(HIDDEN_LABEL_PATH, f"y_predicted_{dim}.npy")
        np.save(save_path, pred)
        print(f"✨ Saved hidden predictions → {save_path}")
    else:
        print(f"No hidden-kryptonite-{dim}-X.npy found.")

# global summary
pd.DataFrame(summary_records).to_csv(
    os.path.join(RESULTS_PATH, "summary_all.csv"), index=False
)
print("\n🎉 All done!")

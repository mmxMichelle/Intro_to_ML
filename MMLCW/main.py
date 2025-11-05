import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

# ======== 设备检测 ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======== 加载数据 ========
X = np.load("data/kryptonite-10-X.npy")
y = np.load("data/kryptonite-10-y.npy")

# ======== 模型结构（不改动） ========
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

# ======== 10 折 Stratified K-Fold CV ========
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1} ---")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    # 模型与优化器
    model = SimpleMLP(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # ======== 训练阶段 ========
    for epoch in range(70):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100 - Train Loss: {total_loss/len(train_loader.dataset):.4f}")

    # ======== 测试阶段 ========
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_t.to(device))
        y_hat = (y_prob >= 0.5).float()

        acc = accuracy_score(y_test_t, y_hat.cpu())
        f1 = f1_score(y_test_t, y_hat.cpu())
        try:
            auc = roc_auc_score(y_test_t, y_prob.cpu())
        except:
            auc = np.nan  # 若单类别导致AUC不可计算

        cm = confusion_matrix(y_test_t, y_hat.cpu())
        fold_results.append([acc, f1, auc])
        print(f"Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

# ======== 汇总统计 ========
fold_results = np.array(fold_results)
mean_acc, mean_f1, mean_auc = np.nanmean(fold_results, axis=0)
std_acc, std_f1, std_auc = np.nanstd(fold_results, axis=0)

print("\n===== Final 10-Fold Summary =====")
print(f"Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
print(f"F1-score : {mean_f1:.4f} ± {std_f1:.4f}")
print(f"AUC      : {mean_auc:.4f} ± {std_auc:.4f}")

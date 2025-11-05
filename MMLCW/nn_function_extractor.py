import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt

# ======== 设备检测 ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======== 加载数据 ========
X = np.load("data/kryptonite-10-X.npy")
y = np.load("data/kryptonite-10-y.npy")

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: {np.bincount(y.astype(int))}")
print(f"Feature statistics:")
print(f"  Mean: {X.mean(axis=0)}")
print(f"  Std:  {X.std(axis=0)}")
print(f"  Min:  {X.min(axis=0)}")
print(f"  Max:  {X.max(axis=0)}")

# ======== 模型结构 ========
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

# ======== 训练一个模型 ========
print("\n" + "="*60)
print("TRAINING NEURAL NETWORK")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

model = SimpleMLP(input_dim=X.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1}/100 - Train Loss: {total_loss/len(train_loader.dataset):.4f}")

model.eval()
with torch.no_grad():
    y_prob = model(X_test_t.to(device))
    y_pred = (y_prob >= 0.5).float()
    acc = accuracy_score(y_test, y_pred.cpu().numpy())
    f1 = f1_score(y_test, y_pred.cpu().numpy())
    print(f"\nNeural Network - Accuracy: {acc:.4f}, F1: {f1:.4f}")

# ======== 1. 权重分析 ========
print("\n" + "="*60)
print("ANALYZING NEURAL NETWORK WEIGHTS")
print("="*60)

first_layer_weights = model.net[0].weight.data.cpu().numpy()
print(f"\nFirst layer weight shape: {first_layer_weights.shape}")
print(f"Weight magnitude per input feature:")
feature_importance = np.abs(first_layer_weights).sum(axis=0)
for i, imp in enumerate(feature_importance):
    print(f"  Feature {i}: {imp:.4f}")

# ======== 2. 特征重要性分析 ========
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# 使用随机森林或逻辑回归
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"\nRandom Forest Feature Importance:")
for i, imp in enumerate(rf.feature_importances_):
    print(f"  Feature {i}: {imp:.4f}")

# ======== 3. 决策树近似 ========
print("\n" + "="*60)
print("DECISION TREE APPROXIMATION")
print("="*60)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
print(f"\nDecision Tree - Accuracy: {dt_acc:.4f}, F1: {dt_f1:.4f}")
print("\nDecision Tree Rules:")
print(export_text(dt, feature_names=[f"x{i}" for i in range(X.shape[1])]))

# ======== 4. 逻辑回归近似 ========
print("\n" + "="*60)
print("LOGISTIC REGRESSION APPROXIMATION")
print("="*60)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
print(f"\nLogistic Regression - Accuracy: {lr_acc:.4f}, F1: {lr_f1:.4f}")
print(f"Coefficients: {lr.coef_[0]}")
print(f"Intercept: {lr.intercept_[0]}")

# 构建公式
print("\nLogistic Regression Formula (on scaled data):")
formula = f"P(y=1) = sigmoid({lr.intercept_[0]:.4f}"
for i, coef in enumerate(lr.coef_[0]):
    formula += f" + {coef:.4f}*x{i}"
formula += ")"
print(formula)

# ======== 5. 简单数学关系探索 ========
print("\n" + "="*60)
print("EXPLORING SIMPLE MATHEMATICAL RELATIONSHIPS")
print("="*60)

# 尝试各种简单的数学组合
print("\nTesting simple functions:")

# 线性组合
linear_combinations = []
for i in range(X.shape[1]):
    for j in range(i+1, X.shape[1]):
        # Sum
        feat = X_train[:, i] + X_train[:, j]
        lr_temp = LogisticRegression(max_iter=1000)
        lr_temp.fit(feat.reshape(-1, 1), y_train)
        score = lr_temp.score(X_test[:, i].reshape(-1, 1) + X_test[:, j].reshape(-1, 1), y_test)
        linear_combinations.append((f"x{i} + x{j}", score))
        
        # Product
        feat = X_train[:, i] * X_train[:, j]
        lr_temp = LogisticRegression(max_iter=1000)
        lr_temp.fit(feat.reshape(-1, 1), y_train)
        score = lr_temp.score((X_test[:, i] * X_test[:, j]).reshape(-1, 1), y_test)
        linear_combinations.append((f"x{i} * x{j}", score))

# 排序并显示最佳
linear_combinations.sort(key=lambda x: x[1], reverse=True)
print("\nTop 10 simple feature combinations:")
for formula, score in linear_combinations[:10]:
    print(f"  {formula}: {score:.4f}")

# ======== 6. 多项式特征探索 ========
print("\n" + "="*60)
print("POLYNOMIAL FEATURE EXPLORATION")
print("="*60)

from sklearn.preprocessing import PolynomialFeatures

for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    lr_poly = LogisticRegression(max_iter=1000)
    lr_poly.fit(X_train_poly, y_train)
    poly_score = lr_poly.score(X_test_poly, y_test)
    
    print(f"\nPolynomial degree {degree}: Accuracy = {poly_score:.4f}")
    print(f"Number of features: {X_train_poly.shape[1]}")
    
    # 找到最重要的系数
    coef_importance = np.abs(lr_poly.coef_[0])
    top_indices = np.argsort(coef_importance)[-10:][::-1]
    print(f"Top 10 polynomial terms:")
    feature_names = poly.get_feature_names_out([f"x{i}" for i in range(X.shape[1])])
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {lr_poly.coef_[0][idx]:.4f}")

# ======== 7. 符号回归（如果可用） ========
print("\n" + "="*60)
print("SYMBOLIC REGRESSION (if gplearn available)")
print("="*60)

try:
    from gplearn.genetic import SymbolicClassifier
    
    gp = SymbolicClassifier(
        population_size=1000,
        generations=20,
        tournament_size=20,
        stopping_criteria=0.01,
        const_range=(-10, 10),
        init_depth=(2, 6),
        parsimony_coefficient=0.01,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    gp.fit(X_train, y_train)
    gp_pred = gp.predict(X_test)
    gp_acc = accuracy_score(y_test, gp_pred)
    gp_f1 = f1_score(y_test, gp_pred)
    
    print(f"\nSymbolic Regression - Accuracy: {gp_acc:.4f}, F1: {gp_f1:.4f}")
    print(f"Discovered Formula: {gp._program}")
    
except ImportError:
    print("\ngplearn not installed. Install with: pip install gplearn")
    print("Skipping symbolic regression...")

# ======== 8. 可视化决策边界（如果是2D） ========
if X.shape[1] == 2:
    print("\n" + "="*60)
    print("VISUALIZING DECISION BOUNDARIES")
    print("="*60)
    
    plt.figure(figsize=(15, 5))
    
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 神经网络
    plt.subplot(1, 3, 1)
    model.eval()
    with torch.no_grad():
        grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
        Z = model(torch.tensor(grid_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        Z = (Z >= 0.5).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.title('Neural Network')
    
    # 决策树
    plt.subplot(1, 3, 2)
    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.title('Decision Tree')
    
    # 逻辑回归
    plt.subplot(1, 3, 3)
    grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = lr.predict(grid_scaled).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.title('Logistic Regression')
    
    plt.tight_layout()
    plt.savefig('decision_boundaries.png', dpi=150, bbox_inches='tight')
    print("\nSaved decision boundary visualization to 'decision_boundaries.png'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("\nSummary:")
print(f"  Neural Network:      Acc={acc:.4f}, F1={f1:.4f}")
print(f"  Decision Tree:       Acc={dt_acc:.4f}, F1={dt_f1:.4f}")
print(f"  Logistic Regression: Acc={lr_acc:.4f}, F1={lr_f1:.4f}")
print("\nCheck the output above for potential formulas and relationships!")

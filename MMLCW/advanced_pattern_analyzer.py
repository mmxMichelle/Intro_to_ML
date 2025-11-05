import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations, product
import matplotlib.pyplot as plt
from collections import Counter

# ======== 设备检测 ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ======== 加载数据 ========
X = np.load("data/kryptonite-10-X.npy")
y = np.load("data/kryptonite-10-y.npy")

print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: {np.bincount(y.astype(int))}")
print(f"Perfectly balanced: {abs(np.bincount(y.astype(int))[0] - np.bincount(y.astype(int))[1]) < 100}")

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
    
    def get_hidden_activations(self, x):
        """Get activations from hidden layers"""
        h1 = self.net[0](x)  # First linear
        h1_relu = self.net[1](h1)  # ReLU
        h2 = self.net[3](h1_relu)  # Second linear
        h2_relu = self.net[4](h2)  # ReLU
        return h1, h1_relu, h2, h2_relu

# ======== 训练模型 ========
print("\n" + "="*70)
print("TRAINING NEURAL NETWORK FOR ANALYSIS")
print("="*70)

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
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    y_pred_train = (model(X_train_t.to(device)) >= 0.5).float().cpu().numpy()
    y_pred_test = (model(X_test_t.to(device)) >= 0.5).float().cpu().numpy()
    nn_train_acc = accuracy_score(y_train, y_pred_train)
    nn_test_acc = accuracy_score(y_test, y_pred_test)
    print(f"NN Train Accuracy: {nn_train_acc:.4f}")
    print(f"NN Test Accuracy:  {nn_test_acc:.4f}")

# ======== TEST 1: XOR/PARITY PATTERNS ========
print("\n" + "="*70)
print("TEST 1: XOR/PARITY PATTERN DETECTION")
print("="*70)

def test_xor_pattern(X, y, threshold=0.5):
    """Test if data follows XOR pattern"""
    best_acc = 0
    best_features = None
    best_k = None
    
    # Test different numbers of features in XOR
    for k in range(2, min(11, X.shape[1] + 1)):
        for feat_combo in combinations(range(X.shape[1]), k):
            # XOR: count how many features exceed threshold
            X_binary = (X[:, feat_combo] > threshold).astype(int)
            xor_result = np.sum(X_binary, axis=1) % 2  # Parity
            acc = accuracy_score(y, xor_result)
            
            if acc > best_acc:
                best_acc = acc
                best_features = feat_combo
                best_k = k
            
            # Also try inverse
            acc_inv = accuracy_score(y, 1 - xor_result)
            if acc_inv > best_acc:
                best_acc = acc_inv
                best_features = feat_combo
                best_k = k
    
    return best_acc, best_features, best_k

print("Testing XOR/Parity patterns (threshold=0.5)...")
xor_acc, xor_features, xor_k = test_xor_pattern(X_train, y_train, threshold=0.5)
print(f"Best XOR pattern: {xor_k} features")
print(f"Features: {xor_features}")
print(f"Train Accuracy: {xor_acc:.4f}")

if xor_features:
    X_binary = (X_test[:, xor_features] > 0.5).astype(int)
    xor_result = np.sum(X_binary, axis=1) % 2
    xor_test_acc = max(accuracy_score(y_test, xor_result), accuracy_score(y_test, 1 - xor_result))
    print(f"Test Accuracy: {xor_test_acc:.4f}")

# Try different thresholds
print("\nTrying different thresholds for XOR...")
for thresh in [0.0, 0.25, 0.5, 0.75, 1.0]:
    acc, feats, k = test_xor_pattern(X_train, y_train, threshold=thresh)
    print(f"  Threshold={thresh:.2f}: Acc={acc:.4f}, k={k}, features={feats}")

# ======== TEST 2: MAJORITY VOTE / COUNTING RULES ========
print("\n" + "="*70)
print("TEST 2: MAJORITY VOTE / COUNTING RULES")
print("="*70)

def test_counting_rules(X, y, threshold=0.5):
    """Test if data follows 'k out of n features exceed threshold' rule"""
    best_acc = 0
    best_rule = None
    
    X_binary = (X > threshold).astype(int)
    counts = np.sum(X_binary, axis=1)
    
    # Test each possible count threshold
    for count_thresh in range(X.shape[1] + 1):
        # >= threshold
        pred = (counts >= count_thresh).astype(int)
        acc = max(accuracy_score(y, pred), accuracy_score(y, 1 - pred))
        if acc > best_acc:
            best_acc = acc
            best_rule = f">= {count_thresh}"
        
        # == threshold
        pred = (counts == count_thresh).astype(int)
        acc = max(accuracy_score(y, pred), accuracy_score(y, 1 - pred))
        if acc > best_acc:
            best_acc = acc
            best_rule = f"== {count_thresh}"
    
    return best_acc, best_rule

print("Testing counting rules (threshold=0.5)...")
count_acc, count_rule = test_counting_rules(X_train, y_train, threshold=0.5)
print(f"Best counting rule: count {count_rule}")
print(f"Train Accuracy: {count_acc:.4f}")

X_test_binary = (X_test > 0.5).astype(int)
test_counts = np.sum(X_test_binary, axis=1)
for count_thresh in range(11):
    pred_ge = (test_counts >= count_thresh).astype(int)
    pred_eq = (test_counts == count_thresh).astype(int)
    acc_ge = max(accuracy_score(y_test, pred_ge), accuracy_score(y_test, 1 - pred_ge))
    acc_eq = max(accuracy_score(y_test, pred_eq), accuracy_score(y_test, 1 - pred_eq))
    print(f"  Count >={count_thresh}: {acc_ge:.4f} | Count =={count_thresh}: {acc_eq:.4f}")

# ======== TEST 3: WEIGHTED VOTING ========
print("\n" + "="*70)
print("TEST 3: WEIGHTED VOTING (BOOLEAN LINEAR COMBINATION)")
print("="*70)

def test_weighted_voting(X, y, threshold=0.5, num_trials=1000):
    """Test random weighted combinations of binary features"""
    best_acc = 0
    best_weights = None
    
    X_binary = (X > threshold).astype(int)
    
    for _ in range(num_trials):
        # Random weights from {-1, 0, 1}
        weights = np.random.choice([-1, 0, 1], size=X.shape[1])
        score = X_binary @ weights
        
        # Try different thresholds on the score
        for score_thresh in np.percentile(score, [10, 25, 50, 75, 90]):
            pred = (score >= score_thresh).astype(int)
            acc = max(accuracy_score(y, pred), accuracy_score(y, 1 - pred))
            if acc > best_acc:
                best_acc = acc
                best_weights = weights
    
    return best_acc, best_weights

print("Testing weighted voting (1000 random trials)...")
vote_acc, vote_weights = test_weighted_voting(X_train, y_train)
print(f"Best weighted voting accuracy: {vote_acc:.4f}")
print(f"Weights: {vote_weights}")

# ======== TEST 4: NEURAL NETWORK KNOWLEDGE DISTILLATION ========
print("\n" + "="*70)
print("TEST 4: KNOWLEDGE DISTILLATION (TRAIN SIMPLE MODEL ON NN PREDICTIONS)")
print("="*70)

# Get NN predictions on all training data
with torch.no_grad():
    nn_train_probs = model(X_train_t.to(device)).cpu().numpy().flatten()
    nn_train_preds = (nn_train_probs >= 0.5).astype(int)

# Train decision tree on NN predictions
print("\nTraining Decision Tree on NN predictions...")
for depth in [3, 5, 7, 10, 15, 20]:
    dt_distill = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_distill.fit(X_train, nn_train_preds)
    
    # How well does it match NN on train?
    dt_train_pred = dt_distill.predict(X_train)
    dt_test_pred = dt_distill.predict(X_test)
    
    match_train = accuracy_score(nn_train_preds, dt_train_pred)
    match_test = accuracy_score(y_pred_test, dt_test_pred)
    actual_test = accuracy_score(y_test, dt_test_pred)
    
    print(f"Depth={depth:2d}: Match NN train={match_train:.4f}, Match NN test={match_test:.4f}, Actual test acc={actual_test:.4f}")

# Best tree
dt_best = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_best.fit(X_train, nn_train_preds)

print("\nBest distilled tree rules (depth=10):")
from sklearn.tree import export_text
tree_rules = export_text(dt_best, feature_names=[f"x{i}" for i in range(X.shape[1])], max_depth=3)
print(tree_rules)

# ======== TEST 5: THRESHOLD SEARCH ========
print("\n" + "="*70)
print("TEST 5: OPTIMAL THRESHOLD SEARCH FOR EACH FEATURE")
print("="*70)

def find_best_threshold_per_feature(X, y):
    """Find best threshold for each feature individually"""
    results = []
    for i in range(X.shape[1]):
        best_acc = 0
        best_thresh = 0
        
        # Try percentile-based thresholds
        for percentile in range(0, 101, 5):
            thresh = np.percentile(X[:, i], percentile)
            pred = (X[:, i] > thresh).astype(int)
            acc = max(accuracy_score(y, pred), accuracy_score(y, 1 - pred))
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        results.append((i, best_thresh, best_acc))
    
    return results

print("Finding optimal thresholds per feature...")
threshold_results = find_best_threshold_per_feature(X_train, y_train)
threshold_results.sort(key=lambda x: x[2], reverse=True)

print("\nFeatures ranked by individual predictive power:")
for feat_idx, thresh, acc in threshold_results:
    print(f"  Feature {feat_idx}: threshold={thresh:.4f}, accuracy={acc:.4f}")

# ======== TEST 6: PAIRWISE FEATURE INTERACTIONS ========
print("\n" + "="*70)
print("TEST 6: PAIRWISE FEATURE INTERACTIONS (XOR, AND, OR)")
print("="*70)

def test_pairwise_logic(X, y, threshold=0.5, top_k=20):
    """Test all pairwise logical operations"""
    results = []
    X_binary = (X > threshold).astype(int)
    
    for i, j in combinations(range(X.shape[1]), 2):
        xi, xj = X_binary[:, i], X_binary[:, j]
        
        # XOR
        xor = xi ^ xj
        acc_xor = max(accuracy_score(y, xor), accuracy_score(y, 1 - xor))
        results.append((f"x{i} XOR x{j}", acc_xor))
        
        # AND
        and_op = xi & xj
        acc_and = max(accuracy_score(y, and_op), accuracy_score(y, 1 - and_op))
        results.append((f"x{i} AND x{j}", acc_and))
        
        # OR
        or_op = xi | xj
        acc_or = max(accuracy_score(y, or_op), accuracy_score(y, 1 - or_op))
        results.append((f"x{i} OR x{j}", acc_or))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

print("Testing pairwise logical operations...")
pairwise_results = test_pairwise_logic(X_train, y_train)
print("\nTop 20 pairwise operations:")
for op, acc in pairwise_results:
    print(f"  {op}: {acc:.4f}")

# ======== TEST 7: FEATURE CORRELATION WITH TARGET ========
print("\n" + "="*70)
print("TEST 7: FEATURE CORRELATIONS AND DISTRIBUTIONS")
print("="*70)

from scipy.stats import pearsonr, spearmanr

print("\nCorrelation with target:")
for i in range(X.shape[1]):
    pearson_corr, _ = pearsonr(X_train[:, i], y_train)
    spearman_corr, _ = spearmanr(X_train[:, i], y_train)
    print(f"  Feature {i}: Pearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f}")

# Check if features look uniform/binary
print("\nFeature distribution analysis:")
for i in range(X.shape[1]):
    unique_vals = len(np.unique(X[:, i]))
    print(f"  Feature {i}: {unique_vals} unique values, range=[{X[:, i].min():.3f}, {X[:, i].max():.3f}]")

# ======== TEST 8: ENSEMBLE OF BINARIZED FEATURES ========
print("\n" + "="*70)
print("TEST 8: EXHAUSTIVE SEARCH ON SUBSETS (3-4 features)")
print("="*70)

def exhaustive_subset_search(X, y, subset_size=3, threshold=0.5):
    """Try all possible Boolean functions on small subsets"""
    best_acc = 0
    best_combo = None
    best_func = None
    
    X_binary = (X > threshold).astype(int)
    
    count = 0
    for features in combinations(range(X.shape[1]), subset_size):
        X_subset = X_binary[:, features]
        
        # Try all possible Boolean functions (2^(2^n) of them)
        # For n=3, that's 256 functions; for n=4, that's 65536
        if subset_size <= 3:
            # Enumerate all possible truth tables
            for func_id in range(2 ** (2 ** subset_size)):
                # Create truth table
                truth_table = [(func_id >> i) & 1 for i in range(2 ** subset_size)]
                
                # Apply function
                indices = X_subset[:, 0]
                for k in range(1, subset_size):
                    indices = indices * 2 + X_subset[:, k]
                pred = np.array([truth_table[idx] for idx in indices])
                
                acc = accuracy_score(y, pred)
                if acc > best_acc:
                    best_acc = acc
                    best_combo = features
                    best_func = truth_table
        
        count += 1
        if count >= 100:  # Limit for performance
            break
    
    return best_acc, best_combo, best_func

print("Searching over 3-feature subsets (limited to 100 combos for speed)...")
subset_acc, subset_features, subset_func = exhaustive_subset_search(X_train, y_train, subset_size=3)
print(f"Best 3-feature Boolean function: {subset_acc:.4f}")
print(f"Features: {subset_features}")
print(f"Truth table: {subset_func}")

# ======== TEST 9: ACTIVATION PATTERN ANALYSIS ========
print("\n" + "="*70)
print("TEST 9: NEURAL NETWORK ACTIVATION PATTERN ANALYSIS")
print("="*70)

# Get hidden layer activations
with torch.no_grad():
    h1, h1_relu, h2, h2_relu = model.get_hidden_activations(X_train_t.to(device))
    h1_np = h1_relu.cpu().numpy()
    h2_np = h2_relu.cpu().numpy()

print(f"Hidden layer 1 shape: {h1_np.shape}")
print(f"Hidden layer 2 shape: {h2_np.shape}")

# Find most discriminative neurons
print("\nMost discriminative neurons in layer 1:")
for i in range(min(128, h1_np.shape[1])):
    corr, _ = pearsonr(h1_np[:, i], y_train)
    if abs(corr) > 0.3:  # Strong correlation
        print(f"  Neuron {i}: correlation={corr:.4f}")

# Check if activations are binary-like
h1_binary_like = np.mean((h1_np > 0.5).astype(float)) + np.mean((h1_np < 0.1).astype(float))
print(f"\nLayer 1 'binariness': {h1_binary_like:.4f} (2.0 = fully binary, 1.0 = uniform)")

# ======== TEST 10: FORMULA HYPOTHESIS TESTING ========
print("\n" + "="*70)
print("TEST 10: SPECIFIC FORMULA HYPOTHESIS TESTING")
print("="*70)

print("Testing specific formula patterns...")

# Based on weight analysis: features 0,1,3,6,7,8 are important
important_features = [0, 1, 3, 6, 7, 8]
print(f"\nImportant features (from weight analysis): {important_features}")

# Test: XOR of important features
X_imp_binary = (X_train[:, important_features] > 0.5).astype(int)
xor_imp = np.sum(X_imp_binary, axis=1) % 2
acc_imp = max(accuracy_score(y_train, xor_imp), accuracy_score(y_train, 1 - xor_imp))
print(f"XOR of important features: {acc_imp:.4f}")

# Test: Majority vote of important features
maj_imp = (np.sum(X_imp_binary, axis=1) >= len(important_features) // 2).astype(int)
acc_maj = max(accuracy_score(y_train, maj_imp), accuracy_score(y_train, 1 - maj_imp))
print(f"Majority vote of important features: {acc_maj:.4f}")

# Test: All important features must be > threshold
all_imp = np.all(X_imp_binary, axis=1).astype(int)
acc_all = max(accuracy_score(y_train, all_imp), accuracy_score(y_train, 1 - all_imp))
print(f"ALL important features > 0.5: {acc_all:.4f}")

# Test: ANY important feature > threshold
any_imp = np.any(X_imp_binary, axis=1).astype(int)
acc_any = max(accuracy_score(y_train, any_imp), accuracy_score(y_train, 1 - any_imp))
print(f"ANY important feature > 0.5: {acc_any:.4f}")

# ======== SUMMARY ========
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)
print(f"Neural Network:              {nn_test_acc:.4f}")
print(f"Best XOR pattern:            {xor_test_acc:.4f if xor_features else 0:.4f}")
print(f"Best counting rule:          {count_acc:.4f}")
print(f"Best weighted voting:        {vote_acc:.4f}")
print(f"Best distilled tree:         {actual_test:.4f}")
print(f"Best 3-feature Boolean:      {subset_acc:.4f}")
print(f"Best pairwise operation:     {pairwise_results[0][1]:.4f}")
print("\n" + "="*70)
print("CONCLUSION: Check the patterns above to identify the underlying function!")
print("="*70)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import pickle
import os
from pathlib import Path

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

class MLPTrainer:
    def __init__(self, data_path="data", model_save_path="saved_models"):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create save directory if it doesn't exist
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Using device: {self.device}")
    
    def load_data(self):
        """Load the training data"""
        self.X = np.load(f"{self.data_path}/kryptonite-10-X.npy")
        self.y = np.load(f"{self.data_path}/kryptonite-10-y.npy")
        print(f"Loaded data: X shape {self.X.shape}, y shape {self.y.shape}")
        return self.X, self.y
    
    def train_cv_models(self, n_epochs=70, save_all_folds=True, save_best_only=False):
        """Train models using 10-fold cross-validation"""
        if not hasattr(self, 'X'):
            self.load_data()
        
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_results = []
        trained_models = []
        scalers = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.X, self.y)):
            print(f"\n--- Training Fold {fold+1} ---")
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Data standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert to tensors
            X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
            
            # DataLoader
            train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
            
            # Model and optimizer
            model = SimpleMLP(input_dim=self.X.shape[1]).to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            
            # Training loop
            for epoch in range(n_epochs):
                model.train()
                total_loss = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * xb.size(0)
                
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {total_loss/len(train_loader.dataset):.4f}")
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                y_prob = model(X_test_t.to(self.device))
                y_hat = (y_prob >= 0.5).float()
                
                acc = accuracy_score(y_test_t, y_hat.cpu())
                f1 = f1_score(y_test_t, y_hat.cpu())
                try:
                    auc = roc_auc_score(y_test_t, y_prob.cpu())
                except:
                    auc = np.nan
                
                cm = confusion_matrix(y_test_t, y_hat.cpu())
                fold_results.append([acc, f1, auc])
                print(f"Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                print(f"Confusion Matrix:\n{cm}")
            
            # Save fold data
            fold_data = {
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'performance': {'accuracy': acc, 'f1': f1, 'auc': auc},
                'input_dim': self.X.shape[1]
            }
            
            if save_all_folds:
                torch.save(fold_data, f"{self.model_save_path}/fold_{fold+1}_model.pt")
            
            trained_models.append(fold_data)
            scalers.append(scaler)
        
        # Summary statistics
        fold_results = np.array(fold_results)
        mean_acc, mean_f1, mean_auc = np.nanmean(fold_results, axis=0)
        std_acc, std_f1, std_auc = np.nanstd(fold_results, axis=0)
        
        print("\n===== Final 10-Fold CV Summary =====")
        print(f"Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"F1-score : {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"AUC      : {mean_auc:.4f} ± {std_auc:.4f}")
        
        # Save summary and best model
        summary = {
            'cv_results': fold_results,
            'mean_performance': {'accuracy': mean_acc, 'f1': mean_f1, 'auc': mean_auc},
            'std_performance': {'accuracy': std_acc, 'f1': std_f1, 'auc': std_auc}
        }
        
        # Find best model based on AUC (or accuracy if AUC is NaN)
        if not np.isnan(mean_auc):
            best_fold_idx = np.nanargmax(fold_results[:, 2])  # AUC
        else:
            best_fold_idx = np.argmax(fold_results[:, 0])  # Accuracy
        
        best_model_data = trained_models[best_fold_idx]
        
        # Save best model and summary
        torch.save(best_model_data, f"{self.model_save_path}/best_model.pt")
        with open(f"{self.model_save_path}/training_summary.pkl", 'wb') as f:
            pickle.dump(summary, f)
        
        # Save original data for analysis
        np.save(f"{self.model_save_path}/X_original.npy", self.X)
        np.save(f"{self.model_save_path}/y_original.npy", self.y)
        
        print(f"\nModels saved to {self.model_save_path}/")
        print(f"Best model: Fold {best_fold_idx+1} with AUC={fold_results[best_fold_idx, 2]:.4f}")
        
        return trained_models, summary

if __name__ == "__main__":
    trainer = MLPTrainer()
    models, summary = trainer.train_cv_models()

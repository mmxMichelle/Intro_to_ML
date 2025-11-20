import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import matplotlib.pyplot as plt


class Regressor(nn.Module):
    """
    Neural Network Regressor for California Housing Price Prediction.
    """

    def __init__(
        self,
        x,
        nb_epoch=1000,
        batch_size=64,
        learning_rate=0.001,
        hidden_layers=[256, 128, 64, 32],
        dropout_rate=0.3,
        weight_decay=1e-5,
        patience=50,
        min_delta=1e-4,
        verbose=True,
    ):
        """
        Initialize the regressor with optimal hyperparameters.
        """
        super(Regressor, self).__init__()

        # Store hyperparameters
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.hidden_layers = hidden_layers
        self.verbose = verbose

        # Preprocessing components
        self.label_binarizer = None
        self.numerical_columns = None
        self.categorical_columns = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.fill_values = None
        self.input_size = None
        self.output_size = 1

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.verbose:
                print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            if self.verbose:
                print("Using CPU")

        # Preprocess to determine input dimensions
        X_preprocessed, _ = self._preprocessor(x, training=True, encode=True)
        self.input_size = X_preprocessed.shape[1]

        # Build the neural network
        self._build_network()

        # Move model to device
        self.to(self.device)

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            min_lr=1e-7,
        )

        # Training history
        self.train_losses = []
        self.val_losses = []

    def _build_network(self):
        """Build the neural network architecture."""
        layers = []
        prev_size = self.input_size

        # Input layer with batch normalization
        layers.append(nn.Linear(prev_size, self.hidden_layers[0]))
        layers.append(nn.BatchNorm1d(self.hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            layers.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
            layers.append(nn.BatchNorm1d(self.hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))

        # Output layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights using He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

    def _preprocessor(self, x, y=None, training=False, encode=False):
        """
        Preprocess input and output data.
        """
        # Ensure x is a DataFrame
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)

        # Make a copy to avoid modifying original
        x = x.copy()

        if training:
            # Identify column types
            self.numerical_columns = x.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = x.select_dtypes(include=["object"]).columns.tolist()

            # Remove target column if present
            if "median_house_value" in self.numerical_columns:
                self.numerical_columns.remove("median_house_value")

            # Store fill values (median for numerical, mode for categorical)
            self.fill_values = {}
            for col in self.numerical_columns:
                self.fill_values[col] = x[col].median()
            for col in self.categorical_columns:
                mode = x[col].mode()
                self.fill_values[col] = mode[0] if not mode.empty else "UNKNOWN"

        # Fill missing values
        for col in self.numerical_columns:
            x[col] = x[col].fillna(self.fill_values[col])
        for col in self.categorical_columns:
            x[col] = x[col].fillna(self.fill_values[col])

        # Handle categorical variables with one-hot encoding
        if self.categorical_columns:
            if training and encode:
                self.label_binarizer = LabelBinarizer()
                all_categories = pd.concat(
                    [x[col].astype(str) for col in self.categorical_columns]
                )
                self.label_binarizer.fit(all_categories)

            encoded_cats = []
            for col in self.categorical_columns:
                encoded = self.label_binarizer.transform(x[col].astype(str))
                if encoded.ndim == 1:
                    encoded = encoded.reshape(-1, 1)
                encoded_cats.append(encoded)

            if encoded_cats:
                categorical_array = np.hstack(encoded_cats)
            else:
                categorical_array = np.zeros((len(x), 0))
        else:
            categorical_array = np.zeros((len(x), 0))

        # Extract numerical features
        numerical_array = x[self.numerical_columns].values

        # Combine numerical and categorical features
        x_processed = np.hstack([numerical_array, categorical_array])

        # Normalize using min-max scaling
        if training:
            self.x_min = x_processed.min(axis=0)
            self.x_max = x_processed.max(axis=0)
            self.x_range = self.x_max - self.x_min
            self.x_range[self.x_range == 0] = 1.0

        x_processed = (x_processed - self.x_min) / self.x_range

        # Handle y if provided
        y_processed = None
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y.values.flatten()
            elif isinstance(y, pd.Series):
                y = y.values

            y_processed = y.reshape(-1, 1)

            if training:
                self.y_min = y_processed.min()
                self.y_max = y_processed.max()
                self.y_range = self.y_max - self.y_min
                if self.y_range == 0:
                    self.y_range = 1.0

            # Normalize target
            y_processed = (y_processed - self.y_min) / self.y_range

        return x_processed, y_processed

    def fit(self, x, y):
        """
        Train the regressor on the provided data.
        """
        # Split into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.15, random_state=42
        )

        # Preprocess data
        x_train, y_train = self._preprocessor(x_train, y_train, training=True)
        x_val, y_val = self._preprocessor(x_val, y_val, training=False)

        # Convert to PyTorch tensors
        x_train_tensor = torch.FloatTensor(x_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        x_val_tensor = torch.FloatTensor(x_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        # Initialize best_state by default
        self.best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}

        for epoch in range(self.nb_epoch):
            # Training phase
            self.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                outputs = self.forward(batch_x)
                loss = self.criterion(outputs, batch_y)

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * batch_x.size(0)

            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)

            # Validation phase
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(x_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                self.val_losses.append(val_loss)

            # LR scheduler
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                self.load_state_dict(
                    {k: v.to(self.device) for k, v in self.best_state.items()}
                )
                break

            # Progress print（可关）
            if self.verbose and (epoch + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.nb_epoch}], "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

    def predict(self, x):
        """
        Predict house values for the given input.
        """
        x_processed, _ = self._preprocessor(x, training=False)
        x_tensor = torch.FloatTensor(x_processed).to(self.device)

        self.eval()
        with torch.no_grad():
            predictions = self.forward(x_tensor)

        predictions = predictions.cpu().numpy()
        predictions = predictions * self.y_range + self.y_min
        return predictions.flatten()

    def score(self, x, y, print_metrics=True):
        """
        Evaluate the model performance on given data.
        """
        predictions = self.predict(x)

        if isinstance(y, pd.DataFrame):
            y = y.values.flatten()
        elif isinstance(y, pd.Series):
            y = y.values

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

        if print_metrics:
            print("\n" + "=" * 50)
            print("Model Performance Metrics:")
            print("=" * 50)
            for metric, value in metrics.items():
                print(f"{metric:10s}: {value:.4f}")
            print("=" * 50 + "\n")

        return rmse, metrics  # 返回 RMSE 和完整指标字典


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    with open("part2_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    with open("part2_model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def perform_hyperparameter_search(
    x_train, y_train, save_csv_path="hyperparameter_results.csv"
):
    """
    Hyperparameter search over hidden_layers and learning_rate
    (5-fold CV), saving results to CSV.
    """
    param_grid = {
        "hidden_layers": [
            [128, 64, 32],
            [256, 128, 64, 32],
            [512, 256, 128, 64],
        ],
        "learning_rate": [0.0005, 0.001, 0.002],
    }

    best_score = float("inf")
    best_params = None
    results = []

    print("Starting hyperparameter search...")
    print(
        f"Total combinations: {len(param_grid['hidden_layers']) * len(param_grid['learning_rate'])}"
    )

    for hidden_layers in param_grid["hidden_layers"]:
        for lr in param_grid["learning_rate"]:
            print(f"\nTesting architecture {hidden_layers} | learning_rate={lr}")

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            fold_times = []

            for train_idx, val_idx in kf.split(x_train):
                if isinstance(x_train, pd.DataFrame):
                    x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                else:
                    x_tr, x_val = x_train[train_idx], x_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]

                start_time = time.time()

                model = Regressor(
                    x_tr,
                    nb_epoch=500,
                    hidden_layers=hidden_layers,
                    learning_rate=lr,
                    batch_size=64,
                    dropout_rate=0.3,
                    weight_decay=1e-5,
                    patience=30,
                    verbose=False,  # 交叉验证阶段静默
                )
                model.fit(x_tr, y_tr)

                elapsed = time.time() - start_time
                fold_times.append(elapsed)

                # Evaluate RMSE (不打印指标，直接取返回值)
                rmse, _ = model.score(x_val, y_val, print_metrics=False)
                cv_scores.append(rmse)

            avg_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            avg_time = np.mean(fold_times)

            print(f"Average RMSE: {avg_score:.4f} (+/- {std_score:.4f})")
            print(f"Average training time: {avg_time:.2f} sec")

            results.append(
                {
                    "hidden_layers": str(hidden_layers),
                    "learning_rate": lr,
                    "avg_rmse": avg_score,
                    "std_rmse": std_score,
                    "avg_time_sec": avg_time,
                }
            )

            if avg_score < best_score:
                best_score = avg_score
                best_params = {"hidden_layers": hidden_layers, "learning_rate": lr}

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_csv_path, index=False)
    print(f"\nResults saved to: {save_csv_path}")

    print("\n" + "=" * 60)
    print("Hyperparameter Search Summary")
    print("=" * 60)
    print(f"Best RMSE: {best_score:.4f}")
    print(f"Best Params: {best_params}")
    print("=" * 60)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": results_df,
    }


def example_main():
    """
    Example usage of the Regressor class.
    """
    # Load data
    data = pd.read_csv("src/housing.csv")

    # Separate features and target
    target_column = "median_house_value"
    x = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print("Training regressor...")

    # Hyperparameter search
    best_params = perform_hyperparameter_search(x_train, y_train)["best_params"]
    hidden_layers = best_params["hidden_layers"]
    lr = best_params["learning_rate"]

    # Train best model
    regressor = Regressor(
        x_train,
        nb_epoch=500,
        batch_size=64,
        learning_rate=lr,
        hidden_layers=hidden_layers,
        dropout_rate=0.3,
        patience=30,
        verbose=True,  # 最终模型可以适当打印
    )

    regressor.fit(x_train, y_train)

    # Evaluate on test set
    print("\nTest Set Performance:")
    _, test_metrics = regressor.score(x_test, y_test, print_metrics=True)

    # Save model
    save_regressor(regressor)

    # 返回所有在画图和报告中需要的数据
    return regressor, x_test, y_test, test_metrics


if __name__ == "__main__":
    # 训练与测试
    regressor, x_test, y_test, test_metrics = example_main()

    # =============== 图 1：Training vs Validation MSE over Epochs ===============
    train_losses = regressor.train_losses
    val_losses = regressor.val_losses
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, marker="o", label="Training MSE")
    plt.plot(epochs, val_losses, marker="s", label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation MSE over Epochs")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=300)
    plt.show()

    # =============== 图 2：Predicted vs Ground Truth (Test Set) ===============
    predictions = regressor.predict(x_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.5, s=10)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", linewidth=2)

    plt.xlabel("Ground Truth House Value")
    plt.ylabel("Predicted House Value")
    plt.title("Predicted vs Ground Truth (Test Set)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("pred_vs_truth.png", dpi=300)
    plt.show()

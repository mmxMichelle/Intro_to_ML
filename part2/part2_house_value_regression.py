import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import time


class Regressor(nn.Module):
    """
    Neural Network Regressor for California Housing Price Prediction
    
    Architecture designed for optimal performance with:
    - Batch normalization for stable training
    - Dropout for regularization
    - Carefully tuned hyperparameters
    """
    
    def __init__(self, x, nb_epoch=1000, batch_size=64, learning_rate=0.001,
                 hidden_layers=[256, 128, 64, 32], dropout_rate=0.3,
                 weight_decay=1e-5, patience=50, min_delta=1e-4):
        """
        Initialize the regressor with optimal hyperparameters.
        
        Args:
            x: Training input features (pandas DataFrame)
            nb_epoch: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            learning_rate: Learning rate for optimizer
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability for regularization
            weight_decay: L2 regularization parameter
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
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
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
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
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            min_lr=1e-7
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
            layers.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
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
        
        Args:
            x: Input features (pandas DataFrame)
            y: Target values (pandas DataFrame or Series), optional
            training: Boolean indicating if this is training data
            
        Returns:
            Tuple of (preprocessed_x, preprocessed_y)
        """
        # Ensure x is a DataFrame
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        
        # Make a copy to avoid modifying original
        x = x.copy()
        
        if training:
            # Identify column types
            self.numerical_columns = x.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_columns = x.select_dtypes(include=['object']).columns.tolist()
            
            # Remove target column if present
            if 'median_house_value' in self.numerical_columns:
                self.numerical_columns.remove('median_house_value')
            
            # Store fill values (median for numerical, mode for categorical)
            self.fill_values = {}
            for col in self.numerical_columns:
                self.fill_values[col] = x[col].median()
            for col in self.categorical_columns:
                self.fill_values[col] = x[col].mode()[0] if not x[col].mode().empty else 'UNKNOWN'



        # else:
        #     # Ensure all training-time columns exist in x
        #     for col in self.numerical_columns:
        #         if col not in x.columns:
        #             x[col] = np.nan
        #
        #     for col in self.categorical_columns:
        #         if col not in x.columns:
        #             x[col] = 'UNKNOWN'
        
        # Fill missing values
        for col in self.numerical_columns:
            x[col] = x[col].fillna(self.fill_values[col])
        for col in self.categorical_columns:
            x[col] = x[col].fillna(self.fill_values[col])

        # Handle categorical variables with one-hot encoding
        if self.categorical_columns:
            if training:
                # ------ FIT encoder only if encode=True ------
                if encode:
                    self.label_binarizer = LabelBinarizer()
                    all_categories = pd.concat(
                        [x[col].astype(str) for col in self.categorical_columns]
                    )
                    self.label_binarizer.fit(all_categories)
                # else:
                #   do NOT retrain encoder, use the already trained one
            
            # Transform each categorical column
            encoded_cats = []
            for col in self.categorical_columns:
                encoded = self.label_binarizer.transform(x[col].astype(str))
                # Handle single class case
                if encoded.ndim == 1:
                    encoded = encoded.reshape(-1, 1)
                encoded_cats.append(encoded)
            
            if encoded_cats:
                categorical_array = np.hstack(encoded_cats)
            else:
                categorical_array = np.array([]).reshape(len(x), 0)
        else:
            categorical_array = np.array([]).reshape(len(x), 0)
        
        # Extract numerical features
        numerical_array = x[self.numerical_columns].values
        
        # Combine numerical and categorical features
        x_processed = np.hstack([numerical_array, categorical_array])
        
        # Normalize numerical features using min-max scaling
        if training:
            self.x_min = x_processed.min(axis=0)
            self.x_max = x_processed.max(axis=0)
            # Avoid division by zero
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
        
        Args:
            x: Training features (pandas DataFrame)
            y: Training targets (pandas DataFrame or Series)
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
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Initialize best_state by default
        self.best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
        
        for epoch in range(self.nb_epoch):
            # Training phase
            self.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                # Forward pass
                outputs = self.forward(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
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
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                self.load_state_dict({k: v.to(self.device) for k, v in self.best_state.items()})
                break
            
            # Print progress
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{self.nb_epoch}], "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    def predict(self, x):
        """
        Predict house values for the given input.
        
        Args:
            x: Input features (pandas DataFrame or numpy array)
            
        Returns:
            Predicted house values (numpy array)
        """
        # Preprocess input
        x_processed, _ = self._preprocessor(x, training=False)
        
        # Convert to tensor
        x_tensor = torch.FloatTensor(x_processed).to(self.device)
        
        # Make predictions
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x_tensor)
        
        predictions = predictions.cpu().numpy()

        # Denormalize predictions
        predictions = predictions * self.y_range + self.y_min
        
        return predictions.flatten()
    
    def score(self, x, y):
        """
        Evaluate the model performance.
        
        Args:
            x: Input features
            y: True target values
            
        Returns:
            Dictionary of performance metrics
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
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print("\n" + "="*50)
        print("Model Performance Metrics:")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:10s}: {value:.4f}")
        print("="*50 + "\n")
        
        return rmse  # Return RMSE as primary metric


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def perform_hyperparameter_search(x_train, y_train, save_csv_path="hyperparameter_results.csv"):
    """
    Hyperparameter search over hidden_layers and learning_rate,
    including average training time + RMSE per combination, and save results to CSV.
    """

    param_grid = {
        'hidden_layers': [
            [128, 64, 32],
            [256, 128, 64, 32],
            [512, 256, 128, 64],
        ],
        'learning_rate': [0.0005, 0.001, 0.002],
    }

    best_score = float('inf')
    best_params = None
    results = []

    print("Starting hyperparameter search...")
    print(f"Total combinations: {len(param_grid['hidden_layers']) * len(param_grid['learning_rate'])}")

    # loop for every hidden_layer and learning rate combination
    for hidden_layers in param_grid['hidden_layers']:
        for lr in param_grid['learning_rate']:

            print(f"\nTesting architecture {hidden_layers} | learning_rate={lr}")

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            fold_times = []

            for train_idx, val_idx in kf.split(x_train):

                # Select subset
                if isinstance(x_train, pd.DataFrame):
                    x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                else:
                    x_tr, x_val = x_train[train_idx], x_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Start timing
                start_time = time.time()

                # Build and train model
                model = Regressor(
                    x_tr,
                    nb_epoch=500,
                    hidden_layers=hidden_layers,
                    learning_rate=lr,
                    batch_size=64,      # fixed
                    dropout_rate=0.3,   # fixed
                    weight_decay=1e-5,  # fixed
                    patience=30         # fixed
                )
                model.fit(x_tr, y_tr)

                elapsed = time.time() - start_time
                fold_times.append(elapsed)

                # Evaluate RMSE
                score_rmse = model.score(x_val, y_val)
                cv_scores.append(score_rmse)

            avg_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            avg_time = np.mean(fold_times)

            print(f"Average RMSE: {avg_score:.4f} (+/- {std_score:.4f})")
            print(f"Average training time: {avg_time:.2f} sec")

            results.append({
                'hidden_layers': str(hidden_layers),
                'learning_rate': lr,
                'avg_rmse': avg_score,
                'std_rmse': std_score,
                'avg_time_sec': avg_time
            })

            if avg_score < best_score:
                best_score = avg_score
                best_params = {
                    'hidden_layers': hidden_layers,
                    'learning_rate': lr
                }

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_csv_path, index=False)
    print(f"\nResults saved to: {save_csv_path}")

    print("\n" + "="*60)
    print("Hyperparameter Search Summary")
    print("="*60)
    print(f"Best RMSE: {best_score:.4f}")
    print(f"Best Params: {best_params}")
    print("="*60)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results_df
    }


def example_main():
    """
    Example usage of the Regressor class.
    """
    # Load data
    data = pd.read_csv("src/housing.csv")
    
    # Separate features and target
    target_column = 'median_house_value'
    x = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    print("Training regressor...")

    # Perform hyperparameter search
    best_params = perform_hyperparameter_search(x_train, y_train)['best_params']
    hidden_layers = best_params['hidden_layers']
    lr = best_params['learning_rate']

    # Create and train best hyperparams regressor
    regressor = Regressor(
        x_train,
        nb_epoch=500,
        batch_size=64,
        learning_rate=lr,
        hidden_layers=hidden_layers,
        dropout_rate=0.3,
        patience=30
    )

    regressor.fit(x_train, y_train)
    
    # Evaluate on test set
    # (optional: load best model)
    # regressor = load_regressor()
    print("\nTest Set Performance:")
    regressor.score(x_test, y_test)
    
    # Save the model
    save_regressor(regressor)
    
    return regressor


if __name__ == "__main__":
    example_main()

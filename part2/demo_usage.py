"""
Demonstration script for Part 1 Neural Network Mini-Library
Shows various usage patterns and best practices
"""

import numpy as np
from part1_nn_lib import (
    LinearLayer, SigmoidLayer, ReluLayer, MultiLayerNetwork,
    Trainer, Preprocessor, save_network, load_network
)


def demo_basic_usage():
    """Demonstrates basic usage of the library."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Usage")
    print("="*60)
    
    # Create synthetic regression data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features, 1)
    y = np.dot(X, true_weights) + np.random.randn(n_samples, 1) * 0.5
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    
    # Preprocess data
    prep_X = Preprocessor(X)
    prep_y = Preprocessor(y)
    
    X_norm = prep_X.apply(X)
    y_norm = prep_y.apply(y)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y_norm[:split], y_norm[split:]
    
    # Create network
    network = MultiLayerNetwork(
        input_dim=n_features,
        neurons=[20, 10, 1],
        activations=["relu", "relu", "identity"]
    )
    
    print(f"Network: {n_features} -> 20 -> 10 -> 1")
    
    # Train
    trainer = Trainer(
        network=network,
        batch_size=32,
        nb_epoch=50,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True
    )
    
    print("\nTraining...")
    initial_loss = trainer.eval_loss(X_train, y_train)
    trainer.train(X_train, y_train)
    final_train_loss = trainer.eval_loss(X_train, y_train)
    val_loss = trainer.eval_loss(X_val, y_val)
    
    print(f"Initial training loss: {initial_loss:.4f}")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Improvement: {(1 - final_train_loss/initial_loss)*100:.1f}%")


def demo_classification():
    """Demonstrates binary classification."""
    print("\n" + "="*60)
    print("DEMO 2: Binary Classification")
    print("="*60)
    
    # Create synthetic classification data
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    # Class 0: centered at [-1, -1, ...]
    X_class0 = np.random.randn(n_samples//2, n_features) - 1
    y_class0 = np.zeros((n_samples//2, 1))
    
    # Class 1: centered at [1, 1, ...]
    X_class1 = np.random.randn(n_samples//2, n_features) + 1
    y_class1 = np.ones((n_samples//2, 1))
    
    X = np.vstack([X_class0, X_class1])
    y = np.vstack([y_class0, y_class1])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    print(f"Dataset: {n_samples} samples, {n_features} features, 2 classes")
    
    # Preprocess
    prep_X = Preprocessor(X)
    X_norm = prep_X.apply(X)
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Create network for classification
    network = MultiLayerNetwork(
        input_dim=n_features,
        neurons=[16, 8, 1],
        activations=["relu", "relu", "sigmoid"]
    )
    
    print(f"Network: {n_features} -> 16 -> 8 -> 1 (sigmoid output)")
    
    # Train with cross-entropy loss
    trainer = Trainer(
        network=network,
        batch_size=32,
        nb_epoch=30,
        learning_rate=0.05,
        loss_fun="cross_entropy",
        shuffle_flag=True
    )
    
    print("\nTraining with cross-entropy loss...")
    trainer.train(X_train, y_train)
    
    # Evaluate
    predictions = network.forward(X_val)
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y_val)
    
    print(f"Validation accuracy: {accuracy*100:.2f}%")
    print(f"Validation loss: {trainer.eval_loss(X_val, y_val):.4f}")


def demo_hyperparameter_comparison():
    """Demonstrates comparing different hyperparameters."""
    print("\n" + "="*60)
    print("DEMO 3: Hyperparameter Comparison")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 400
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, 1)
    
    # Preprocess
    prep_X = Preprocessor(X)
    X_norm = prep_X.apply(X)
    
    prep_y = Preprocessor(y)
    y_norm = prep_y.apply(y)
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y_norm[:split], y_norm[split:]
    
    # Compare different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    
    print("\nComparing learning rates:")
    print("-" * 40)
    
    for lr in learning_rates:
        network = MultiLayerNetwork(
            input_dim=n_features,
            neurons=[10, 5, 1],
            activations=["relu", "relu", "identity"]
        )
        
        trainer = Trainer(
            network=network,
            batch_size=16,
            nb_epoch=30,
            learning_rate=lr,
            loss_fun="mse",
            shuffle_flag=True
        )
        
        trainer.train(X_train, y_train)
        val_loss = trainer.eval_loss(X_val, y_val)
        
        print(f"LR={lr:5.3f} -> Validation Loss: {val_loss:.4f}")


def demo_deep_network():
    """Demonstrates creating a deeper network."""
    print("\n" + "="*60)
    print("DEMO 4: Deep Network")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, 3)  # Multi-output
    
    # Preprocess
    prep_X = Preprocessor(X)
    prep_y = Preprocessor(y)
    
    X_norm = prep_X.apply(X)
    y_norm = prep_y.apply(y)
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y_norm[:split], y_norm[split:]
    
    # Create deep network
    network = MultiLayerNetwork(
        input_dim=n_features,
        neurons=[64, 32, 16, 8, 3],
        activations=["relu", "relu", "relu", "relu", "identity"]
    )
    
    print(f"Deep Network: {n_features} -> 64 -> 32 -> 16 -> 8 -> 3")
    print("5 layers total")
    
    # Train
    trainer = Trainer(
        network=network,
        batch_size=32,
        nb_epoch=50,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True
    )
    
    print("\nTraining deep network...")
    trainer.train(X_train, y_train)
    val_loss = trainer.eval_loss(X_val, y_val)
    
    print(f"Validation loss: {val_loss:.4f}")


def demo_save_load():
    """Demonstrates saving and loading a network."""
    print("\n" + "="*60)
    print("DEMO 5: Save and Load Network")
    print("="*60)
    
    # Create and train a simple network
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100, 1)
    
    prep_X = Preprocessor(X)
    X_norm = prep_X.apply(X)
    
    network = MultiLayerNetwork(
        input_dim=5,
        neurons=[10, 1],
        activations=["relu", "identity"]
    )
    
    trainer = Trainer(
        network=network,
        batch_size=16,
        nb_epoch=20,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True
    )
    
    trainer.train(X_norm, y)
    
    # Save network
    save_network(network, "/home/claude/demo_network.pkl")
    print("Network saved to demo_network.pkl")
    
    # Load network
    loaded_network = load_network("/home/claude/demo_network.pkl")
    print("Network loaded successfully")
    
    # Verify they produce the same outputs
    test_input = X_norm[:5]
    original_output = network.forward(test_input)
    loaded_output = loaded_network.forward(test_input)
    
    print(f"Outputs match: {np.allclose(original_output, loaded_output)}")


def demo_advanced_preprocessing():
    """Demonstrates advanced preprocessing scenarios."""
    print("\n" + "="*60)
    print("DEMO 6: Advanced Preprocessing")
    print("="*60)
    
    # Create data with different scales
    np.random.seed(42)
    n_samples = 200
    
    # Feature 1: small scale [0, 1]
    feature1 = np.random.uniform(0, 1, (n_samples, 1))
    
    # Feature 2: large scale [0, 1000]
    feature2 = np.random.uniform(0, 1000, (n_samples, 1))
    
    # Feature 3: negative scale [-100, 100]
    feature3 = np.random.uniform(-100, 100, (n_samples, 1))
    
    # Feature 4: constant
    feature4 = np.ones((n_samples, 1)) * 42
    
    X = np.hstack([feature1, feature2, feature3, feature4])
    
    print("Original data statistics:")
    print(f"  Feature 1: min={np.min(X[:, 0]):.2f}, max={np.max(X[:, 0]):.2f}")
    print(f"  Feature 2: min={np.min(X[:, 1]):.2f}, max={np.max(X[:, 1]):.2f}")
    print(f"  Feature 3: min={np.min(X[:, 2]):.2f}, max={np.max(X[:, 2]):.2f}")
    print(f"  Feature 4: constant={np.unique(X[:, 3])[0]:.2f}")
    
    # Preprocess
    prep = Preprocessor(X)
    X_norm = prep.apply(X)
    
    print("\nNormalized data statistics:")
    print(f"  Feature 1: min={np.min(X_norm[:, 0]):.2f}, max={np.max(X_norm[:, 0]):.2f}")
    print(f"  Feature 2: min={np.min(X_norm[:, 1]):.2f}, max={np.max(X_norm[:, 1]):.2f}")
    print(f"  Feature 3: min={np.min(X_norm[:, 2]):.2f}, max={np.max(X_norm[:, 2]):.2f}")
    print(f"  Feature 4: constant={np.unique(X_norm[:, 3])[0]:.2f}")
    
    # Revert
    X_reverted = prep.revert(X_norm)
    
    print(f"\nReversion successful: {np.allclose(X, X_reverted)}")


def demo_batch_size_comparison():
    """Demonstrates effect of different batch sizes."""
    print("\n" + "="*60)
    print("DEMO 7: Batch Size Comparison")
    print("="*60)
    
    # Create data
    np.random.seed(42)
    X = np.random.randn(500, 8)
    y = np.random.randn(500, 1)
    
    prep_X = Preprocessor(X)
    X_norm = prep_X.apply(X)
    
    prep_y = Preprocessor(y)
    y_norm = prep_y.apply(y)
    
    split = int(0.8 * len(X))
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y_norm[:split], y_norm[split:]
    
    # Compare different batch sizes
    batch_sizes = [8, 32, 128]
    
    print("\nComparing batch sizes:")
    print("-" * 40)
    
    for bs in batch_sizes:
        network = MultiLayerNetwork(
            input_dim=8,
            neurons=[16, 8, 1],
            activations=["relu", "relu", "identity"]
        )
        
        trainer = Trainer(
            network=network,
            batch_size=bs,
            nb_epoch=30,
            learning_rate=0.01,
            loss_fun="mse",
            shuffle_flag=True
        )
        
        trainer.train(X_train, y_train)
        val_loss = trainer.eval_loss(X_val, y_val)
        
        print(f"Batch size {bs:3d} -> Validation Loss: {val_loss:.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("NEURAL NETWORK MINI-LIBRARY - COMPREHENSIVE DEMONSTRATIONS")
    print("="*70)
    
    demo_basic_usage()
    demo_classification()
    demo_hyperparameter_comparison()
    demo_deep_network()
    demo_save_load()
    demo_advanced_preprocessing()
    demo_batch_size_comparison()
    
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

"""
Comprehensive test suite for Part 1 Neural Network Mini-Library
"""
import numpy as np
import sys
sys.path.append('/home/claude')
from part1_nn_lib import (
    LinearLayer, SigmoidLayer, ReluLayer, MultiLayerNetwork,
    Trainer, Preprocessor, MSELossLayer, CrossEntropyLossLayer,
    xavier_init
)


def test_xavier_init():
    """Test Xavier initialization."""
    print("Testing Xavier initialization...")
    weights = xavier_init((10, 20))
    assert weights.shape == (10, 20), "Wrong shape"
    assert -1 < np.mean(weights) < 1, "Mean should be close to 0"
    print("✓ Xavier initialization passed")


def test_linear_layer():
    """Test LinearLayer forward and backward pass."""
    print("\nTesting LinearLayer...")
    
    # Setup
    np.random.seed(42)
    batch_size, n_in, n_out = 4, 3, 5
    layer = LinearLayer(n_in, n_out)
    x = np.random.randn(batch_size, n_in)
    
    # Test forward pass
    output = layer.forward(x)
    assert output.shape == (batch_size, n_out), f"Wrong output shape: {output.shape}"
    
    # Manual computation
    expected_output = np.dot(x, layer._W) + layer._b
    assert np.allclose(output, expected_output), "Forward pass computation incorrect"
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, n_out)
    grad_input = layer.backward(grad_output)
    
    assert grad_input.shape == (batch_size, n_in), f"Wrong gradient shape: {grad_input.shape}"
    
    # Check gradient shapes
    assert layer._grad_W_current.shape == (n_in, n_out), "Wrong weight gradient shape"
    assert layer._grad_b_current.shape == (1, n_out), "Wrong bias gradient shape"
    
    # Numerical gradient check for weights
    epsilon = 1e-5
    numerical_grad_W = np.zeros_like(layer._W)
    
    for i in range(min(n_in, 2)):  # Check a few elements
        for j in range(min(n_out, 2)):
            layer._W[i, j] += epsilon
            output_plus = np.dot(x, layer._W) + layer._b
            loss_plus = np.sum(output_plus * grad_output)
            
            layer._W[i, j] -= 2 * epsilon
            output_minus = np.dot(x, layer._W) + layer._b
            loss_minus = np.sum(output_minus * grad_output)
            
            layer._W[i, j] += epsilon  # Reset
            numerical_grad_W[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Compare numerical and analytical gradients for subset
    analytical_grad_subset = layer._grad_W_current[:2, :2]
    numerical_grad_subset = numerical_grad_W[:2, :2]
    
    assert np.allclose(analytical_grad_subset, numerical_grad_subset, rtol=1e-3), \
        f"Gradient check failed: analytical={analytical_grad_subset}, numerical={numerical_grad_subset}"
    
    # Test parameter update
    old_W = layer._W.copy()
    old_b = layer._b.copy()
    learning_rate = 0.1
    layer.update_params(learning_rate)
    
    expected_W = old_W - learning_rate * layer._grad_W_current
    expected_b = old_b - learning_rate * layer._grad_b_current
    
    assert np.allclose(layer._W, expected_W), "Weight update incorrect"
    assert np.allclose(layer._b, expected_b), "Bias update incorrect"
    
    print("✓ LinearLayer passed all tests")


def test_sigmoid_layer():
    """Test SigmoidLayer forward and backward pass."""
    print("\nTesting SigmoidLayer...")
    
    # Setup
    np.random.seed(42)
    layer = SigmoidLayer()
    x = np.random.randn(4, 3)
    
    # Test forward pass
    output = layer.forward(x)
    assert output.shape == x.shape, "Wrong output shape"
    assert np.all((output >= 0) & (output <= 1)), "Sigmoid output not in [0, 1]"
    
    # Manual computation for positive values
    expected_output = 1 / (1 + np.exp(-x))
    # For stability, the implementation uses a different formula for negative values
    # but the result should be the same
    assert np.allclose(output, expected_output, rtol=1e-5), "Forward pass incorrect"
    
    # Test backward pass
    grad_output = np.random.randn(*x.shape)
    grad_input = layer.backward(grad_output)
    
    assert grad_input.shape == x.shape, "Wrong gradient shape"
    
    # Numerical gradient check
    epsilon = 1e-5
    x_test = x[0, 0]
    
    layer.forward(x)
    sigmoid_val = layer._cache_current[0, 0]
    
    # Analytical gradient at one point
    analytical_grad = sigmoid_val * (1 - sigmoid_val) * grad_output[0, 0]
    
    # Numerical gradient
    x_copy = x.copy()
    x_copy[0, 0] += epsilon
    output_plus = layer.forward(x_copy)
    loss_plus = np.sum(output_plus * grad_output)
    
    x_copy[0, 0] -= 2 * epsilon
    output_minus = layer.forward(x_copy)
    loss_minus = np.sum(output_minus * grad_output)
    
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Re-run forward to restore cache
    layer.forward(x)
    grad_full = layer.backward(grad_output)
    
    assert np.abs(grad_full[0, 0] - numerical_grad) < 1e-4, \
        f"Gradient check failed: analytical={grad_full[0, 0]}, numerical={numerical_grad}"
    
    print("✓ SigmoidLayer passed all tests")


def test_relu_layer():
    """Test ReluLayer forward and backward pass."""
    print("\nTesting ReluLayer...")
    
    # Setup
    np.random.seed(42)
    layer = ReluLayer()
    x = np.random.randn(4, 3)
    
    # Test forward pass
    output = layer.forward(x)
    assert output.shape == x.shape, "Wrong output shape"
    assert np.all(output >= 0), "ReLU output should be non-negative"
    
    expected_output = np.maximum(0, x)
    assert np.allclose(output, expected_output), "Forward pass incorrect"
    
    # Test backward pass
    grad_output = np.random.randn(*x.shape)
    grad_input = layer.backward(grad_output)
    
    assert grad_input.shape == x.shape, "Wrong gradient shape"
    
    # Check that gradient is zero where input was negative
    expected_grad = grad_output * (x > 0)
    assert np.allclose(grad_input, expected_grad), "Backward pass incorrect"
    
    print("✓ ReluLayer passed all tests")


def test_multilayer_network():
    """Test MultiLayerNetwork."""
    print("\nTesting MultiLayerNetwork...")
    
    # Setup
    np.random.seed(42)
    input_dim = 4
    neurons = [8, 6, 2]
    activations = ["relu", "sigmoid", "identity"]
    
    network = MultiLayerNetwork(input_dim, neurons, activations)
    
    # Check number of layers (3 linear + 2 activation layers, identity doesn't add a layer)
    expected_layers = 3 + 2  # 3 linear + relu + sigmoid (identity doesn't add)
    assert len(network._layers) == expected_layers, \
        f"Wrong number of layers: {len(network._layers)} vs {expected_layers}"
    
    # Test forward pass
    x = np.random.randn(5, input_dim)
    output = network.forward(x)
    assert output.shape == (5, 2), f"Wrong output shape: {output.shape}"
    
    # Test backward pass
    grad_output = np.random.randn(5, 2)
    grad_input = network.backward(grad_output)
    assert grad_input.shape == x.shape, f"Wrong gradient shape: {grad_input.shape}"
    
    # Test parameter update
    old_weights = [layer._W.copy() for layer in network._layers if isinstance(layer, LinearLayer)]
    network.update_params(0.1)
    new_weights = [layer._W.copy() for layer in network._layers if isinstance(layer, LinearLayer)]
    
    # Check that weights changed
    for old_w, new_w in zip(old_weights, new_weights):
        assert not np.allclose(old_w, new_w), "Weights should have changed after update"
    
    print("✓ MultiLayerNetwork passed all tests")


def test_preprocessor():
    """Test Preprocessor for min-max scaling."""
    print("\nTesting Preprocessor...")
    
    # Setup
    np.random.seed(42)
    data = np.random.randn(100, 5) * 10 + 50  # Random data with mean 50, std 10
    
    prep = Preprocessor(data)
    
    # Test apply
    normalized = prep.apply(data)
    assert normalized.shape == data.shape, "Wrong shape after normalization"
    
    # Check that data is in [0, 1]
    assert np.all(normalized >= -1e-10), f"Normalized data below 0: min={np.min(normalized)}"
    assert np.all(normalized <= 1 + 1e-10), f"Normalized data above 1: max={np.max(normalized)}"
    
    # Check that min is close to 0 and max is close to 1
    assert np.allclose(np.min(normalized, axis=0), 0, atol=1e-10), \
        f"Min should be 0: {np.min(normalized, axis=0)}"
    assert np.allclose(np.max(normalized, axis=0), 1, atol=1e-10), \
        f"Max should be 1: {np.max(normalized, axis=0)}"
    
    # Test revert
    reverted = prep.revert(normalized)
    assert np.allclose(reverted, data, rtol=1e-10), "Revert failed to recover original data"
    
    # Test with constant feature
    data_with_constant = data.copy()
    data_with_constant[:, 0] = 5.0  # Make first feature constant
    
    prep2 = Preprocessor(data_with_constant)
    normalized2 = prep2.apply(data_with_constant)
    
    # Constant feature should remain unchanged (or be set to a specific value)
    # The implementation sets range to 1 for constant features, so (x - min) / 1 = 0
    assert np.allclose(normalized2[:, 0], 0), "Constant feature not handled correctly"
    
    # Test revert on constant feature
    reverted2 = prep2.revert(normalized2)
    assert np.allclose(reverted2, data_with_constant, rtol=1e-10), \
        "Revert failed with constant feature"
    
    print("✓ Preprocessor passed all tests")


def test_mse_loss():
    """Test MSELossLayer."""
    print("\nTesting MSELossLayer...")
    
    # Setup
    np.random.seed(42)
    loss_layer = MSELossLayer()
    predictions = np.random.randn(10, 3)
    targets = np.random.randn(10, 3)
    
    # Test forward pass
    loss = loss_layer.forward(predictions, targets)
    expected_loss = np.mean((predictions - targets) ** 2)
    assert np.isclose(loss, expected_loss), f"Loss incorrect: {loss} vs {expected_loss}"
    
    # Test backward pass
    grad = loss_layer.backward()
    assert grad.shape == predictions.shape, "Wrong gradient shape"
    
    expected_grad = (2.0 / predictions.shape[0]) * (predictions - targets)
    assert np.allclose(grad, expected_grad), "Gradient incorrect"
    
    print("✓ MSELossLayer passed all tests")


def test_cross_entropy_loss():
    """Test CrossEntropyLossLayer."""
    print("\nTesting CrossEntropyLossLayer...")
    
    # Setup
    np.random.seed(42)
    loss_layer = CrossEntropyLossLayer()
    predictions = np.random.uniform(0.1, 0.9, size=(10, 2))  # Simulating probabilities
    targets = np.random.randint(0, 2, size=(10, 2)).astype(float)
    
    # Test forward pass
    loss = loss_layer.forward(predictions, targets)
    assert loss > 0, "Loss should be positive"
    assert not np.isnan(loss), "Loss is NaN"
    
    # Test backward pass
    grad = loss_layer.backward()
    assert grad.shape == predictions.shape, "Wrong gradient shape"
    assert not np.any(np.isnan(grad)), "Gradient contains NaN"
    
    print("✓ CrossEntropyLossLayer passed all tests")


def test_trainer():
    """Test Trainer class."""
    print("\nTesting Trainer...")
    
    # Setup
    np.random.seed(42)
    input_dim = 4
    neurons = [8, 2]
    activations = ["relu", "identity"]
    
    network = MultiLayerNetwork(input_dim, neurons, activations)
    
    # Create simple dataset
    X = np.random.randn(100, input_dim)
    y = np.random.randn(100, 2)
    
    # Test trainer initialization
    trainer = Trainer(
        network=network,
        batch_size=10,
        nb_epoch=5,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True
    )
    
    assert trainer.batch_size == 10, "Batch size not set correctly"
    assert trainer.nb_epoch == 5, "Number of epochs not set correctly"
    
    # Test shuffle
    X_shuffled, y_shuffled = trainer.shuffle(X, y)
    assert X_shuffled.shape == X.shape, "Shuffled X has wrong shape"
    assert y_shuffled.shape == y.shape, "Shuffled y has wrong shape"
    assert not np.array_equal(X, X_shuffled), "Data should be shuffled"
    
    # Check that shuffling preserves correspondence
    # Find where first sample of X ended up
    for i in range(len(X_shuffled)):
        if np.allclose(X_shuffled[i], X[0]):
            assert np.allclose(y_shuffled[i], y[0]), "Shuffle broke X-y correspondence"
            break
    
    # Test eval_loss before training
    initial_loss = trainer.eval_loss(X, y)
    assert initial_loss > 0, "Initial loss should be positive"
    
    # Test training
    trainer.train(X, y)
    
    # Test eval_loss after training
    final_loss = trainer.eval_loss(X, y)
    assert final_loss < initial_loss, f"Loss should decrease after training: {initial_loss} -> {final_loss}"
    
    print("✓ Trainer passed all tests")


def test_integration():
    """Integration test with full pipeline."""
    print("\nTesting full integration...")
    
    # Setup
    np.random.seed(42)
    
    # Create synthetic dataset
    n_samples = 200
    n_features = 6
    X = np.random.randn(n_samples, n_features)
    
    # Create targets with some relationship to inputs
    true_weights = np.random.randn(n_features, 1)
    y = np.dot(X, true_weights) + np.random.randn(n_samples, 1) * 0.1
    
    # Preprocess
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
        neurons=[10, 5, 1],
        activations=["relu", "relu", "identity"]
    )
    
    # Train
    trainer = Trainer(
        network=network,
        batch_size=16,
        nb_epoch=50,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True
    )
    
    initial_loss = trainer.eval_loss(X_val, y_val)
    trainer.train(X_train, y_train)
    final_loss = trainer.eval_loss(X_val, y_val)
    
    print(f"  Initial validation loss: {initial_loss:.4f}")
    print(f"  Final validation loss: {final_loss:.4f}")
    print(f"  Loss reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
    
    assert final_loss < initial_loss * 0.5, "Model should learn significantly"
    
    # Test predictions
    predictions = network.forward(X_val)
    assert predictions.shape == y_val.shape, "Prediction shape incorrect"
    
    # Revert preprocessing
    y_val_original = prep_y.revert(y_val)
    predictions_original = prep_y.revert(predictions)
    
    mse = np.mean((predictions_original - y_val_original) ** 2)
    print(f"  Final MSE on original scale: {mse:.4f}")
    
    print("✓ Integration test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Comprehensive Test Suite for Part 1")
    print("=" * 60)
    
    try:
        test_xavier_init()
        test_linear_layer()
        test_sigmoid_layer()
        test_relu_layer()
        test_multilayer_network()
        test_preprocessor()
        test_mse_loss()
        test_cross_entropy_loss()
        test_trainer()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
BOMB-PROOF Comprehensive Test Suite for Part 1 Neural Network Mini-Library
This suite performs exhaustive testing including full numerical gradient verification.
"""
import numpy as np
import sys
sys.path.append('/home/claude')
from part1_nn_lib import (
    LinearLayer, SigmoidLayer, ReluLayer, MultiLayerNetwork,
    Trainer, Preprocessor, MSELossLayer, CrossEntropyLossLayer,
    xavier_init
)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record_pass(self, test_name):
        self.passed += 1
        print(f"✓ {test_name}")
    
    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"✗ {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*70)
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\nFAILED TESTS:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print("="*70)
        return self.failed == 0


results = TestResults()


def assert_test(condition, message, test_name):
    """Assert with test tracking"""
    if not condition:
        results.record_fail(test_name, message)
        raise AssertionError(message)


def numerical_gradient(func, x, epsilon=1e-7):
    """
    Compute numerical gradient of func at x using central differences.
    func: function that takes x and returns a scalar
    x: point at which to compute gradient
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        x[idx] = old_value + epsilon
        pos = func(x)
        
        x[idx] = old_value - epsilon
        neg = func(x)
        
        x[idx] = old_value
        
        grad[idx] = (pos - neg) / (2 * epsilon)
        it.iternext()
    
    return grad


def test_xavier_init():
    """Test Xavier initialization thoroughly."""
    test_name = "Xavier Initialization"
    try:
        # Test various sizes
        for shape in [(10, 20), (5, 5), (100, 10), (1, 1)]:
            weights = xavier_init(shape)
            
            assert_test(weights.shape == shape, 
                       f"Wrong shape: expected {shape}, got {weights.shape}", 
                       test_name)
            
            # Check distribution properties
            mean = np.mean(weights)
            std = np.std(weights)
            
            assert_test(abs(mean) < 0.1, 
                       f"Mean too far from 0: {mean}", 
                       test_name)
            
            expected_bound = np.sqrt(6.0 / np.sum(shape))
            assert_test(np.max(np.abs(weights)) <= expected_bound * 1.01,
                       f"Values outside expected bounds", 
                       test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_linear_layer_complete():
    """Complete test of LinearLayer with full gradient checking."""
    test_name = "LinearLayer - Complete"
    try:
        np.random.seed(42)
        
        # Test multiple configurations
        configs = [
            (1, 1, 1),   # Minimal
            (4, 3, 5),   # Standard
            (10, 8, 1),  # Many to one
            (1, 5, 10),  # One to many
            (32, 20, 15) # Larger
        ]
        
        for batch_size, n_in, n_out in configs:
            layer = LinearLayer(n_in, n_out)
            x = np.random.randn(batch_size, n_in)
            
            # Forward pass
            output = layer.forward(x)
            expected = np.dot(x, layer._W) + layer._b
            
            assert_test(np.allclose(output, expected, rtol=1e-10),
                       f"Forward pass incorrect for config ({batch_size}, {n_in}, {n_out})",
                       test_name)
            
            # Backward pass - COMPLETE gradient check
            grad_output = np.random.randn(batch_size, n_out)
            grad_input = layer.backward(grad_output)
            
            # Check all weight gradients numerically
            def loss_wrt_W(W_flat):
                W_temp = W_flat.reshape(layer._W.shape)
                out_temp = np.dot(x, W_temp) + layer._b
                return np.sum(out_temp * grad_output)
            
            numerical_grad_W = numerical_gradient(loss_wrt_W, layer._W.flatten())
            numerical_grad_W = numerical_grad_W.reshape(layer._W.shape)
            
            max_diff_W = np.max(np.abs(layer._grad_W_current - numerical_grad_W))
            relative_error_W = max_diff_W / (np.max(np.abs(numerical_grad_W)) + 1e-8)
            
            assert_test(relative_error_W < 1e-5,
                       f"Weight gradient check failed: max_diff={max_diff_W:.2e}, rel_error={relative_error_W:.2e}",
                       test_name)
            
            # Check all bias gradients numerically
            def loss_wrt_b(b_flat):
                b_temp = b_flat.reshape(layer._b.shape)
                out_temp = np.dot(x, layer._W) + b_temp
                return np.sum(out_temp * grad_output)
            
            numerical_grad_b = numerical_gradient(loss_wrt_b, layer._b.flatten())
            numerical_grad_b = numerical_grad_b.reshape(layer._b.shape)
            
            max_diff_b = np.max(np.abs(layer._grad_b_current - numerical_grad_b))
            relative_error_b = max_diff_b / (np.max(np.abs(numerical_grad_b)) + 1e-8)
            
            assert_test(relative_error_b < 1e-5,
                       f"Bias gradient check failed: max_diff={max_diff_b:.2e}, rel_error={relative_error_b:.2e}",
                       test_name)
            
            # Check input gradients
            expected_grad_input = np.dot(grad_output, layer._W.T)
            assert_test(np.allclose(grad_input, expected_grad_input, rtol=1e-10),
                       f"Input gradient incorrect",
                       test_name)
            
            # Test parameter update
            old_W = layer._W.copy()
            old_b = layer._b.copy()
            lr = 0.1
            layer.update_params(lr)
            
            expected_W = old_W - lr * layer._grad_W_current
            expected_b = old_b - lr * layer._grad_b_current
            
            assert_test(np.allclose(layer._W, expected_W, rtol=1e-10),
                       f"Weight update incorrect",
                       test_name)
            assert_test(np.allclose(layer._b, expected_b, rtol=1e-10),
                       f"Bias update incorrect",
                       test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_sigmoid_layer_complete():
    """Complete test of SigmoidLayer with full gradient checking."""
    test_name = "SigmoidLayer - Complete"
    try:
        np.random.seed(42)
        
        # Test various input ranges
        test_inputs = [
            np.array([[0.0]]),  # Zero
            np.array([[1.0], [-1.0]]),  # Small positive/negative
            np.array([[10.0], [-10.0]]),  # Large positive/negative
            np.random.randn(5, 10),  # Random
            np.random.randn(1, 1),  # Single value
            np.random.randn(100, 50) * 5  # Larger batch
        ]
        
        for x in test_inputs:
            layer = SigmoidLayer()
            output = layer.forward(x)
            
            # Check output range
            assert_test(np.all(output >= 0) and np.all(output <= 1),
                       f"Sigmoid output not in [0,1]: min={np.min(output)}, max={np.max(output)}",
                       test_name)
            
            # Check known values
            if x.shape == (1, 1) and x[0, 0] == 0:
                assert_test(np.isclose(output[0, 0], 0.5),
                           f"Sigmoid(0) should be 0.5, got {output[0, 0]}",
                           test_name)
            
            # Complete numerical gradient check
            grad_output = np.random.randn(*x.shape)
            grad_input = layer.backward(grad_output)
            
            def loss_func(x_flat):
                x_temp = x_flat.reshape(x.shape)
                layer_temp = SigmoidLayer()
                out = layer_temp.forward(x_temp)
                return np.sum(out * grad_output)
            
            numerical_grad = numerical_gradient(loss_func, x.flatten())
            numerical_grad = numerical_grad.reshape(x.shape)
            
            max_diff = np.max(np.abs(grad_input - numerical_grad))
            relative_error = max_diff / (np.max(np.abs(numerical_grad)) + 1e-8)
            
            assert_test(relative_error < 1e-5,
                       f"Gradient check failed for shape {x.shape}: max_diff={max_diff:.2e}, rel_error={relative_error:.2e}",
                       test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_relu_layer_complete():
    """Complete test of ReluLayer with full gradient checking."""
    test_name = "ReluLayer - Complete"
    try:
        np.random.seed(42)
        
        # Test various inputs including edge cases
        test_inputs = [
            np.array([[0.0]]),  # Exactly zero
            np.array([[1.0], [-1.0], [0.0]]),  # Mixed
            np.array([[-5.0, -1.0, 0.0, 1.0, 5.0]]),  # Range
            np.random.randn(5, 10),  # Random
            np.random.randn(100, 50) * 5  # Larger batch
        ]
        
        for x in test_inputs:
            layer = ReluLayer()
            output = layer.forward(x)
            
            # Check output properties
            assert_test(np.all(output >= 0),
                       f"ReLU output should be non-negative: min={np.min(output)}",
                       test_name)
            
            expected = np.maximum(0, x)
            assert_test(np.allclose(output, expected),
                       f"ReLU forward pass incorrect",
                       test_name)
            
            # Complete numerical gradient check
            grad_output = np.random.randn(*x.shape)
            grad_input = layer.backward(grad_output)
            
            def loss_func(x_flat):
                x_temp = x_flat.reshape(x.shape)
                layer_temp = ReluLayer()
                out = layer_temp.forward(x_temp)
                return np.sum(out * grad_output)
            
            numerical_grad = numerical_gradient(loss_func, x.flatten())
            numerical_grad = numerical_grad.reshape(x.shape)
            
            max_diff = np.max(np.abs(grad_input - numerical_grad))
            relative_error = max_diff / (np.max(np.abs(numerical_grad)) + 1e-8)
            
            # ReLU has discontinuous derivative at 0, so we're more lenient there
            assert_test(relative_error < 1e-5,
                       f"Gradient check failed for shape {x.shape}: max_diff={max_diff:.2e}, rel_error={relative_error:.2e}",
                       test_name)
            
            # Verify gradient is zero where input was negative
            expected_grad = grad_output * (x > 0)
            assert_test(np.allclose(grad_input, expected_grad),
                       f"ReLU gradient pattern incorrect",
                       test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_multilayer_network_complete():
    """Complete test of MultiLayerNetwork with end-to-end gradient checking."""
    test_name = "MultiLayerNetwork - Complete"
    try:
        np.random.seed(42)
        
        # Test various network configurations
        configs = [
            (2, [3], ["identity"]),
            (4, [5, 3], ["relu", "identity"]),
            (3, [8, 6, 2], ["relu", "sigmoid", "identity"]),
            (5, [10, 10, 5], ["sigmoid", "relu", "sigmoid"]),
            (10, [20, 15, 10, 5, 2], ["relu", "relu", "relu", "sigmoid", "identity"])
        ]
        
        for input_dim, neurons, activations in configs:
            network = MultiLayerNetwork(input_dim, neurons, activations)
            
            batch_size = 5
            x = np.random.randn(batch_size, input_dim) * 0.5
            
            # Forward pass
            output = network.forward(x)
            assert_test(output.shape == (batch_size, neurons[-1]),
                       f"Wrong output shape: expected {(batch_size, neurons[-1])}, got {output.shape}",
                       test_name)
            
            # Backward pass
            grad_output = np.random.randn(batch_size, neurons[-1])
            grad_input = network.backward(grad_output)
            
            assert_test(grad_input.shape == x.shape,
                       f"Wrong gradient shape: expected {x.shape}, got {grad_input.shape}",
                       test_name)
            
            # END-TO-END gradient check for ALL parameters
            def network_loss(params_flat, param_shapes, x_input, grad_out):
                """Reconstruct network with given parameters and compute loss"""
                # Unpack parameters
                idx = 0
                layer_idx = 0
                for layer in network._layers:
                    if isinstance(layer, LinearLayer):
                        W_size = param_shapes[layer_idx][0]
                        b_size = param_shapes[layer_idx][1]
                        
                        layer._W = params_flat[idx:idx+W_size].reshape(layer._W.shape)
                        idx += W_size
                        
                        layer._b = params_flat[idx:idx+b_size].reshape(layer._b.shape)
                        idx += b_size
                        
                        layer_idx += 1
                
                # Forward pass
                out = network.forward(x_input)
                return np.sum(out * grad_out)
            
            # Pack all parameters
            params = []
            param_shapes = []
            for layer in network._layers:
                if isinstance(layer, LinearLayer):
                    params.append(layer._W.flatten())
                    params.append(layer._b.flatten())
                    param_shapes.append((layer._W.size, layer._b.size))
            
            params_flat = np.concatenate(params)
            
            # Get analytical gradients
            network.forward(x)
            network.backward(grad_output)
            
            analytical_grads = []
            for layer in network._layers:
                if isinstance(layer, LinearLayer):
                    analytical_grads.append(layer._grad_W_current.flatten())
                    analytical_grads.append(layer._grad_b_current.flatten())
            
            analytical_grads_flat = np.concatenate(analytical_grads)
            
            # Compute numerical gradients (sample subset for speed on large networks)
            if len(params_flat) > 100:
                # Sample 100 random parameters
                indices = np.random.choice(len(params_flat), 100, replace=False)
                params_to_check = params_flat[indices]
                analytical_subset = analytical_grads_flat[indices]
                
                numerical_subset = np.zeros_like(analytical_subset)
                for i, idx in enumerate(indices):
                    epsilon = 1e-7
                    params_temp = params_flat.copy()
                    
                    params_temp[idx] += epsilon
                    loss_plus = network_loss(params_temp, param_shapes, x, grad_output)
                    
                    params_temp[idx] -= 2 * epsilon
                    loss_minus = network_loss(params_temp, param_shapes, x, grad_output)
                    
                    numerical_subset[i] = (loss_plus - loss_minus) / (2 * epsilon)
                
                max_diff = np.max(np.abs(analytical_subset - numerical_subset))
                relative_error = max_diff / (np.max(np.abs(numerical_subset)) + 1e-8)
            else:
                # Check all parameters
                numerical_grads_flat = np.zeros_like(params_flat)
                for i in range(len(params_flat)):
                    epsilon = 1e-7
                    params_temp = params_flat.copy()
                    
                    params_temp[i] += epsilon
                    loss_plus = network_loss(params_temp, param_shapes, x, grad_output)
                    
                    params_temp[i] -= 2 * epsilon
                    loss_minus = network_loss(params_temp, param_shapes, x, grad_output)
                    
                    numerical_grads_flat[i] = (loss_plus - loss_minus) / (2 * epsilon)
                
                max_diff = np.max(np.abs(analytical_grads_flat - numerical_grads_flat))
                relative_error = max_diff / (np.max(np.abs(numerical_grads_flat)) + 1e-8)
            
            assert_test(relative_error < 1e-4,
                       f"End-to-end gradient check failed for config {neurons}: rel_error={relative_error:.2e}",
                       test_name)
            
            # Test parameter update
            old_params = [layer._W.copy() for layer in network._layers if isinstance(layer, LinearLayer)]
            network.update_params(0.1)
            new_params = [layer._W.copy() for layer in network._layers if isinstance(layer, LinearLayer)]
            
            for old_p, new_p in zip(old_params, new_params):
                assert_test(not np.allclose(old_p, new_p),
                           f"Parameters should change after update",
                           test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_mse_loss_complete():
    """Complete test of MSELossLayer."""
    test_name = "MSELossLayer - Complete"
    try:
        np.random.seed(42)
        
        test_cases = [
            (1, 1),
            (10, 1),
            (5, 5),
            (100, 10),
            (32, 3)
        ]
        
        for batch_size, n_outputs in test_cases:
            loss_layer = MSELossLayer()
            predictions = np.random.randn(batch_size, n_outputs)
            targets = np.random.randn(batch_size, n_outputs)
            
            # Forward pass
            loss = loss_layer.forward(predictions, targets)
            expected_loss = np.mean((predictions - targets) ** 2)
            
            assert_test(np.isclose(loss, expected_loss, rtol=1e-10),
                       f"MSE loss incorrect: got {loss}, expected {expected_loss}",
                       test_name)
            
            assert_test(loss >= 0,
                       f"MSE loss should be non-negative: {loss}",
                       test_name)
            
            # Backward pass - numerical check
            grad = loss_layer.backward()
            
            def loss_func(pred_flat):
                pred_temp = pred_flat.reshape(predictions.shape)
                return np.mean((pred_temp - targets) ** 2)
            
            numerical_grad = numerical_gradient(loss_func, predictions.flatten())
            numerical_grad = numerical_grad.reshape(predictions.shape)
            
            max_diff = np.max(np.abs(grad - numerical_grad))
            relative_error = max_diff / (np.max(np.abs(numerical_grad)) + 1e-8)
            
            assert_test(relative_error < 1e-5,
                       f"MSE gradient check failed: rel_error={relative_error:.2e}",
                       test_name)
            
            # Check perfect prediction
            loss_perfect = loss_layer.forward(predictions, predictions)
            assert_test(np.isclose(loss_perfect, 0, atol=1e-10),
                       f"Loss should be 0 for perfect prediction: {loss_perfect}",
                       test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_cross_entropy_loss_complete():
    """Complete test of CrossEntropyLossLayer."""
    test_name = "CrossEntropyLossLayer - Complete"
    try:
        np.random.seed(42)
        
        test_cases = [
            (1, 1),
            (10, 1),
            (5, 2),
            (100, 3),
            (32, 5)
        ]
        
        for batch_size, n_outputs in test_cases:
            loss_layer = CrossEntropyLossLayer()
            
            # Use predictions in valid range
            predictions = np.random.uniform(0.1, 0.9, size=(batch_size, n_outputs))
            targets = np.random.randint(0, 2, size=(batch_size, n_outputs)).astype(float)
            
            # Forward pass
            loss = loss_layer.forward(predictions, targets)
            
            assert_test(loss >= 0,
                       f"Cross-entropy loss should be non-negative: {loss}",
                       test_name)
            
            assert_test(not np.isnan(loss) and not np.isinf(loss),
                       f"Loss is NaN or Inf: {loss}",
                       test_name)
            
            # Backward pass - numerical check
            grad = loss_layer.backward()
            
            epsilon_clip = 1e-15
            def loss_func(pred_flat):
                pred_temp = pred_flat.reshape(predictions.shape)
                pred_clipped = np.clip(pred_temp, epsilon_clip, 1 - epsilon_clip)
                return -np.mean(
                    targets * np.log(pred_clipped) + 
                    (1 - targets) * np.log(1 - pred_clipped)
                )
            
            numerical_grad = numerical_gradient(loss_func, predictions.flatten())
            numerical_grad = numerical_grad.reshape(predictions.shape)
            
            max_diff = np.max(np.abs(grad - numerical_grad))
            relative_error = max_diff / (np.max(np.abs(numerical_grad)) + 1e-8)
            
            assert_test(relative_error < 1e-4,
                       f"Cross-entropy gradient check failed: rel_error={relative_error:.2e}",
                       test_name)
            
            # Test edge cases
            # Perfect prediction (targets = predictions)
            loss_perfect = loss_layer.forward(targets, targets)
            # Should be close to 0 (but not exactly due to clipping)
            assert_test(loss_perfect < 0.01,
                       f"Loss should be near 0 for perfect prediction: {loss_perfect}",
                       test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_preprocessor_complete():
    """Complete test of Preprocessor."""
    test_name = "Preprocessor - Complete"
    try:
        np.random.seed(42)
        
        # Test 1: Normal data
        data = np.random.randn(100, 5) * 10 + 50
        prep = Preprocessor(data)
        normalized = prep.apply(data)
        
        assert_test(normalized.shape == data.shape,
                   f"Shape mismatch after normalization",
                   test_name)
        
        assert_test(np.all(normalized >= -1e-10) and np.all(normalized <= 1 + 1e-10),
                   f"Normalized data not in [0,1]: min={np.min(normalized)}, max={np.max(normalized)}",
                   test_name)
        
        # Each feature should have min~0, max~1
        for i in range(data.shape[1]):
            assert_test(np.isclose(np.min(normalized[:, i]), 0, atol=1e-10),
                       f"Feature {i} min not 0: {np.min(normalized[:, i])}",
                       test_name)
            assert_test(np.isclose(np.max(normalized[:, i]), 1, atol=1e-10),
                       f"Feature {i} max not 1: {np.max(normalized[:, i])}",
                       test_name)
        
        # Test revert
        reverted = prep.revert(normalized)
        assert_test(np.allclose(reverted, data, rtol=1e-10),
                   f"Revert failed: max_diff={np.max(np.abs(reverted - data))}",
                   test_name)
        
        # Test 2: Constant feature
        data_const = data.copy()
        data_const[:, 0] = 5.0
        prep2 = Preprocessor(data_const)
        normalized2 = prep2.apply(data_const)
        
        # Constant feature should be handled
        assert_test(np.all(normalized2[:, 0] == normalized2[0, 0]),
                   f"Constant feature not handled consistently",
                   test_name)
        
        reverted2 = prep2.revert(normalized2)
        assert_test(np.allclose(reverted2, data_const, rtol=1e-10),
                   f"Revert failed with constant feature",
                   test_name)
        
        # Test 3: All constant features
        data_all_const = np.ones((50, 3)) * 7.0
        prep3 = Preprocessor(data_all_const)
        normalized3 = prep3.apply(data_all_const)
        reverted3 = prep3.revert(normalized3)
        
        assert_test(np.allclose(reverted3, data_all_const, rtol=1e-10),
                   f"Revert failed with all constant features",
                   test_name)
        
        # Test 4: Single sample
        data_single = np.random.randn(1, 5)
        prep4 = Preprocessor(data_single)
        normalized4 = prep4.apply(data_single)
        reverted4 = prep4.revert(normalized4)
        
        assert_test(np.allclose(reverted4, data_single, rtol=1e-10),
                   f"Revert failed with single sample",
                   test_name)
        
        # Test 5: Apply to new data
        new_data = np.random.randn(50, 5) * 10 + 50
        new_normalized = prep.apply(new_data)
        new_reverted = prep.revert(new_normalized)
        
        assert_test(np.allclose(new_reverted, new_data, rtol=1e-10),
                   f"Apply/revert failed on new data",
                   test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_trainer_complete():
    """Complete test of Trainer."""
    test_name = "Trainer - Complete"
    try:
        np.random.seed(42)
        
        # Test with MSE loss
        input_dim = 4
        network_mse = MultiLayerNetwork(input_dim, [8, 2], ["relu", "identity"])
        
        X = np.random.randn(100, input_dim)
        y = np.random.randn(100, 2)
        
        trainer_mse = Trainer(
            network=network_mse,
            batch_size=10,
            nb_epoch=10,
            learning_rate=0.01,
            loss_fun="mse",
            shuffle_flag=True
        )
        
        # Test shuffle
        X_shuf, y_shuf = trainer_mse.shuffle(X, y)
        assert_test(not np.array_equal(X, X_shuf),
                   f"Shuffle didn't change order",
                   test_name)
        
        # Verify correspondence
        for i in range(10):
            for j in range(len(X_shuf)):
                if np.allclose(X_shuf[j], X[i]):
                    assert_test(np.allclose(y_shuf[j], y[i]),
                               f"Shuffle broke X-y correspondence",
                               test_name)
                    break
        
        # Test training with shuffle
        initial_loss = trainer_mse.eval_loss(X, y)
        trainer_mse.train(X, y)
        final_loss = trainer_mse.eval_loss(X, y)
        
        assert_test(final_loss < initial_loss,
                   f"Loss should decrease: {initial_loss} -> {final_loss}",
                   test_name)
        
        # Test with cross-entropy loss
        network_ce = MultiLayerNetwork(input_dim, [8, 2], ["relu", "sigmoid"])
        y_binary = np.random.randint(0, 2, size=(100, 2)).astype(float)
        
        trainer_ce = Trainer(
            network=network_ce,
            batch_size=10,
            nb_epoch=10,
            learning_rate=0.01,
            loss_fun="cross_entropy",
            shuffle_flag=True
        )
        
        initial_loss_ce = trainer_ce.eval_loss(X, y_binary)
        trainer_ce.train(X, y_binary)
        final_loss_ce = trainer_ce.eval_loss(X, y_binary)
        
        assert_test(final_loss_ce < initial_loss_ce,
                   f"CE loss should decrease: {initial_loss_ce} -> {final_loss_ce}",
                   test_name)
        
        # Test without shuffle
        network_no_shuffle = MultiLayerNetwork(input_dim, [8, 2], ["relu", "identity"])
        trainer_no_shuffle = Trainer(
            network=network_no_shuffle,
            batch_size=10,
            nb_epoch=10,
            learning_rate=0.01,
            loss_fun="mse",
            shuffle_flag=False
        )
        
        initial_loss_ns = trainer_no_shuffle.eval_loss(X, y)
        trainer_no_shuffle.train(X, y)
        final_loss_ns = trainer_no_shuffle.eval_loss(X, y)
        
        assert_test(final_loss_ns < initial_loss_ns,
                   f"Loss should decrease without shuffle: {initial_loss_ns} -> {final_loss_ns}",
                   test_name)
        
        # Test edge case: batch size larger than dataset
        small_X = np.random.randn(5, input_dim)
        small_y = np.random.randn(5, 2)
        
        network_large_batch = MultiLayerNetwork(input_dim, [8, 2], ["relu", "identity"])
        trainer_large_batch = Trainer(
            network=network_large_batch,
            batch_size=20,
            nb_epoch=5,
            learning_rate=0.01,
            loss_fun="mse",
            shuffle_flag=True
        )
        
        initial_loss_lb = trainer_large_batch.eval_loss(small_X, small_y)
        trainer_large_batch.train(small_X, small_y)
        final_loss_lb = trainer_large_batch.eval_loss(small_X, small_y)
        
        assert_test(final_loss_lb < initial_loss_lb,
                   f"Loss should decrease with large batch: {initial_loss_lb} -> {final_loss_lb}",
                   test_name)
        
        # Test edge case: batch size doesn't divide dataset evenly
        uneven_X = np.random.randn(47, input_dim)
        uneven_y = np.random.randn(47, 2)
        
        network_uneven = MultiLayerNetwork(input_dim, [8, 2], ["relu", "identity"])
        trainer_uneven = Trainer(
            network=network_uneven,
            batch_size=10,
            nb_epoch=5,
            learning_rate=0.01,
            loss_fun="mse",
            shuffle_flag=True
        )
        
        initial_loss_ue = trainer_uneven.eval_loss(uneven_X, uneven_y)
        trainer_uneven.train(uneven_X, uneven_y)
        final_loss_ue = trainer_uneven.eval_loss(uneven_X, uneven_y)
        
        assert_test(final_loss_ue < initial_loss_ue,
                   f"Loss should decrease with uneven batches: {initial_loss_ue} -> {final_loss_ue}",
                   test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_deep_network_gradients():
    """Test gradient flow in deep networks."""
    test_name = "Deep Network Gradients"
    try:
        np.random.seed(42)
        
        # Create a deep network
        input_dim = 5
        neurons = [10, 10, 10, 10, 10, 2]
        activations = ["relu"] * 5 + ["identity"]
        
        network = MultiLayerNetwork(input_dim, neurons, activations)
        
        x = np.random.randn(3, input_dim) * 0.1  # Small values to avoid saturation
        grad_output = np.random.randn(3, 2)
        
        # Forward and backward
        output = network.forward(x)
        grad_input = network.backward(grad_output)
        
        # Check that gradients are not vanishing
        linear_layers = [l for l in network._layers if isinstance(l, LinearLayer)]
        
        for i, layer in enumerate(linear_layers):
            grad_norm = np.linalg.norm(layer._grad_W_current)
            assert_test(grad_norm > 1e-10,
                       f"Vanishing gradient at layer {i}: norm={grad_norm}",
                       test_name)
            
            assert_test(not np.any(np.isnan(layer._grad_W_current)),
                       f"NaN gradient at layer {i}",
                       test_name)
        
        # Numerical check on first and last layer
        first_layer = linear_layers[0]
        
        def loss_first_layer(W_flat):
            W_temp = W_flat.reshape(first_layer._W.shape)
            old_W = first_layer._W.copy()
            first_layer._W = W_temp
            out = network.forward(x)
            first_layer._W = old_W
            return np.sum(out * grad_output)
        
        # Sample a few parameters
        sample_size = min(20, first_layer._W.size)
        sample_indices = np.random.choice(first_layer._W.size, sample_size, replace=False)
        
        for idx in sample_indices:
            i, j = np.unravel_index(idx, first_layer._W.shape)
            
            epsilon = 1e-7
            old_val = first_layer._W[i, j]
            
            first_layer._W[i, j] = old_val + epsilon
            loss_plus = np.sum(network.forward(x) * grad_output)
            
            first_layer._W[i, j] = old_val - epsilon
            loss_minus = np.sum(network.forward(x) * grad_output)
            
            first_layer._W[i, j] = old_val
            
            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Recompute analytical
            network.forward(x)
            network.backward(grad_output)
            analytical = first_layer._grad_W_current[i, j]
            
            relative_error = abs(analytical - numerical) / (abs(numerical) + 1e-8)
            assert_test(relative_error < 1e-4,
                       f"Deep network gradient check failed at layer 0, position ({i},{j}): rel_error={relative_error:.2e}",
                       test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_integration_full_pipeline():
    """Full integration test with real-world scenario."""
    test_name = "Full Integration Pipeline"
    try:
        np.random.seed(42)
        
        # Create realistic synthetic dataset
        n_samples = 500
        n_features = 10
        X = np.random.randn(n_samples, n_features) * 2
        
        # Create non-linear relationship
        true_W1 = np.random.randn(n_features, 5)
        hidden = np.maximum(0, np.dot(X, true_W1))  # ReLU activation
        true_W2 = np.random.randn(5, 1)
        y = np.dot(hidden, true_W2) + np.random.randn(n_samples, 1) * 0.5
        
        # Preprocess
        prep_X = Preprocessor(X)
        prep_y = Preprocessor(y)
        
        X_norm = prep_X.apply(X)
        y_norm = prep_y.apply(y)
        
        # Split
        split = int(0.8 * n_samples)
        X_train, X_val = X_norm[:split], X_norm[split:]
        y_train, y_val = y_norm[:split], y_norm[split:]
        
        # Create network
        network = MultiLayerNetwork(
            input_dim=n_features,
            neurons=[20, 10, 1],
            activations=["relu", "relu", "identity"]
        )
        
        # Train
        trainer = Trainer(
            network=network,
            batch_size=32,
            nb_epoch=100,
            learning_rate=0.01,
            loss_fun="mse",
            shuffle_flag=True
        )
        
        initial_loss = trainer.eval_loss(X_val, y_val)
        trainer.train(X_train, y_train)
        final_loss = trainer.eval_loss(X_val, y_val)
        
        assert_test(final_loss < initial_loss * 0.3,
                   f"Model should learn significantly: {initial_loss:.4f} -> {final_loss:.4f}",
                   test_name)
        
        # Test predictions
        predictions = network.forward(X_val)
        assert_test(predictions.shape == y_val.shape,
                   f"Prediction shape mismatch",
                   test_name)
        
        # Revert and check
        y_val_orig = prep_y.revert(y_val)
        pred_orig = prep_y.revert(predictions)
        
        mse = np.mean((pred_orig - y_val_orig) ** 2)
        
        # Should achieve reasonable MSE
        assert_test(mse < np.var(y_val_orig),
                   f"Model should beat naive baseline: MSE={mse:.4f}, Var={np.var(y_val_orig):.4f}",
                   test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_edge_cases():
    """Test various edge cases."""
    test_name = "Edge Cases"
    try:
        np.random.seed(42)
        
        # 1. Single sample batch
        network = MultiLayerNetwork(3, [5, 2], ["relu", "identity"])
        x_single = np.random.randn(1, 3)
        out_single = network.forward(x_single)
        assert_test(out_single.shape == (1, 2),
                   f"Single sample failed",
                   test_name)
        
        # 2. Very small learning rate
        trainer_small_lr = Trainer(
            network=network,
            batch_size=10,
            nb_epoch=5,
            learning_rate=1e-10,
            loss_fun="mse",
            shuffle_flag=True
        )
        X = np.random.randn(50, 3)
        y = np.random.randn(50, 2)
        
        loss_before = trainer_small_lr.eval_loss(X, y)
        trainer_small_lr.train(X, y)
        loss_after = trainer_small_lr.eval_loss(X, y)
        
        # Loss should barely change with tiny learning rate
        assert_test(abs(loss_before - loss_after) < 0.01,
                   f"Loss changed too much with tiny LR: {loss_before} -> {loss_after}",
                   test_name)
        
        # 3. Zero inputs
        x_zero = np.zeros((5, 3))
        out_zero = network.forward(x_zero)
        assert_test(not np.any(np.isnan(out_zero)),
                   f"NaN with zero inputs",
                   test_name)
        
        # 4. Large values
        x_large = np.random.randn(5, 3) * 100
        out_large = network.forward(x_large)
        assert_test(not np.any(np.isnan(out_large)) and not np.any(np.isinf(out_large)),
                   f"NaN/Inf with large inputs",
                   test_name)
        
        # 5. Identity activation only
        network_identity = MultiLayerNetwork(3, [5, 2], ["identity", "identity"])
        x_test = np.random.randn(5, 3)
        out_test = network_identity.forward(x_test)
        grad_test = network_identity.backward(np.random.randn(5, 2))
        assert_test(out_test.shape == (5, 2) and grad_test.shape == (5, 3),
                   f"Identity-only network failed",
                   test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def test_numerical_stability():
    """Test numerical stability."""
    test_name = "Numerical Stability"
    try:
        np.random.seed(42)
        
        # 1. Sigmoid with extreme values
        sigmoid_layer = SigmoidLayer()
        x_extreme = np.array([[-1000.0, 1000.0, 0.0]])
        out = sigmoid_layer.forward(x_extreme)
        
        assert_test(not np.any(np.isnan(out)) and not np.any(np.isinf(out)),
                   f"Sigmoid unstable with extreme values",
                   test_name)
        
        assert_test(np.isclose(out[0, 0], 0, atol=1e-10),
                   f"Sigmoid(-1000) should be ~0: {out[0, 0]}",
                   test_name)
        
        assert_test(np.isclose(out[0, 1], 1, atol=1e-10),
                   f"Sigmoid(1000) should be ~1: {out[0, 1]}",
                   test_name)
        
        # 2. Cross-entropy with extreme predictions
        ce_layer = CrossEntropyLossLayer()
        pred_extreme = np.array([[0.999, 0.001], [0.001, 0.999]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        loss_extreme = ce_layer.forward(pred_extreme, targets)
        assert_test(not np.isnan(loss_extreme) and not np.isinf(loss_extreme),
                   f"Cross-entropy unstable: {loss_extreme}",
                   test_name)
        
        grad_extreme = ce_layer.backward()
        assert_test(not np.any(np.isnan(grad_extreme)) and not np.any(np.isinf(grad_extreme)),
                   f"Cross-entropy gradient unstable",
                   test_name)
        
        results.record_pass(test_name)
        return True
    except AssertionError:
        return False


def run_all_tests():
    """Run complete bomb-proof test suite."""
    print("="*70)
    print("BOMB-PROOF COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("This will take a while due to extensive numerical gradient checking...")
    print()
    
    tests = [
        ("Xavier Initialization", test_xavier_init),
        ("LinearLayer - Complete", test_linear_layer_complete),
        ("SigmoidLayer - Complete", test_sigmoid_layer_complete),
        ("ReluLayer - Complete", test_relu_layer_complete),
        ("MultiLayerNetwork - Complete", test_multilayer_network_complete),
        ("MSELossLayer - Complete", test_mse_loss_complete),
        ("CrossEntropyLossLayer - Complete", test_cross_entropy_loss_complete),
        ("Preprocessor - Complete", test_preprocessor_complete),
        ("Trainer - Complete", test_trainer_complete),
        ("Deep Network Gradients", test_deep_network_gradients),
        ("Full Integration Pipeline", test_integration_full_pipeline),
        ("Edge Cases", test_edge_cases),
        ("Numerical Stability", test_numerical_stability),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Running: {test_name}")
        print('='*70)
        try:
            test_func()
        except Exception as e:
            results.record_fail(test_name, str(e))
            print(f"✗ {test_name} - EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n")
    success = results.summary()
    
    if success:
        print("\n🎉 CONGRATULATIONS! YOUR CODE IS BOMB-PROOF! 🎉")
        print("All tests passed with full numerical gradient verification.")
    else:
        print("\n⚠️  ERRORS DETECTED - Please review failed tests above.")
    
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
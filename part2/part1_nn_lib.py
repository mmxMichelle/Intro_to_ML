import numpy as np
import pickle


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    
    Arguments:
        size {tuple} -- size of the network to initialise.
        gain {float} -- gain for the Xavier initialisation.
    
    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor for the LinearLayer.

        Arguments:
            n_in {int} -- number of input neurons
            n_out {int} -- number of output neurons

        """
        # Initialize weights using Xavier initialization
        self._W = xavier_init((n_in, n_out))
        self._b = np.zeros((1, n_out))
        
        # Placeholders for gradients
        self._grad_W_current = None
        self._grad_b_current = None
        
        # Cache for backward pass
        self._cache_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Arguments:
            x {np.ndarray} -- input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- output array of shape (batch_size, n_out)
        """
        # Cache input for backward pass
        self._cache_current = x
        
        # Compute affine transformation: XW + b
        # Broadcasting handles adding bias to each sample in batch
        output = np.dot(x, self._W) + self._b
        
        return output

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- gradient of the loss with respect to the output of this layer.
                                   Shape: (batch_size, n_out)

        Returns:
            {np.ndarray} -- gradient of the loss with respect to the input of this layer.
                            Shape: (batch_size, n_in)
        """
        # Retrieve cached input
        x = self._cache_current
        
        # Gradient with respect to weights: X^T @ grad_z
        # Shape: (n_in, batch_size) @ (batch_size, n_out) = (n_in, n_out)
        self._grad_W_current = np.dot(x.T, grad_z)
        
        # Gradient with respect to bias: sum over batch dimension
        # Shape: (batch_size, n_out) -> (1, n_out)
        self._grad_b_current = np.sum(grad_z, axis=0, keepdims=True)
        
        # Gradient with respect to input: grad_z @ W^T
        # Shape: (batch_size, n_out) @ (n_out, n_in) = (batch_size, n_in)
        grad_x = np.dot(grad_z, self._W.T)
        
        return grad_x

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the layer's parameters using currently computed gradients.

        Arguments:
            learning_rate {float} -- learning rate of update step.
        """
        # Update weights: W = W - lr * grad_W
        self._W -= learning_rate * self._grad_W_current
        
        # Update bias: b = b - lr * grad_b
        self._b -= learning_rate * self._grad_b_current


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """
        Constructor for the SigmoidLayer.
        """
        self._cache_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns sigmoid(x)).

        Arguments:
            x {np.ndarray} -- input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- output array of shape (batch_size, n_in)
        """
        # Numerically stable sigmoid computation
        # For positive values: 1 / (1 + exp(-x))
        # For negative values: exp(x) / (1 + exp(x))
        output = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        
        # Cache output for backward pass
        self._cache_current = output
        
        return output

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- gradient of the loss with respect to the output of this layer.
                                   Shape: (batch_size, n_in)

        Returns:
            {np.ndarray} -- gradient of the loss with respect to the input of this layer.
                            Shape: (batch_size, n_in)
        """
        # Retrieve cached sigmoid output
        sigmoid_output = self._cache_current
        
        # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_grad = sigmoid_output * (1 - sigmoid_output)
        
        # Chain rule: grad_x = grad_z * sigmoid'(x)
        grad_x = grad_z * sigmoid_grad
        
        return grad_x


class ReluLayer(Layer):
    """
    ReluLayer: Applies ReLU function elementwise.
    """

    def __init__(self):
        """
        Constructor for the ReluLayer.
        """
        self._cache_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns max(0, x)).

        Arguments:
            x {np.ndarray} -- input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- output array of shape (batch_size, n_in)
        """
        # Cache input for backward pass
        self._cache_current = x
        
        # ReLU: max(0, x)
        output = np.maximum(0, x)
        
        return output

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- gradient of the loss with respect to the output of this layer.
                                   Shape: (batch_size, n_in)

        Returns:
            {np.ndarray} -- gradient of the loss with respect to the input of this layer.
                            Shape: (batch_size, n_in)
        """
        # Retrieve cached input
        x = self._cache_current
        
        # Derivative of ReLU: 1 if x > 0, else 0
        relu_grad = (x > 0).astype(float)
        
        # Chain rule: grad_x = grad_z * relu'(x)
        grad_x = grad_z * relu_grad
        
        return grad_x


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor for the MultiLayerNetwork.

        Arguments:
            input_dim {int} -- number of dimensions of the input data
            neurons {list} -- number of neurons in each linear layer 
                              represented as a list. The length of the list determines the 
                              number of linear layers.
            activations {list} -- list of the activation functions to apply 
                                  to the output of each linear layer. Possible values: "relu", "sigmoid", "identity".
        """
        self._layers = []
        
        # Build the network layer by layer
        layer_input_dim = input_dim
        
        for i, (n_neurons, activation) in enumerate(zip(neurons, activations)):
            # Add linear layer
            self._layers.append(LinearLayer(layer_input_dim, n_neurons))
            
            # Add activation layer
            if activation == "relu":
                self._layers.append(ReluLayer())
            elif activation == "sigmoid":
                self._layers.append(SigmoidLayer())
            elif activation == "identity":
                # Identity activation - no additional layer needed
                pass
            else:
                raise ValueError(f"Unknown activation function: {activation}")
            
            # Update input dimension for next layer
            layer_input_dim = n_neurons

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- output of the network.
        """
        # Sequentially pass through all layers
        output = x
        for layer in self._layers:
            output = layer.forward(output)
        
        return output

    def __call__(self, x):
        """
        Allows using the network as a function.
        """
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- gradient of the loss with respect to 
                                   the output of the network.

        Returns:
            {np.ndarray} -- gradient of the loss with respect to the input of the network.
        """
        # Sequentially backpropagate through all layers in reverse order
        grad = grad_z
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        
        return grad

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the 
        parameters of all layers using currently computed gradients.

        Arguments:
            learning_rate {float} -- learning rate of update step.
        """
        # Update parameters of all layers
        for layer in self._layers:
            layer.update_params(learning_rate)


def save_network(network, fpath):
    """
    Utility function to pickle a `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load a network previously saved at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor for the Trainer.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- training batch size.
            nb_epoch {int} -- number of training epochs.
            learning_rate {float} -- learning rate to be used for training.
            loss_fun {str} -- loss function to be used. 
                              Possible values: "mse", "cross_entropy".
            shuffle_flag {bool} -- If True, training data is shuffled before every epoch.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.shuffle_flag = shuffle_flag
        
        # Initialize loss layer based on loss function
        if loss_fun == "mse":
            self._loss_layer = MSELossLayer()
        elif loss_fun == "cross_entropy":
            self._loss_layer = CrossEntropyLossLayer()
        else:
            raise ValueError(f"Unknown loss function: {loss_fun}")

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            input_dataset {np.ndarray} -- array of input features, of shape
                                          (#_data_points, n_features) or
                                          (#_data_points,).
            target_dataset {np.ndarray} -- array of corresponding targets, of
                                           shape (#_data_points, #output_neurons).

        Returns: 
            tuple -- shuffled inputs and targets, same shape as input arguments.
        """
        # Generate random permutation of indices
        indices = np.random.permutation(len(input_dataset))
        
        # Shuffle both datasets using the same permutation
        shuffled_inputs = input_dataset[indices]
        shuffled_targets = target_dataset[indices]
        
        return shuffled_inputs, shuffled_targets

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs training epochs, 
        shuffling the data if required, and updating the network parameters.

        Arguments:
            input_dataset {np.ndarray} -- array of input features, of shape
                                          (#_training_data_points, n_features).
            target_dataset {np.ndarray} -- array of corresponding targets, of
                                           shape (#_training_data_points, #output_neurons).
        """
        for epoch in range(self.nb_epoch):
            # Shuffle data if flag is set
            if self.shuffle_flag:
                input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)
            
            # Get number of samples
            n_samples = len(input_dataset)
            
            # Iterate over minibatches
            for i in range(0, n_samples, self.batch_size):
                # Get minibatch
                batch_end = min(i + self.batch_size, n_samples)
                input_batch = input_dataset[i:batch_end]
                target_batch = target_dataset[i:batch_end]
                
                # Forward pass
                predictions = self.network.forward(input_batch)
                
                # Compute loss (not used for training, but loss_layer caches values)
                _ = self._loss_layer.forward(predictions, target_batch)
                
                # Backward pass through loss layer
                grad_loss = self._loss_layer.backward()
                
                # Backward pass through network
                self.network.backward(grad_loss)
                
                # Update parameters
                self.network.update_params(self.learning_rate)

    def eval_loss(self, input_dataset, target_dataset):
        """
        Evaluates the loss function on the given dataset.

        Arguments:
            input_dataset {np.ndarray} -- array of input features, of shape
                                          (#_evaluation_data_points, n_features).
            target_dataset {np.ndarray} -- array of corresponding targets, of
                                           shape (#_evaluation_data_points, #output_neurons).

        Returns:
            float -- loss value.
        """
        # Forward pass
        predictions = self.network.forward(input_dataset)
        
        # Compute loss
        loss = self._loss_layer.forward(predictions, target_dataset)
        
        return loss


class Preprocessor(object):
    """
    Preprocessor: Performs min-max scaling to [0, 1] range.
    """

    def __init__(self, data):
        """
        Constructor for the Preprocessor.
        Computes and stores the normalization parameters based on the provided dataset.

        Arguments:
            data {np.ndarray} -- data to be normalized, of shape
                                 (#_data_points, n_features).
        """
        # Compute min and max for each feature
        self._min = np.min(data, axis=0)
        self._max = np.max(data, axis=0)
        
        # Compute range for each feature
        self._range = self._max - self._min
        
        # Handle case where range is zero (constant feature)
        # Set range to 1 to avoid division by zero
        self._range[self._range == 0] = 1

    def apply(self, data):
        """
        Applies the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} -- data to be normalized, of shape
                                 (#_data_points, n_features).

        Returns:
            {np.ndarray} -- normalized data, same shape as input.
        """
        # Apply min-max scaling: (x - min) / (max - min)
        normalized = (data - self._min) / self._range
        
        return normalized

    def revert(self, data):
        """
        Reverts the pre-processing operations applied to the provided dataset.

        Arguments:
            data {np.ndarray} -- data to be reverted, of shape
                                 (#_data_points, n_features).

        Returns:
            {np.ndarray} -- reverted data, same shape as input.
        """
        # Revert min-max scaling: x_original = x_normalized * (max - min) + min
        reverted = data * self._range + self._min
        
        return reverted


def example_main():
    """
    Example usage of the neural network library.
    """
    # Load iris dataset
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    # Prepare data
    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    # Preprocess data
    prep_input = Preprocessor(x)
    x_preprocessed = prep_input.apply(x)

    # Split data
    split_idx = int(0.8 * len(x))

    x_train = x_preprocessed[:split_idx]
    y_train = y[:split_idx]
    x_val = x_preprocessed[split_idx:]
    y_val = y[split_idx:]

    # Train network
    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True,
    )

    trainer.train(x_train, y_train)
    print("Validation loss: {}".format(trainer.eval_loss(x_val, y_val)))


class MSELossLayer(object):
    """
    MSELossLayer: Computes mean-squared error loss.
    """

    def __init__(self):
        """
        Constructor for the MSELossLayer.
        """
        self._cache_current = None

    def forward(self, predictions, targets):
        """
        Performs forward pass through the loss layer.

        Arguments:
            predictions {np.ndarray} -- predictions of the network, of shape
                                        (batch_size, #output_neurons).
            targets {np.ndarray} -- ground truth labels, of shape
                                   (batch_size, #output_neurons).

        Returns:
            float -- mean squared error loss value.
        """
        # Cache predictions and targets for backward pass
        self._cache_current = (predictions, targets)
        
        # Compute MSE: (1/n) * sum((predictions - targets)^2)
        loss = np.mean((predictions - targets) ** 2)
        
        return loss

    def backward(self):
        """
        Performs backward pass through the loss layer.

        Returns:
            {np.ndarray} -- gradient of the loss with respect to predictions, 
                            of shape (batch_size, #output_neurons).
        """
        # Retrieve cached values
        predictions, targets = self._cache_current
        
        # Get batch size
        batch_size = predictions.shape[0]
        
        # Gradient of MSE: (2/n) * (predictions - targets)
        grad = (2.0 / batch_size) * (predictions - targets)
        
        return grad


class CrossEntropyLossLayer(object):
    """
    CrossEntropyLossLayer: Computes cross-entropy loss (for binary classification).
    """

    def __init__(self):
        """
        Constructor for the CrossEntropyLossLayer.
        """
        self._cache_current = None

    def forward(self, predictions, targets):
        """
        Performs forward pass through the loss layer.

        Arguments:
            predictions {np.ndarray} -- predictions of the network, of shape
                                        (batch_size, #output_neurons).
            targets {np.ndarray} -- ground truth labels, of shape
                                   (batch_size, #output_neurons).

        Returns:
            float -- cross entropy loss value.
        """
        # Cache predictions and targets for backward pass
        self._cache_current = (predictions, targets)
        
        # Clip predictions to avoid log(0)
        epsilon = 1e-15
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Compute cross-entropy: -mean(targets * log(predictions) + (1 - targets) * log(1 - predictions))
        loss = -np.mean(
            targets * np.log(predictions_clipped) + 
            (1 - targets) * np.log(1 - predictions_clipped)
        )
        
        return loss

    def backward(self):
        """
        Performs backward pass through the loss layer.

        Returns:
            {np.ndarray} -- gradient of the loss with respect to predictions,
                            of shape (batch_size, #output_neurons).
        """
        # Retrieve cached values
        predictions, targets = self._cache_current
        
        # Get batch size
        batch_size = predictions.shape[0]
        
        # Clip predictions to avoid division by zero
        epsilon = 1e-15
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Gradient of cross-entropy: (1/n) * (predictions - targets) / (predictions * (1 - predictions))
        grad = (predictions_clipped - targets) / (predictions_clipped * (1 - predictions_clipped))
        grad = grad / batch_size
        
        return grad


if __name__ == "__main__":
    example_main()

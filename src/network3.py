"""network3.py
~~~~~~~~~~~~~~

Convolutional neural network using PyTorch.  Supports conv+pool,
fully connected, and softmax layers.  Runs on CPU, CUDA, or Apple
Silicon (MPS).  API mirrors network2.py so the book exercises work
unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pick the best available hardware: GPU > Apple Silicon > CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Activation functions -- sigmoid is the default (same as the book),
# ReLU is used in Chapter 6 to avoid the vanishing gradient problem.
def linear(z):
    return z


def ReLU(z):
    return torch.clamp(z, min=0.0)


from torch import sigmoid, tanh  # noqa: F401 -- tanh available for experimentation


def load_data_shared(filename=None):
    """Load MNIST data and return as PyTorch tensors on the active device."""
    if filename is not None:
        import gzip
        import pickle

        with gzip.open(filename, "rb") as f:
            tr_d, va_d, te_d = pickle.load(f, encoding="latin1")
    else:
        try:
            from .mnist_loader import load_data
        except ImportError:
            from mnist_loader import load_data
        tr_d, va_d, te_d = load_data()

    def to_tensors(data):
        x = torch.tensor(data[0], dtype=torch.float32, device=device)
        y = torch.tensor(data[1], dtype=torch.long, device=device)
        return x, y

    return [to_tensors(tr_d), to_tensors(va_d), to_tensors(te_d)]


class ConvPoolLayer(nn.Module):
    """Convolutional layer followed by max-pooling.

    This is the building block of a CNN: it slides small filters across
    the image to detect local features (edges, corners, textures), then
    pools to reduce spatial size and gain translation invariance.
    """

    def __init__(
        self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid
    ):
        super().__init__()
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # Conv2d learns filter_shape[0] filters, each looking at a small
        # (filter_h x filter_w) patch of the input feature maps
        self.conv = nn.Conv2d(
            filter_shape[1],
            filter_shape[0],
            kernel_size=(filter_shape[2], filter_shape[3]),
        )
        # Max-pooling keeps only the strongest activation in each pool region
        self.pool = nn.MaxPool2d(kernel_size=poolsize)
        # Initialize weights with small random values scaled by 1/sqrt(n_out)
        # so that activations don't explode or vanish in early training
        n_out = filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize)
        nn.init.normal_(self.conv.weight, mean=0, std=np.sqrt(1.0 / n_out))
        nn.init.normal_(self.conv.bias, mean=0, std=1.0)

    def forward(self, x):
        # conv -> pool -> activate
        return self.activation_fn(self.pool(self.conv(x)))


class FullyConnectedLayer(nn.Module):
    """Standard fully-connected layer with optional dropout.

    Every neuron connects to every neuron in the previous layer.
    Dropout randomly disables neurons during training to prevent
    the network from relying too heavily on any single neuron.
    """

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=p_dropout)
        nn.init.normal_(self.linear.weight, mean=0, std=np.sqrt(1.0 / n_out))
        nn.init.normal_(self.linear.bias, mean=0, std=1.0)

    def forward(self, x):
        # Dropout is applied to the input (not output) -- this zeros out
        # random input features, forcing the layer to learn redundant
        # representations rather than memorizing specific input patterns
        return self.activation_fn(self.linear(self.dropout(x)))


class SoftmaxLayer(nn.Module):
    """Output layer that converts raw scores into class probabilities.

    The softmax function ensures outputs sum to 1, so each output
    can be interpreted as "probability this digit is a 0, 1, ..., 9".
    """

    def __init__(self, n_in, n_out, p_dropout=0.0):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=p_dropout)
        # Softmax layer starts with zero weights -- the network learns
        # to assign meaning to each output neuron during training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # log_softmax is used instead of softmax for numerical stability
        # (avoids overflow when computing log-probabilities for the loss)
        return F.log_softmax(self.linear(self.dropout(x)), dim=1)


class Network(nn.Module):
    """A network built from a sequence of layers (conv, FC, softmax).

    The layers are stacked in order: typically one or more ConvPoolLayers
    (to extract features from images), followed by FullyConnectedLayers
    (to reason about those features), ending with a SoftmaxLayer
    (to produce a classification).
    """

    def __init__(self, layers, mini_batch_size):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.mini_batch_size = mini_batch_size
        self.to(device)

    def forward(self, x):
        # Input is flat (784 values) -- reshape to a 28x28 image
        out = x.view(-1, 1, 28, 28)
        for layer in self.layers:
            # Flatten back to 1D before fully-connected layers
            if isinstance(layer, (FullyConnectedLayer, SoftmaxLayer)):
                out = out.view(out.size(0), -1)
            out = layer(out)
        return out

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        validation_data,
        test_data,
        lmbda=0.0,
    ):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        num_training_batches = len(training_x) // mini_batch_size
        num_validation_batches = len(validation_x) // mini_batch_size
        num_test_batches = len(test_x) // mini_batch_size
        # L2 regularization: only penalize weights, not biases (matches
        # the book's formulation where the regularization term is
        # (lmbda/2n) * sum(w^2) over weights only)
        weight_params = [p for n, p in self.named_parameters() if "weight" in n]
        bias_params = [p for n, p in self.named_parameters() if "bias" in n]
        optimizer = torch.optim.SGD(
            [
                {"params": weight_params, "weight_decay": lmbda / num_training_batches},
                {"params": bias_params, "weight_decay": 0.0},
            ],
            lr=eta,
        )
        best_validation_accuracy = 0.0
        best_iteration = 0
        test_accuracy = 0.0
        for epoch in range(epochs):
            self.train()  # enable dropout
            # Shuffle the training data each epoch so mini-batches vary
            perm = torch.randperm(len(training_x), device=device)
            training_x = training_x[perm]
            training_y = training_y[perm]
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                # Grab one mini-batch of images and labels
                start = minibatch_index * mini_batch_size
                end = start + mini_batch_size
                x_batch = training_x[start:end]
                y_batch = training_y[start:end]
                # Forward: compute predictions and measure how wrong they are
                output = self(x_batch)
                loss = F.nll_loss(output, y_batch)
                # Backward: compute gradients, then update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # At the end of each epoch, check accuracy on held-out data
                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = self._evaluate(
                        validation_x,
                        validation_y,
                        num_validation_batches,
                        mini_batch_size,
                    )
                    print(
                        "Epoch {0}: validation accuracy {1:.2%}".format(
                            epoch, validation_accuracy
                        )
                    )
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = self._evaluate(
                                test_x, test_y, num_test_batches, mini_batch_size
                            )
                            print(
                                "The corresponding test accuracy is {0:.2%}".format(
                                    test_accuracy
                                )
                            )
        print("Finished training network.")
        print(
            "Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
                best_validation_accuracy, best_iteration
            )
        )
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

    def _evaluate(self, x, y, num_batches, mini_batch_size):
        """Evaluate accuracy on a dataset (validation or test)."""
        self.eval()  # disable dropout for evaluation
        correct = 0
        total = 0
        with torch.no_grad():  # no need to track gradients during evaluation
            for j in range(num_batches):
                start = j * mini_batch_size
                end = start + mini_batch_size
                output = self(x[start:end])
                predictions = output.argmax(dim=1)
                correct += (predictions == y[start:end]).sum().item()
                total += mini_batch_size
        self.train()  # re-enable dropout for training
        return correct / total


def size(data):
    return len(data[0])

"""Smoke tests for the neural network modules.

These tests verify basic functionality: data loading, training for
a single epoch, and save/load round-trips. They are meant to catch
import errors and obvious regressions, not to validate full accuracy.
"""

import os
import tempfile

import numpy as np


def test_mnist_loader():
    """Verify MNIST data loads correctly (50k/10k/10k split)."""
    from src import mnist_loader

    tr_d, va_d, te_d = mnist_loader.load_data()
    assert len(tr_d[0]) == 50000
    assert len(va_d[0]) == 10000
    assert len(te_d[0]) == 10000

    tr_w, va_w, te_w = mnist_loader.load_data_wrapper()
    assert len(tr_w) == 50000
    assert len(va_w) == 10000
    assert len(te_w) == 10000

    # Training data should be (784,1) inputs with (10,1) one-hot labels
    x, y = tr_w[0]
    assert x.shape == (784, 1)
    assert y.shape == (10, 1)

    # Validation/test data should have integer labels
    x, y = va_w[0]
    assert x.shape == (784, 1)
    assert isinstance(int(y), int)


def test_network_basic():
    """Train network.py [784, 30, 10] for 1 epoch, verify >80% accuracy."""
    from src import mnist_loader, network

    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 1, 10, 3.0, test_data=test_data)

    # Evaluate: count correct predictions on test set
    correct = sum(int(np.argmax(net.feedforward(x)) == y) for x, y in test_data)
    accuracy = correct / len(test_data)
    assert accuracy > 0.80, f"Expected >80% accuracy, got {accuracy:.1%}"


def test_network2_cross_entropy():
    """Train network2.py with cross-entropy for 1 epoch."""
    from src import mnist_loader
    from src.network2 import Network, CrossEntropyCost

    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], cost=CrossEntropyCost)
    ec, ea, tc, ta = net.SGD(
        training_data,
        1,
        10,
        0.5,
        lmbda=5.0,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
    )
    # Should return one accuracy measurement for the single epoch
    assert len(ea) == 1
    accuracy = ea[0] / len(test_data)
    assert accuracy > 0.85, f"Expected >85% accuracy, got {accuracy:.1%}"


def test_network2_save_load():
    """Verify save/load round-trip works."""
    from src import mnist_loader
    from src.network2 import Network, CrossEntropyCost, load

    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], cost=CrossEntropyCost)
    net.SGD(training_data, 1, 10, 0.5, lmbda=5.0)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmppath = f.name
    try:
        net.save(tmppath)
        net2 = load(tmppath)

        assert net2.sizes == net.sizes
        assert net2.cost.__name__ == "CrossEntropyCost"
        for w1, w2 in zip(net.weights, net2.weights):
            np.testing.assert_array_equal(w1, w2)
        for b1, b2 in zip(net.biases, net2.biases):
            np.testing.assert_array_equal(b1, b2)
    finally:
        os.unlink(tmppath)


def test_network2_momentum():
    """Train network2.py with momentum (mu=0.9) for 1 epoch."""
    from src import mnist_loader
    from src.network2 import Network, CrossEntropyCost

    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], cost=CrossEntropyCost)
    ec, ea, tc, ta = net.SGD(
        training_data,
        1,
        10,
        0.1,
        lmbda=5.0,
        mu=0.9,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
    )
    assert len(ea) == 1
    accuracy = ea[0] / len(test_data)
    assert accuracy > 0.80, f"Expected >80% accuracy with momentum, got {accuracy:.1%}"


def test_network2_early_stopping():
    """Verify early stopping halts training before max epochs."""
    from src import mnist_loader
    from src.network2 import Network, CrossEntropyCost

    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], cost=CrossEntropyCost)
    ec, ea, tc, ta = net.SGD(
        training_data,
        100,
        10,
        0.5,
        lmbda=5.0,
        early_stopping_n=3,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
    )
    # Should stop well before 100 epochs
    assert len(ea) < 100, f"Expected early stop before 100 epochs, ran {len(ea)}"
    assert len(ea) >= 3, f"Should run at least early_stopping_n epochs, ran {len(ea)}"


def test_network2_learning_rate_schedule():
    """Verify learning rate schedule runs and terminates."""
    from src import mnist_loader
    from src.network2 import Network, CrossEntropyCost

    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], cost=CrossEntropyCost)
    ec, ea, tc, ta = net.SGD(
        training_data,
        200,
        10,
        0.5,
        lmbda=5.0,
        schedule_eta_n=3,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
    )
    # Should terminate when eta drops to 1/128 of original, well before 200
    assert len(ea) < 200, f"Expected schedule to terminate early, ran {len(ea)} epochs"
    assert len(ea) >= 3, f"Should run at least schedule_eta_n epochs, ran {len(ea)}"


def test_network2_l1_regularization():
    """Train network2.py with L1 regularization for 1 epoch."""
    from src import mnist_loader
    from src.network2 import Network, CrossEntropyCost

    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], cost=CrossEntropyCost, regularization="l1")
    ec, ea, tc, ta = net.SGD(
        training_data,
        1,
        10,
        0.5,
        lmbda=5.0,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
    )
    assert len(ea) == 1
    accuracy = ea[0] / len(test_data)
    assert accuracy > 0.80, f"Expected >80% accuracy with L1, got {accuracy:.1%}"


def test_network2_save_load_l1():
    """Verify save/load round-trip preserves L1 regularization type."""
    from src import mnist_loader
    from src.network2 import Network, CrossEntropyCost, load

    training_data, _, _ = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10], cost=CrossEntropyCost, regularization="l1")
    net.SGD(training_data, 1, 10, 0.5, lmbda=5.0)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmppath = f.name
    try:
        net.save(tmppath)
        net2 = load(tmppath)

        assert net2.regularization == "l1", (
            f"Expected regularization='l1', got '{net2.regularization}'"
        )
        assert net2.sizes == net.sizes
        for w1, w2 in zip(net.weights, net2.weights):
            np.testing.assert_array_equal(w1, w2)
    finally:
        os.unlink(tmppath)


def test_network3_conv():
    """Train network3.py basic conv net for 1 epoch on small subset."""
    import torch
    from src.network3 import (
        Network,
        ConvPoolLayer,
        FullyConnectedLayer,
        SoftmaxLayer,
        load_data_shared,
        device,
    )

    training_data, validation_data, test_data = load_data_shared()

    # Use a small subset for speed: first 1000 training, 200 val/test
    tr = (training_data[0][:1000], training_data[1][:1000])
    va = (validation_data[0][:200], validation_data[1][:200])
    te = (test_data[0][:200], test_data[1][:200])

    mini_batch_size = 10
    net = Network(
        [
            ConvPoolLayer(
                image_shape=(mini_batch_size, 1, 28, 28),
                filter_shape=(5, 1, 5, 5),
                poolsize=(2, 2),
            ),
            FullyConnectedLayer(n_in=5 * 12 * 12, n_out=30),
            SoftmaxLayer(n_in=30, n_out=10),
        ],
        mini_batch_size,
    )
    net.SGD(tr, 1, mini_batch_size, 0.1, va, te)

    # Verify it can do inference
    net.eval()
    with torch.no_grad():
        sample = te[0][:mini_batch_size].to(device)
        output = net(sample)
        assert output.shape == (mini_batch_size, 10)

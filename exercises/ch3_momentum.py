"""Chapter 3 -- Momentum-based gradient descent.

Standard SGD updates weights based only on the current gradient.
Momentum accumulates a "velocity" from past gradients -- helping the
network move faster through flat regions and dampen oscillations in
steep ones.

  v = mu * v - eta * gradient
  w = w + v

Run:  uv run python exercises/ch3_momentum.py

What to watch: with momentum (mu=0.9), the network often converges
faster in early epochs compared to standard SGD (mu=0).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

print("=== Standard SGD (no momentum) ===")
net_no_mu = net2.Network([784, 30, 10], cost=net2.CrossEntropyCost)
net_no_mu.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.5,
    lmbda=5.0,
    mu=0.0,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
)

print("\n=== Momentum SGD (mu=0.9) ===")
net_mu = net2.Network([784, 30, 10], cost=net2.CrossEntropyCost)
net_mu.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.1,
    lmbda=5.0,
    mu=0.9,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
)

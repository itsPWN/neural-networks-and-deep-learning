"""Chapter 3 -- L1 vs L2 regularization.

L1 regularization penalizes the sum of absolute weights (driving many
toward exactly zero -- sparsity), while L2 penalizes the sum of squared
weights (driving them toward small but non-zero values).

Run:  uv run python exercises/ch3_l1_regularization.py

What to watch: both reach similar accuracy, but L1 produces sparser
weights (more weights near zero).  After training, the script prints
the fraction of weights close to zero for each approach.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

print("=== L2 regularization (default) ===")
net_l2 = net2.Network([784, 30, 10], cost=net2.CrossEntropyCost, regularization="l2")
net_l2.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.5,
    lmbda=5.0,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
)

print("\n=== L1 regularization ===")
net_l1 = net2.Network([784, 30, 10], cost=net2.CrossEntropyCost, regularization="l1")
net_l1.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.5,
    lmbda=5.0,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
)

# Compare weight sparsity
threshold = 0.01
for name, network in [("L2", net_l2), ("L1", net_l1)]:
    total = sum(w.size for w in network.weights)
    near_zero = sum(np.sum(np.abs(w) < threshold) for w in network.weights)
    print(f"\n{name}: {near_zero}/{total} weights near zero ({near_zero / total:.1%})")

"""Chapter 3 -- Weight initialization comparison.

Compares two strategies:
  1. Default (good): weights ~ N(0, 1/sqrt(n_inputs))
  2. Old (large):    weights ~ N(0, 1)

With large initial weights, neurons saturate (output stuck near 0 or 1)
and gradients nearly vanish, so learning is very slow.

Run:  uv run python exercises/ch3_weight_init.py

What to watch: the good initialization reaches high accuracy much
faster in early epochs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

print("=== Good initialization (1/sqrt(n)) ===")
net_good = net2.Network([784, 30, 10])
net_good.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.1,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
)

print("\n=== Old initialization (large weights) ===")
net_old = net2.Network([784, 30, 10])
net_old.large_weight_initializer()
net_old.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.1,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
)

"""Chapter 3 -- Quadratic cost (for comparison with cross-entropy).

Same setup as ch3_cross_entropy.py but using the old quadratic cost.
Compare the results to see why cross-entropy is preferred.

Run:  uv run python exercises/ch3_quadratic_cost.py

Expected result: ~95.42% -- similar to Chapter 1.  The book's point is
that cross-entropy learns faster in early epochs; quadratic cost
produces tiny gradients when the network is confidently wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

network = net2.Network([784, 30, 10], cost=net2.QuadraticCost)
network.large_weight_initializer()
network.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.5,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
)

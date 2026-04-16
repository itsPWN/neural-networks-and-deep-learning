"""Chapter 1 -- Network with no hidden layer.

The book asks: what happens with just [784, 10]?  Without a hidden
layer the network can only learn (approximately) linear decision
boundaries -- not enough for complex digit shapes.

Run:  uv run python exercises/ch1_no_hidden_layer.py

Expected result: ~88-92% accuracy -- significantly worse than the 95%
with a 30-neuron hidden layer, proving that hidden layers are essential
for learning complex patterns.

Compare with ch1_basic_network.py which uses [784, 30, 10].
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network as net

training_data, validation_data, test_data = ml.load_data_wrapper()

print("=== No hidden layer: [784, 10] ===")
network = net.Network([784, 10])
network.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

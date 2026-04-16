"""Chapter 6 -- Double conv with dropout (best single-network result).

Three improvements over ch6_basic_conv.py:
  1. Two conv layers instead of one (deeper = more abstract features)
  2. ReLU activation instead of sigmoid (avoids vanishing gradients)
  3. Dropout (p=0.5) randomly disables neurons during training,
     preventing the network from memorizing the training data

Before running, generate expanded data (once, takes ~5 minutes):

    uv run python src/expand_mnist.py

Run:  uv run python exercises/ch6_double_conv.py

Expected result: ~99.6% accuracy -- only ~40 errors out of 10,000!

This is the best single-network result from the book.  Training takes
~15-30 minutes depending on hardware.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network3 import *

expanded_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "mnist_expanded.pkl.gz"
)

if not os.path.exists(expanded_path):
    print("Expanded data not found. Generate it first with:")
    print("    uv run python src/expand_mnist.py")
    print("This takes a few minutes and ~500MB RAM.")
    sys.exit(1)

print("Loading expanded training data (250,000 images)...")
training_data, validation_data, test_data = load_data_shared(expanded_path)

network = Network(
    [
        # First conv layer: 20 filters detect simple features (edges, corners)
        ConvPoolLayer(
            image_shape=(10, 1, 28, 28),
            filter_shape=(20, 1, 5, 5),
            poolsize=(2, 2),
            activation_fn=ReLU,
        ),
        # Second conv layer: 40 filters combine simple features into complex ones
        ConvPoolLayer(
            image_shape=(10, 20, 12, 12),
            filter_shape=(40, 20, 5, 5),
            poolsize=(2, 2),
            activation_fn=ReLU,
        ),
        # Two large FC layers with dropout for regularization
        FullyConnectedLayer(
            n_in=40 * 4 * 4, n_out=1000, activation_fn=ReLU, p_dropout=0.5
        ),
        FullyConnectedLayer(n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        # Output layer
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5),
    ],
    mini_batch_size=10,
)

network.SGD(
    training_data,
    epochs=40,
    mini_batch_size=10,
    eta=0.03,
    validation_data=validation_data,
    test_data=test_data,
)

"""Chapter 6 -- Training with expanded data (250,000 images).

The original MNIST has 50,000 training images.  expand_mnist.py creates
250,000 by shifting each image 1 pixel in all 4 directions.  This
teaches the network translation invariance -- a digit shifted slightly
is still the same digit.

Before running this exercise, generate the expanded data (once):

    uv run python src/expand_mnist.py

Then run:

    uv run python exercises/ch6_expanded_data.py

Expected result: ~99.3%+ accuracy -- an improvement over the basic
double conv network trained on standard data.
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
        ConvPoolLayer(
            image_shape=(10, 1, 28, 28),
            filter_shape=(20, 1, 5, 5),
            poolsize=(2, 2),
            activation_fn=ReLU,
        ),
        ConvPoolLayer(
            image_shape=(10, 20, 12, 12),
            filter_shape=(40, 20, 5, 5),
            poolsize=(2, 2),
            activation_fn=ReLU,
        ),
        FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10),
    ],
    mini_batch_size=10,
)

network.SGD(
    training_data,
    epochs=60,
    mini_batch_size=10,
    eta=0.03,
    validation_data=validation_data,
    test_data=test_data,
    lmbda=0.1,
)

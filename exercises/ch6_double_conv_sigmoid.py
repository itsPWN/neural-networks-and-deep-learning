"""Chapter 6 -- Double convolutional network (sigmoid activation).

An intermediate step in the book's progression: two convolutional
layers with the default sigmoid activation.  This shows the accuracy
gain from stacking conv layers, before adding ReLU or dropout.

Run:  uv run python exercises/ch6_double_conv_sigmoid.py

Expected result: ~99.06% accuracy.

Compare with:
  - ch6_basic_conv.py (~98.78%) to see the gain from a second conv layer
  - ch6_double_conv_relu.py (~99.23%) to see the further gain from ReLU
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network3 import *

training_data, validation_data, test_data = load_data_shared()

network = Network(
    [
        ConvPoolLayer(
            image_shape=(10, 1, 28, 28),
            filter_shape=(20, 1, 5, 5),
            poolsize=(2, 2),
        ),
        ConvPoolLayer(
            image_shape=(10, 20, 12, 12),
            filter_shape=(40, 20, 5, 5),
            poolsize=(2, 2),
        ),
        FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10),
    ],
    mini_batch_size=10,
)

network.SGD(
    training_data,
    epochs=60,
    mini_batch_size=10,
    eta=0.1,
    validation_data=validation_data,
    test_data=test_data,
)

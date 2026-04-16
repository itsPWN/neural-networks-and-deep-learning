"""Chapter 6 -- Double convolutional network with ReLU activation.

Switching from sigmoid to ReLU avoids the vanishing gradient problem
from Chapter 5, allowing the network to train deeper layers effectively.

Run:  uv run python exercises/ch6_double_conv_relu.py

Expected result: ~99.23% accuracy.

Compare with:
  - ch6_double_conv_sigmoid.py (~99.06%) to see the gain from ReLU
  - ch6_expanded_data.py (~99.3%+) to see the gain from more data
  - ch6_double_conv.py (~99.5%+) to see the gain from dropout
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

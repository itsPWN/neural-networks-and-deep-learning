"""Chapter 6 -- Exercise: Conv + Softmax only (no fully-connected layer).

The book asks: what accuracy do you get if you omit the fully-connected
layer and use only a convolutional-pooling layer feeding directly into
the softmax output?  Does the FC layer help?

Run:  uv run python exercises/ch6_conv_softmax_only.py

Expected result: lower than ch6_basic_conv.py (~98.78%), demonstrating
that the FC layer adds value by combining the features the conv layer
detected into higher-level reasoning about digit identity.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network3 import *

training_data, validation_data, test_data = load_data_shared()

# Conv layer outputs 20 * 12 * 12 = 2880 features after pooling.
# Without an FC layer, these feed directly into softmax.
network = Network(
    [
        ConvPoolLayer(
            image_shape=(10, 1, 28, 28),
            filter_shape=(20, 1, 5, 5),
            poolsize=(2, 2),
        ),
        SoftmaxLayer(n_in=20 * 12 * 12, n_out=10),
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

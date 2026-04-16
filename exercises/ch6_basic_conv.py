"""Chapter 6 -- Basic convolutional network.

Instead of treating each pixel independently (784 separate inputs),
a convolutional layer slides a small 5x5 filter across the image to
detect local patterns (edges, curves).  This is much more natural for
image recognition.

Run:  uv run python exercises/ch6_basic_conv.py

Expected result: ~98.78% accuracy -- a big jump from Ch1's 95%
and Ch3's 96.5%.

Note: this uses PyTorch and will automatically use GPU if available
(CUDA or Apple Silicon MPS).  Training takes several minutes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network3 import *

training_data, validation_data, test_data = load_data_shared()

network = Network(
    [
        # 20 filters, each scanning 5x5 patches, then 2x2 max-pooling
        ConvPoolLayer(
            image_shape=(10, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2, 2)
        ),
        # Fully-connected layer to combine the features the conv layer found
        FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
        # Output: probability for each of the 10 digits
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

"""Chapter 6 -- Fully-connected baseline (no convolution).

A plain fully-connected network using network3.py (PyTorch) for
comparison with the convolutional architectures.  This treats each
pixel independently, without exploiting the spatial structure of images.

Run:  uv run python exercises/ch6_fc_baseline.py

Expected result: ~97.80% accuracy.  Compare with:
  - ch6_basic_conv.py (~98.78%) to see the improvement from convolution
  - ch1_basic_network.py (~95%) to see the improvement from PyTorch's
    optimizer and better training
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network3 import *

training_data, validation_data, test_data = load_data_shared()

network = Network(
    [
        FullyConnectedLayer(n_in=784, n_out=100),
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

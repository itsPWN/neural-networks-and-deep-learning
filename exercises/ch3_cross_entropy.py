"""Chapter 3 -- Cross-entropy cost function.

The key improvement over Chapter 1: cross-entropy doesn't suffer from
"learning slowdown" when the network is confidently wrong.  This uses
the same old weight initialization and no regularization so the only
change from Chapter 1 is the cost function.

Run:  uv run python exercises/ch3_cross_entropy.py

Expected result: ~95.49% accuracy -- close to Chapter 1's 95.42%, but
cross-entropy learns faster in early epochs (the real advantage shows
with more neurons and regularization, covered in later exercises).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

network = net2.Network([784, 30, 10], cost=net2.CrossEntropyCost)
network.large_weight_initializer()
network.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.5,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
)

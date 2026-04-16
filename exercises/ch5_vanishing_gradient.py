"""Chapter 5 -- The vanishing gradient problem.

Trains networks with 1, 2, 3, and 4 hidden layers to show that adding
more layers does NOT always help with sigmoid activation.  Gradients
shrink exponentially as they propagate backward, causing early layers
to learn very slowly.

Run:  uv run python exercises/ch5_vanishing_gradient.py

Expected results (from the book, using eta=0.1, lmbda=5.0):
  [784, 30, 10]              ~96.48%  (1 hidden layer)
  [784, 30, 30, 10]          ~96.90%  (2 hidden layers -- slight gain)
  [784, 30, 30, 30, 10]      ~96.57%  (3 hidden layers -- starts dropping)
  [784, 30, 30, 30, 30, 10]  ~96.53%  (4 hidden layers -- no benefit)

This is why Chapter 6 uses convolutional architectures and ReLU
activation to overcome the vanishing gradient problem.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

architectures = [
    [784, 30, 10],
    [784, 30, 30, 10],
    [784, 30, 30, 30, 10],
    [784, 30, 30, 30, 30, 10],
]

for sizes in architectures:
    n_hidden = len(sizes) - 2
    print(f"\n{'=' * 60}")
    print(f"=== {n_hidden} hidden layer(s): {sizes} ===")
    print(f"{'=' * 60}\n")
    network = net2.Network(sizes, cost=net2.CrossEntropyCost)
    network.SGD(
        training_data,
        epochs=30,
        mini_batch_size=10,
        eta=0.1,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
    )

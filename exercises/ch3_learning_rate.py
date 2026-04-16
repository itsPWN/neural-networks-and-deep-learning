"""Chapter 3 -- Learning rate experiments.

The learning rate (eta) controls how big each weight update is.
This script tries three values to show the effect:

  - eta=0.025: Too small -- the network barely learns
  - eta=0.25:  Just right -- steady, reliable improvement
  - eta=2.5:   Too large -- cost oscillates wildly, network fails to learn

Run:  uv run python exercises/ch3_learning_rate.py

This is one of the most important practical lessons: the learning rate
is the single most important hyper-parameter to get right.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

for eta in [0.025, 0.25, 2.5]:
    print(f"\n=== Learning rate eta = {eta} ===")
    network = net2.Network([784, 30, 10])
    network.SGD(
        training_data,
        epochs=30,
        mini_batch_size=10,
        eta=eta,
        lmbda=5.0,
        evaluation_data=test_data,
        monitor_training_cost=True,
    )

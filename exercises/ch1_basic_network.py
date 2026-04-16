"""Chapter 1 -- Train your first neural network.

A 3-layer network with 784 inputs (one per pixel), 30 hidden neurons,
and 10 outputs (one per digit).  Trains for 30 epochs on 50,000 MNIST
images and tests on 10,000.

Run:  uv run python exercises/ch1_basic_network.py

What to watch:
  - Accuracy jumps fast in early epochs, then improves more slowly
  - Final accuracy should reach ~95% (about 500 errors out of 10,000)

Try changing:
  - The hidden layer size: replace 30 with 100 or 10
  - The learning rate: replace 3.0 with 0.1 or 10.0
  - The number of epochs: replace 30 with 5 or 60
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network as net

training_data, validation_data, test_data = ml.load_data_wrapper()

network = net.Network([784, 30, 10])
network.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

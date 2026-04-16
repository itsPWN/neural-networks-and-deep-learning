"""Chapter 3 -- Early stopping.

Instead of training for a fixed number of epochs, stop when validation
accuracy hasn't improved for N consecutive epochs.  This prevents
overfitting by halting training at the optimal point.

Run:  uv run python exercises/ch3_early_stopping.py

What to watch: training stops before reaching the full 100 epochs,
typically around epoch 20-30, at the point where further training
would only lead to overfitting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

print("=== Early stopping: halt when no improvement for 10 epochs ===\n")
network = net2.Network([784, 30, 10], cost=net2.CrossEntropyCost)
network.SGD(
    training_data,
    epochs=100,
    mini_batch_size=10,
    eta=0.5,
    lmbda=5.0,
    early_stopping_n=10,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
)

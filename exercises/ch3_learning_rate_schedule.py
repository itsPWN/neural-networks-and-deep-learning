"""Chapter 3 -- Learning rate schedule.

Instead of using a fixed learning rate, this exercise implements an
adaptive schedule: halve eta each time validation accuracy stalls for
10 epochs.  Training terminates when eta drops to 1/128 of its
original value.

This combines the best of both worlds: start with a large learning
rate for fast progress, then slow down for fine-tuning.

Run:  uv run python exercises/ch3_learning_rate_schedule.py

What to watch: the learning rate halves several times as training
progresses, and the network converges to a slightly better final
accuracy than with a fixed learning rate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

print("=== Learning rate schedule: halve eta on no-improvement-in-10 ===")
print("Starting eta=0.5, will terminate at eta=0.5/128\n")
network = net2.Network([784, 30, 10], cost=net2.CrossEntropyCost)
network.SGD(
    training_data,
    epochs=200,
    mini_batch_size=10,
    eta=0.5,
    lmbda=5.0,
    schedule_eta_n=10,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
)

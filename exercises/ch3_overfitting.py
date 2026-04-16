"""Chapter 3 -- See overfitting in action.

Trains on only 1,000 images (instead of 50,000) with NO regularization.
The network memorizes the small training set but fails to generalize
to unseen test images.

Run:  uv run python exercises/ch3_overfitting.py

What to watch:
  - Training accuracy climbs toward 100% (memorization)
  - Test accuracy plateaus around 82-84% and may decrease
  - The gap between training and test accuracy IS overfitting
  - This is why Chapter 3 introduces regularization (lmbda > 0)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_loader as ml
import src.network2 as net2

training_data, validation_data, test_data = ml.load_data_wrapper()

# Use only 1,000 training images (2% of the full set) and no regularization
small_training_data = training_data[:1000]

print("=== Overfitting demo: 1,000 images, no regularization ===")
print("Watch training accuracy approach 100% while test accuracy stalls.\n")
network = net2.Network([784, 30, 10], cost=net2.CrossEntropyCost)
network.large_weight_initializer()
network.SGD(
    small_training_data,
    epochs=400,
    mini_batch_size=10,
    eta=0.5,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
)

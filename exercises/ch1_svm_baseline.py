"""Chapter 1 -- SVM baseline (a classical, non-neural-network approach).

A Support Vector Machine trained on the same MNIST data.  This takes
~5 minutes because SVMs scale poorly to 50,000 images with 784 features.

Run:  uv run python exercises/ch1_svm_baseline.py

Expected result: ~9435 / 10000 correct (~94.35%).

The neural network from ch1_basic_network.py beats this with ~95%,
and it trains much faster.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.svm import SVC
import src.mnist_loader as ml

training_data, validation_data, test_data = ml.load_data()

print("Training SVM on 50,000 images (this takes a few minutes)...")
clf = SVC(gamma="auto")
clf.fit(training_data[0], training_data[1])

predictions = clf.predict(test_data[0])
num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
print(f"{num_correct} / {len(test_data[1])} correct")

"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
../data/mnist_expanded.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.

"""

import gzip
import os
import pickle
import random

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EXPANDED_PATH = os.path.join(DATA_DIR, "mnist_expanded.pkl.gz")
MNIST_PATH = os.path.join(DATA_DIR, "mnist.pkl.gz")


def main():
    print("Expanding the MNIST training set")
    if os.path.exists(EXPANDED_PATH):
        print("The expanded training set already exists.  Exiting.")
        return
    f = gzip.open(MNIST_PATH, "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    expanded_training_pairs = []
    j = 0
    for x, y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0:
            print("Expanding image number", j)
        # Create 4 shifted copies of each image: up, down, left, right.
        # This 5x-es the training set, teaching the network that a digit
        # shifted by one pixel is still the same digit.
        for d, axis, index_position, index in [
            (1, 0, "first", 0),
            (-1, 0, "first", 27),
            (1, 1, "last", 0),
            (-1, 1, "last", 27),
        ]:
            # np.roll shifts pixels but wraps them around the edge.
            # Zero out the wrapped edge so the shift looks like a real
            # displacement (new pixels entering the frame are black).
            new_img = np.roll(image, d, axis)
            if index_position == "first":
                new_img[index, :] = np.zeros(28)
            else:
                new_img[:, index] = np.zeros(28)
            expanded_training_pairs.append((np.reshape(new_img, 784), y))
    random.shuffle(expanded_training_pairs)
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    print("Saving expanded data. This may take a few minutes.")
    f = gzip.open(EXPANDED_PATH, "wb")
    pickle.dump((expanded_training_data, validation_data, test_data), f)
    f.close()


if __name__ == "__main__":
    main()

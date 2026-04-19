# How To Use This Code

> **New to neural networks?** Start with **[TUTORIAL.md](TUTORIAL.md)** instead --
> it walks you through the book and exercises step by step, chapter by chapter.
> This page is a technical reference for all available commands.

Hands-on exercise code for [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen.
The full book is available offline in the [`book/`](book/) directory.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- ~200MB disk space (for dependencies)

## Installation

### 1. Install uv (if you don't have it)

    curl -LsSf https://astral.sh/uv/install.sh | sh

### 2. Clone and set up

    git clone https://github.com/itsPWN/neural-networks-and-deep-learning.git
    cd neural-networks-and-deep-learning
    uv sync

That's it. `uv sync` installs Python 3.12, all dependencies, and creates
a virtual environment automatically. The MNIST dataset (~17MB) is included
in the repo.

### 3. Verify installation

    uv run python -c "from src import mnist_loader; d = mnist_loader.load_data_wrapper(); print(f'{len(d[0])} training images loaded')"

Expected output: `50000 training images loaded`

## Running the Exercises

All commands are run from the repo root. Each exercise is a standalone
script in `exercises/` -- run it, read it, modify it.

### Chapter 1: Using neural nets to recognize handwritten digits

    uv run python exercises/ch1_basic_network.py       # ~95% accuracy
    uv run python exercises/ch1_svm_baseline.py        # SVM comparison (~94%, takes ~5 min)
    uv run python exercises/ch1_average_darkness.py    # Naive baseline (22%)
    uv run python exercises/ch1_no_hidden_layer.py     # No hidden layer (~88-92%)

To experiment, open `exercises/ch1_basic_network.py` and change the
hidden layer size (30), learning rate (3.0), or number of epochs (30).

### Chapter 3: Improving the way neural networks learn

    uv run python exercises/ch3_cross_entropy.py            # ~96.5%+ accuracy
    uv run python exercises/ch3_quadratic_cost.py           # ~95% (worse -- shows why cross-entropy wins)
    uv run python exercises/ch3_weight_init.py              # Good vs bad initialization
    uv run python exercises/ch3_learning_rate.py            # Too small / just right / too large
    uv run python exercises/ch3_overfitting.py              # Overfitting on small data
    uv run python exercises/ch3_l1_regularization.py        # L1 vs L2 regularization
    uv run python exercises/ch3_early_stopping.py           # Stop when accuracy plateaus
    uv run python exercises/ch3_momentum.py                 # Momentum-based SGD
    uv run python exercises/ch3_learning_rate_schedule.py   # Adaptive learning rate

### Chapter 5: The vanishing gradient problem

    uv run python exercises/ch5_vanishing_gradient.py       # 1-4 hidden layers comparison

### Chapter 6: Deep learning (convolutional networks)

    uv run python exercises/ch6_fc_baseline.py              # FC-only baseline (~97.80%)
    uv run python exercises/ch6_basic_conv.py               # Single conv (~98.78%)
    uv run python exercises/ch6_conv_softmax_only.py        # Conv + softmax only (no FC)
    uv run python exercises/ch6_double_conv_sigmoid.py      # Double conv sigmoid (~99.06%)
    uv run python exercises/ch6_double_conv_relu.py         # Double conv ReLU (~99.23%)
    uv run python exercises/ch6_double_conv.py              # Dropout + expanded (~99.6%)
    uv run python exercises/ch6_expanded_data.py            # Expanded data (~99.3%+)

## Running the Visualization Scripts

The `plots/` directory contains scripts that generate the plots from the book.

    # Overfitting visualization (Chapter 3)
    uv run python plots/overfitting.py

    # Effect of training set size (Chapter 3)
    uv run python plots/more_data.py

    # Learning rate comparison (Chapter 3)
    uv run python plots/multiple_eta.py

    # Vanishing gradient problem (Chapter 5)
    uv run python plots/generate_gradient.py

    # Weight initialization comparison (Chapter 3)
    uv run python plots/weight_initialization.py

    # Gradient magnitude across layers (Chapter 5)
    uv run python plots/backprop_magnitude_nabla.py

Note: some plots/ scripts are interactive and prompt for parameters.

## Using the Interactive REPL

For experimentation beyond the exercises:

    uv run ipython

Then inside IPython:

    from src import mnist_loader, network, network2
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # ... experiment freely

## Running Tests

    uv run pytest

## Generating Expanded Training Data

For Chapter 6 experiments that use augmented data (250,000 images):

    uv run python src/expand_mnist.py

This takes a few minutes and ~500MB RAM.

## Troubleshooting

**"No module named src"**: Make sure you're running from the repo root,
not from inside `src/`.

**MNIST data missing**: The dataset ships with the repo in `data/mnist.pkl.gz`.
If you deleted it, re-download from
https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz
and place in `data/mnist.pkl.gz`.

**Slow training**: network.py and network2.py are pure numpy (CPU only).
network3.py uses PyTorch and will use GPU if available (CUDA or Apple
Silicon MPS).

## Expected Results Summary

| Exercise                              | Expected Accuracy | Chapter |
| ------------------------------------- | ----------------- | ------- |
| network.py [784, 30, 10] 30 epochs   | ~95.42%           | 1       |
| network.py [784, 10] no hidden layer  | ~88-92%           | 1       |
| SVM baseline (gamma='auto')           | ~94.35%           | 1       |
| Average darkness baseline             | 22.25%            | 1       |
| network2.py cross-entropy 30 epochs   | ~95.49%           | 3       |
| network2.py L1 regularization         | ~96.5%            | 3       |
| network2.py 1-4 hidden layers         | ~96.5-96.9%       | 5       |
| FC-only baseline (network3.py)        | ~97.80%           | 6       |
| Conv net (basic, 60 epochs)           | ~98.78%           | 6       |
| Double conv sigmoid (60 epochs)       | ~99.06%           | 6       |
| Double conv ReLU (60 epochs)          | ~99.23%           | 6       |
| Conv net (expanded data, 60 epochs)   | ~99.3%+           | 6       |
| Conv net (double + dropout, 40 ep)    | ~99.6%            | 6       |

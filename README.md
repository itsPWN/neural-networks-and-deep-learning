# Neural Networks and Deep Learning -- Hands-On Tutorial

Learn how neural networks actually work by building them from scratch.
Based on Michael Nielsen's free book
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).

**What you'll build:** A handwritten digit recognizer that goes from 22%
accuracy (naive baseline) to 99.6% accuracy (deep convolutional network),
step by step.

**What you need:** Basic Python, some high-school math. No ML experience.
See [PREREQUISITES.md](PREREQUISITES.md) for a detailed checklist and recommended resources.

**Works fully offline** -- the book, dataset, and all dependencies are included.

## Get Started

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone, install, and train your first neural network
git clone <repo-url>
cd neural-networks-and-deep-learning
uv sync
uv run python exercises/ch1_basic_network.py
```

Watch the network learn to recognize handwritten digits, reaching ~95%
accuracy in under a minute.

## What's Included

| | What | Description |
|---|------|-------------|
| **[TUTORIAL.md](TUTORIAL.md)** | **Step-by-step learning path** | **Start here.** Read a chapter, run exercises, repeat. |
| [PREREQUISITES.md](PREREQUISITES.md) | Math & Python prerequisites | What to know (or brush up on) before starting |
| [exercises/](exercises/) | Runnable exercise scripts | One script per exercise -- run, read, modify |
| [book/](book/) | The full book (offline) | All 6 chapters as HTML -- open in any browser |
| [src/](src/) | The code | Three neural networks of increasing sophistication |
| [plots/](plots/) | Visualizations | Scripts that generate the book's plots |
| [HOWTO.md](HOWTO.md) | Technical reference | Every command, troubleshooting |

## The Journey

| Chapter | What you learn | Accuracy |
|---------|---------------|----------|
| 1. Your first neural network | How neurons, layers, and gradient descent work | ~95% |
| 2. Backpropagation | How training actually computes which weights to change | -- |
| 3. Improving your network | Cross-entropy, regularization, weight initialization | ~96.5% |
| 4. Universal approximation | Why neural networks can theoretically compute anything | -- |
| 5. Vanishing gradients | Why deep networks were hard to train (historically) | -- |
| 6. Deep learning with CNNs | Convolutional networks that exploit image structure | **~99.6%** |

## Modernized Codebase

This is a fork of [the original repository](https://github.com/mnielsen/neural-networks-and-deep-learning),
modernized to run on current systems:

- Python 2 → **Python 3.12**
- Theano (abandoned 2017) → **PyTorch**
- `requirements.txt` → **uv** project (`pyproject.toml`)
- MNIST data included in the repo (no download needed)

All exercises produce the same results as the original book.

## License

Code: MIT License -- Copyright (c) 2012-2022 Michael Nielsen

Book content (in `book/`): [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/)
-- Copyright (c) 2015 Michael A. Nielsen, Determination Press

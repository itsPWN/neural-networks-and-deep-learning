# Neural Networks and Deep Learning -- Step-by-Step Tutorial

A hands-on learning path through Michael Nielsen's
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).
The book is included offline in the [`book/`](book/) directory -- open the HTML
files in any browser to read them.

> **Who is this for?** Anyone curious about how neural networks actually work.
> You need basic Python and some high-school math (algebra, a little calculus).
> No ML experience required.

---

## Setup (5 minutes)

Before you start, install the project so you can run the exercises:

```bash
# 1. Install uv (Python package manager) if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install everything
git clone https://github.com/itsPWN/neural-networks-and-deep-learning.git
cd neural-networks-and-deep-learning
uv sync

# 3. Verify it works
uv run python -c "from src import mnist_loader; d = mnist_loader.load_data_wrapper(); print(f'{len(d[0])} training images loaded')"
```

You should see: `50000 training images loaded`

That's it. The MNIST dataset (handwritten digits) is included in the repo.

> **Having trouble?** See the [Troubleshooting section in HOWTO.md](HOWTO.md#troubleshooting).

---

## Learning Path

The book has 6 chapters. Some are theory-heavy, others are hands-on.
Follow them in order -- each builds on the previous one.

| Step | Chapter | Type | Time |
|------|---------|------|------|
| 1 | [Ch 1: Your First Neural Network](#step-1-your-first-neural-network) | Read + Code | ~3 hours |
| 2 | [Ch 2: How Backpropagation Works](#step-2-how-backpropagation-works) | Read (theory) | ~1.5 hours |
| 3 | [Ch 3: Improving Your Network](#step-3-improving-your-network) | Read + Code | ~4 hours |
| 4 | [Ch 4: Universal Approximation](#step-4-universal-approximation-theorem) | Read (theory) | ~1 hour |
| 5 | [Ch 5: The Vanishing Gradient Problem](#step-5-the-vanishing-gradient-problem) | Read + Code | ~1 hour |
| 6 | [Ch 6: Deep Learning with CNNs](#step-6-deep-learning-with-cnns) | Read + Code | ~3 hours |

---

## Step 1: Your First Neural Network

### Read

Open [book/chap1.html](book/chap1.html) in your browser and read through it.

**Key sections and what you'll learn:**

| Section | What you'll learn |
|---------|-------------------|
| Perceptrons | The simplest "neuron" -- a function that takes inputs and outputs 0 or 1 |
| Sigmoid neurons | A smoother version that can learn gradually |
| Architecture of neural networks | How neurons are arranged in layers |
| A simple network to classify handwritten digits | The actual problem we'll solve: recognizing 0-9 |
| Learning with gradient descent | How the network improves by taking small steps downhill |
| Implementing our network to classify digits | The code behind `network.py` |

**Don't worry if:** The calculus in the gradient descent section feels
heavy. The key idea is simple: the network has a "cost" (how wrong it is),
and gradient descent finds which direction to nudge the weights to reduce
that cost. You'll see it work concretely in the exercises.

### Exercise 1.1: Train your first network

```bash
uv run python exercises/ch1_basic_network.py
```

This trains a network with 784 input neurons (one per pixel), 30 hidden
neurons, and 10 output neurons (one per digit). Watch the accuracy climb
from ~90% to ~95% over 30 epochs.

**What to notice:**
- Accuracy jumps fast in the first few epochs, then improves more slowly
- ~95% means it gets about 500 out of 10,000 test digits wrong
- This is already better than many hand-crafted algorithms!

**Try changing the code:** Open `exercises/ch1_basic_network.py` and
change the hidden layer size from 30 to 100, or the learning rate
from 3.0 to 0.1. Re-run and see what happens.

### Exercise 1.2: Compare with non-neural-network approaches

**SVM baseline** (a classical machine learning algorithm, takes ~5 min):

```bash
uv run python exercises/ch1_svm_baseline.py
```

Expected: ~94.35%.

**Average darkness baseline** (the simplest possible approach):

```bash
uv run python exercises/ch1_average_darkness.py
```

Expected: 22.25% -- barely better than random guessing (10%).

**What to notice:** Your simple neural network (95%) already beats the SVM
(94%) and crushes the naive baseline (22%).

### Exercise 1.3: Why hidden layers matter

```bash
uv run python exercises/ch1_no_hidden_layer.py
```

Expected: ~88-92% -- significantly worse than 95%.

**What to notice:** Without a hidden layer, the network can only learn
approximately linear decision boundaries. The hidden layer is what lets
it learn the complex patterns that make digit recognition work.

### Checkpoint

At this point you should understand:
- A neural network is layers of neurons connected by weights
- Training = adjusting weights to minimize a cost function
- Gradient descent = repeatedly taking small steps downhill
- Your network can recognize 95% of handwritten digits
- Hidden layers are essential for learning complex patterns

---

## Step 2: How Backpropagation Works

### Read

Open [book/chap2.html](book/chap2.html) in your browser.

**Key sections:**

| Section | What you'll learn |
|---------|-------------------|
| Matrix-based approach | How to compute the network's output efficiently |
| The four fundamental equations | The math behind how errors flow backward through the network |
| The backpropagation algorithm | The step-by-step procedure |
| The code for backpropagation | How `network.py`'s `backprop()` method implements it |

**This chapter is theory-heavy.** It explains *how* gradient descent actually
computes which weights to change. The key insight: instead of testing each
weight individually (impossibly slow), backpropagation efficiently computes
all the gradients in one backward pass through the network.

### Exercise 2.1: Map the four equations to the code

The book's Chapter 2 introduces four backpropagation equations (BP1-BP4).
The `backprop` method in `src/network.py` is a direct translation -- about
20 lines of code, one to one with the math. Open
[src/network.py:84-116](src/network.py#L84-L116) next to the book's
"The code for backpropagation" section and use this map:

| Equation | Plain English | Line in `network.py` |
|----------|---------------|----------------------|
| **BP1**: δᴸ = ∇ₐC ⊙ σ′(zᴸ) | Output-layer error: how wrong each output neuron is, scaled by how sensitive sigmoid is at that point | [101](src/network.py#L101) |
| **BP2**: δˡ = ((wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ σ′(zˡ) | Propagate error one layer back: push the next layer's error backward through the transposed weights, scale by this layer's sensitivity | [113](src/network.py#L113) |
| **BP3**: ∂C/∂bˡ = δˡ | The bias gradient *is* the error -- no extra math needed | [102, 114](src/network.py#L102) |
| **BP4**: ∂C/∂wˡ = δˡ (aˡ⁻¹)ᵀ | The weight gradient is the error times the previous layer's activation | [103, 115](src/network.py#L103) |

**Don't worry if:** The notation feels overwhelming. Focus on BP1 first --
it's just "how wrong is the output, scaled by the slope of sigmoid". BP2
is the same idea applied to hidden layers. BP3 and BP4 are bookkeeping:
once you know the error δ at each layer, you already have the gradients.

No need to run anything -- this is a reading exercise.

### Checkpoint

At this point you should understand:
- Backpropagation computes gradients by propagating errors backward
- It's efficient: one forward pass + one backward pass
- The code in `network.py` directly implements the four equations

---

## Step 3: Improving Your Network

### Read

Open [book/chap3.html](book/chap3.html) in your browser. This is the longest
chapter and the most practically useful.

**Key sections:**

| Section | What you'll learn |
|---------|-------------------|
| The cross-entropy cost function | A better way to measure "how wrong" the network is |
| Overfitting and regularization | When the network memorizes instead of learning, and how to fix it |
| Weight initialization | Why starting weights matter |
| Handwriting recognition revisited | The improved `network2.py` code |
| How to choose hyper-parameters | Practical advice for tuning |
| Other techniques | Momentum, dropout, and more |

### Exercise 3.1: Cross-entropy vs quadratic cost

```bash
uv run python exercises/ch3_cross_entropy.py
```

Expected: ~95.49% accuracy (similar to Chapter 1's 95.42%).

Now compare with the old quadratic cost:

```bash
uv run python exercises/ch3_quadratic_cost.py
```

Expected: ~95.42%.

**What to notice:** Both reach similar final accuracy, but cross-entropy
learns faster in early epochs. The book explains why: quadratic cost has a
"learning slowdown" problem when the network is confidently wrong. The
real accuracy gains come from combining cross-entropy with regularization
and better weight initialization (exercises 3.2-3.4).

### Exercise 3.2: Weight initialization

```bash
uv run python exercises/ch3_weight_init.py
```

**What to notice:** Good initialization converges faster. With large initial
weights, neurons saturate (output near 0 or 1) and learn slowly.

### Exercise 3.3: Learning rate experiments

```bash
uv run python exercises/ch3_learning_rate.py
```

**What to notice:**
- `eta=0.025`: Too slow -- cost barely decreases
- `eta=0.25`: Just right -- steady improvement
- `eta=2.5`: Too fast -- cost oscillates wildly, network doesn't learn

This is one of the most important practical lessons: the learning rate
is the single most important hyper-parameter to tune.

### Exercise 3.4: See overfitting in action

```bash
uv run python exercises/ch3_overfitting.py
```

This trains on only 1,000 images with no regularization. Watch training
accuracy climb toward 100% while test accuracy stalls around 82-84% --
the classic sign of overfitting.

**Why it matters:** This is the motivation for regularization. The network
memorizes the small training set instead of learning general patterns.

### Exercise 3.5: L1 vs L2 regularization

```bash
uv run python exercises/ch3_l1_regularization.py
```

**What to notice:** Both reach similar accuracy, but L1 produces sparser
weights (more weights driven to exactly zero). The script prints weight
sparsity statistics after training.

### Exercise 3.6: Early stopping

```bash
uv run python exercises/ch3_early_stopping.py
```

**What to notice:** Training stops before 100 epochs -- typically around
epoch 20-30 -- at the point where further training would only overfit.
This is a simple but effective regularization technique.

### Exercise 3.7: Momentum-based gradient descent

```bash
uv run python exercises/ch3_momentum.py
```

**What to notice:** Momentum (mu=0.9) accumulates a "velocity" from past
gradients, helping the network move faster through flat regions and
dampen oscillations. Compare convergence speed with standard SGD.

### Exercise 3.8: Learning rate schedule

```bash
uv run python exercises/ch3_learning_rate_schedule.py
```

**What to notice:** Instead of a fixed learning rate, eta is halved each
time accuracy plateaus for 10 epochs. Training terminates when eta drops
to 1/128 of its original value. This adapts the learning rate
automatically -- fast at first, then fine-grained.

### Checkpoint

At this point you should understand:
- Cross-entropy is better than quadratic cost for classification
- Regularization (L2 and L1) prevents overfitting
- Early stopping halts training at the optimal point
- Momentum accelerates convergence
- Adaptive learning rate schedules automate hyper-parameter tuning
- Good weight initialization speeds up training
- `network2.py` adds all these improvements to `network.py`

---

## Step 4: Universal Approximation Theorem

### Read

Open [book/chap4.html](book/chap4.html) in your browser.

**This chapter is different** -- it's a visual, interactive proof that neural
networks can (in theory) compute *any* function. The chapter has interactive
JavaScript demos where you can manipulate neuron weights and see the effect
in real time.

**Key idea:** A single hidden layer with enough neurons can approximate any
continuous function to arbitrary precision. This doesn't mean it's *easy* to
find the right weights (that's what training is for), but it means neural
networks are fundamentally powerful.

There are no coding exercises for this chapter -- just read and play with
the interactive demos.

### Checkpoint

You should understand: Neural networks are universal approximators. The
challenge isn't *whether* a network can represent the answer, but *how*
to train it to find the right weights.

---

## Step 5: The Vanishing Gradient Problem

### Read

Open [book/chap5.html](book/chap5.html) in your browser.

**Key sections:**

| Section | What you'll learn |
|---------|-------------------|
| The vanishing gradient problem | Why deeper networks are harder to train |
| Unstable gradients | The mathematical reason: gradients shrink exponentially |
| Other obstacles | Additional challenges in deep learning |

**Key idea:** In deep networks (many layers), gradients get multiplied
together during backpropagation. If each multiplication makes them smaller,
by the time you reach the first layers, the gradient is essentially zero --
those early layers stop learning. This is why Chapter 1's simple network
only had one hidden layer.

### Exercise 5.1: See the vanishing gradient in action

```bash
uv run python exercises/ch5_vanishing_gradient.py
```

This trains networks with 1, 2, 3, and 4 hidden layers. Watch how adding
more layers fails to improve accuracy -- and may even hurt it.

**Expected results:**

| Architecture | Hidden Layers | Accuracy |
|---|---|---|
| [784, 30, 10] | 1 | ~96.48% |
| [784, 30, 30, 10] | 2 | ~96.90% |
| [784, 30, 30, 30, 10] | 3 | ~96.57% |
| [784, 30, 30, 30, 30, 10] | 4 | ~96.53% |

**What to notice:** More layers should help, but they don't! Gradients
shrink exponentially as they propagate backward through sigmoid layers.
Early layers barely learn at all.

### Exercise 5.2: Visualize gradient magnitudes

```bash
uv run python plots/generate_gradient.py
```

This generates plots showing how gradient norms decrease in earlier layers
during training. The visualization makes the vanishing gradient problem
concrete.

### Checkpoint

You now understand *why* deep networks were hard to train historically --
and why techniques like ReLU activation, better initialization, and
convolutional architectures (Chapter 6) were needed to make deep learning
work.

---

## Step 6: Deep Learning with CNNs

### Read

Open [book/chap6.html](book/chap6.html) in your browser.

**Key sections:**

| Section | What you'll learn |
|---------|-------------------|
| Introducing convolutional networks | How CNNs exploit spatial structure in images |
| CNNs in practice | Pooling, shared weights, and why CNNs work |
| The code for our CNNs | The `network3.py` implementation |
| Recent progress in image recognition | Where the field has gone |

**Key ideas:**
- **Convolutional layers** learn local features (edges, corners, textures)
- **Pooling** reduces spatial size and provides translation invariance
- **Stacking conv layers** builds up from simple features to complex ones
- This is what "deep" in deep learning means: many layers of increasing abstraction

### Exercise 6.1: Fully-connected baseline

```bash
uv run python exercises/ch6_fc_baseline.py
```

Expected: ~97.80% accuracy. This uses the same PyTorch framework as the
CNN exercises but with only fully-connected layers -- no convolution.
It sets the baseline for seeing how much convolution helps.

### Exercise 6.2: Basic convolutional network

```bash
uv run python exercises/ch6_basic_conv.py
```

Expected: ~98.78% accuracy -- a big jump from the FC baseline's 97.80%.

**What changed:** Instead of treating each pixel independently, the
convolutional layer looks at 5x5 patches and learns local patterns.
This is much more natural for images.

### Exercise 6.3: Conv + softmax only (no FC layer)

```bash
uv run python exercises/ch6_conv_softmax_only.py
```

The book asks: does the fully-connected layer actually help? This exercise
removes it to find out. Expected: lower accuracy than the basic conv net,
proving the FC layer adds value.

### Exercise 6.4: Double conv (sigmoid)

```bash
uv run python exercises/ch6_double_conv_sigmoid.py
```

Expected: ~99.06%. Adding a second conv layer detects more complex
features (combinations of edges), boosting accuracy even with sigmoid.

### Exercise 6.5: Double conv with ReLU

```bash
uv run python exercises/ch6_double_conv_relu.py
```

Expected: ~99.23%. Switching from sigmoid to ReLU avoids the vanishing
gradient problem from Chapter 5, allowing the deeper layers to train
effectively.

### Exercise 6.6: Double conv with dropout (best single-network result)

First, generate expanded data if you haven't already:

```bash
uv run python src/expand_mnist.py
```

Then:

```bash
uv run python exercises/ch6_double_conv.py
```

Expected: ~99.6% accuracy. This adds dropout regularization, larger FC
layers (1000 neurons each), and uses expanded training data (250k images).

### Exercise 6.7: Training with expanded data

First, generate the expanded dataset (once, takes ~5 minutes):

```bash
uv run python src/expand_mnist.py
```

Then train:

```bash
uv run python exercises/ch6_expanded_data.py
```

Expected: ~99.3%+ accuracy. By shifting each image 1 pixel in all four
directions, we 5x the training set and teach the network translation
invariance. The book achieves ~99.6% by combining expanded data with the
dropout architecture from Exercise 6.6.

### Checkpoint

You've gone from 95% (simple network) to 99.6% (deep CNN) -- a 10x
reduction in errors. You understand *why* each improvement helps:
convolution, deeper layers, ReLU, dropout, and more data.

---

## Your Progress

| Chapter | Key Code File | Accuracy | What It Proves |
|---------|---------------|----------|----------------|
| 1 | `src/network.py` | ~95% | Neural networks work |
| 1 | (no hidden layer) | ~88-92% | Hidden layers are essential |
| 3 | `src/network2.py` | ~96.5% | Better cost functions and regularization help |
| 5 | (4 hidden layers) | ~96.5% | Deeper isn't always better (vanishing gradient) |
| 6 | `src/network3.py` (FC) | ~97.80% | PyTorch baseline without convolution |
| 6 | `src/network3.py` (CNN) | ~98.78% | Convolution exploits spatial structure |
| 6 | `src/network3.py` (best) | ~99.6% | ReLU + dropout + deep architecture |

---

## What's Next?

You've completed the core tutorial. Here are some directions to explore:

### Experiment more

The exercise scripts are meant to be modified. Open any script in
`exercises/`, change the parameters, and re-run to see what happens.

Or start an interactive REPL:

```bash
uv run ipython
```

```python
from src import mnist_loader, network, network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# Try your own architectures, learning rates, etc.
```

### Run the visualization scripts

The `plots/` directory has scripts that generate the book's plots:

```bash
uv run python plots/overfitting.py             # Overfitting visualization
uv run python plots/more_data.py               # Effect of training set size
uv run python plots/multiple_eta.py            # Learning rate comparison
uv run python plots/weight_initialization.py   # Weight init comparison
```

### Go deeper

- **Modern frameworks:** The principles you've learned apply directly to
  PyTorch, TensorFlow, and JAX. Try reimplementing the exercises using
  PyTorch's `nn.Module` API.
- **Other datasets:** Try CIFAR-10, Fashion-MNIST, or your own image dataset.
- **Read the literature:** The book's Chapter 6 has excellent pointers to
  landmark papers (AlexNet, GoogLeNet, ResNet).

### Run the test suite

Verify everything works:

```bash
uv run pytest
```

---

## Quick Reference

### All exercises at a glance

```bash
# Chapter 1
uv run python exercises/ch1_basic_network.py            # Train your first network (~95%)
uv run python exercises/ch1_svm_baseline.py             # SVM comparison (~94%, slow)
uv run python exercises/ch1_average_darkness.py         # Naive baseline (22%)
uv run python exercises/ch1_no_hidden_layer.py          # No hidden layer (~88-92%)

# Chapter 3
uv run python exercises/ch3_cross_entropy.py            # Cross-entropy cost (~95.49%)
uv run python exercises/ch3_quadratic_cost.py           # Quadratic cost (~95.42%)
uv run python exercises/ch3_weight_init.py              # Good vs bad initialization
uv run python exercises/ch3_learning_rate.py            # Too small / just right / too large
uv run python exercises/ch3_overfitting.py              # Overfitting on small data
uv run python exercises/ch3_l1_regularization.py        # L1 vs L2 regularization
uv run python exercises/ch3_early_stopping.py           # Stop when accuracy plateaus
uv run python exercises/ch3_momentum.py                 # Momentum-based SGD
uv run python exercises/ch3_learning_rate_schedule.py   # Adaptive learning rate

# Chapter 5
uv run python exercises/ch5_vanishing_gradient.py       # 1-4 hidden layers comparison

# Chapter 6
uv run python exercises/ch6_fc_baseline.py              # FC-only baseline (~97.80%)
uv run python exercises/ch6_basic_conv.py               # Single conv (~98.78%)
uv run python exercises/ch6_conv_softmax_only.py        # Conv + softmax only (no FC)
uv run python exercises/ch6_double_conv_sigmoid.py      # Double conv sigmoid (~99.06%)
uv run python exercises/ch6_double_conv_relu.py         # Double conv ReLU (~99.23%)
uv run python exercises/ch6_double_conv.py              # Dropout + expanded (~99.6%)
uv run python exercises/ch6_expanded_data.py            # Expanded data (~99.3%+)
```

### File map

| Directory | What it contains | Purpose |
|-----------|-----------------|---------|
| `exercises/` | Runnable exercise scripts | One per exercise -- run, read, modify |
| `src/` | Neural network implementations | `network.py` (Ch1), `network2.py` (Ch3), `network3.py` (Ch6) |
| `book/` | Offline copy of the full book | Open HTML files in any browser |
| `plots/` | Visualization scripts | Generate the book's plots |

---

*Book content (c) Michael A. Nielsen, 2015 ([CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/)).
Code modernized for Python 3.12 with PyTorch.*

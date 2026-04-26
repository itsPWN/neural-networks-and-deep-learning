# Prerequisites

Everything you should know (or brush up on) before working through this tutorial. The tutorial teaches neural networks and deep learning from scratch, but it assumes comfort with the foundations listed below.

---

## 1. Python Programming

You don't need to be an expert, but you should be comfortable with:

### Core Syntax
- **Variables, loops, conditionals** — `for`, `while`, `if/else`, `range()`
- **Functions** — defining with `def`, calling, return values, default arguments
- **List comprehensions** — compact loops that build lists: `[x**2 for x in range(10)]` gives `[0, 1, 4, 9, ...]`. Used everywhere in the codebase for building weight matrices and mini-batches

### Data Structures
- **Lists** — ordered collections: `[1, 2, 3]`. Slicing is used to create mini-batches: `data[k : k + batch_size]`
- **Tuples** — immutable pairs/groups: `(x, y)`. All training data is stored as lists of `(input, label)` tuples
- **`zip()`** — pairs up elements from two lists: `list(zip([1,2], [3,4]))` gives `[(1,3), (2,4)]`. Used in nearly every function to iterate over weights and biases together: `for b, w in zip(self.biases, self.weights)`

### Object-Oriented Basics
- **Classes** — creating with `class`, the `__init__` constructor, `self`, methods
- **Inheritance** — one class building on another (used in PyTorch's `nn.Module`)
- **`@staticmethod`** — methods that don't need `self` (used for cost functions in `network2.py`)

### Practical
- **Importing modules** — `import numpy as np`, `from sklearn.svm import SVC`
- **File I/O** — the tutorial uses `pickle` and `gzip` to load MNIST data
- **Command line** — running scripts with `python exercises/ch1_basic_network.py`

### Where it's used
The entire codebase is Python. `src/network.py` and `src/network2.py` use pure Python + NumPy. Chapter 6 (`src/network3.py`) adds PyTorch.

### Recommended resources
- [Python Tutorial](https://docs.python.org/3/tutorial/) (official docs, free — Chapters 3-5 for basics, Chapter 9 for classes)
- [Automate the Boring Stuff — Chapter 1-6](https://automatetheboringstuff.com/) (free online, very beginner-friendly)

---

## 2. High School Algebra

### Functions
- **What is a function** — a rule that takes an input and gives an output, written as f(x). For example, f(x) = 2x + 1 means "double the input and add 1"
- **Evaluating functions** — plugging in values: if f(x) = x^2, then f(3) = 9
- **Function composition** — feeding one function's output into another: if f(x) = x^2 and g(x) = x + 1, then f(g(x)) = (x + 1)^2. This idea is everywhere in neural networks — each layer is a function applied to the previous layer's output

### Exponents and Logarithms
- **Exponent rules** — `x^a * x^b = x^(a+b)`, `(x^a)^b = x^(a*b)`, `x^0 = 1`, `x^(-1) = 1/x`
- **The number e (Euler's number)** — approximately 2.718, shows up constantly. The function `e^x` (also written `exp(x)`) is special because its rate of growth equals its current value
- **Natural logarithm ln(x)** — the inverse of e^x. If `e^2 = 7.389`, then `ln(7.389) = 2`. Key property: `ln(e^x) = x` and `e^(ln(x)) = x`
- **Log rules** — `ln(a*b) = ln(a) + ln(b)`, `ln(a/b) = ln(a) - ln(b)`, `ln(a^n) = n*ln(a)`
- **Why this matters** — the sigmoid function `1 / (1 + e^(-z))` uses e^x, and the cross-entropy cost function uses ln(x). You'll see these on almost every page

### Summation Notation (Sigma Notation)
- **Reading it** — the symbol `Σ` (capital sigma) means "add up a series of terms". For example, `Σ(i=1 to 3) of i^2` means `1^2 + 2^2 + 3^2 = 14`
- **With indices** — `Σ(j=0 to 9) y_j` means "add up y_0 + y_1 + y_2 + ... + y_9". The subscript j is just a counter
- **Why this matters** — cost functions sum over all training examples, and layer computations sum over all inputs to a neuron

### Equation Manipulation
- **Rearranging** — isolating a variable: if `y = mx + b`, then `x = (y - b) / m`
- **Substitution** — replacing a variable with an expression: if `z = wx + b` and `a = sigmoid(z)`, you can write `a = sigmoid(wx + b)`
- **Factoring** — pulling out common terms: `2x + 2y = 2(x + y)`
- **Fractions** — adding, multiplying, and simplifying fractions (the learning rate `eta/n` divides every gradient)

### Where it's used
Every equation in the tutorial builds on these. For example:
- The sigmoid function `sigma(z) = 1 / (1 + e^(-z))` requires understanding e^x and fractions
- The cost function `C = -1/n * Σ [y*ln(a) + (1-y)*ln(1-a)]` uses summation, logarithms, and fractions all at once
- Weight updates `w' = w - (eta/n) * ∂C/∂w` require rearranging and substitution

### Recommended resources
- [Khan Academy — Algebra 1](https://www.khanacademy.org/math/algebra) (free — covers functions, equations, exponents)
- [Khan Academy — Algebra 2](https://www.khanacademy.org/math/algebra2) (free — covers logarithms, exponentials, sigma notation)
- [Math is Fun — Exponents](https://www.mathsisfun.com/exponent.html) (quick visual refresher)
- [Math is Fun — Logarithms](https://www.mathsisfun.com/algebra/logarithms.html) (short and beginner-friendly)
- [Better Explained — An Intuitive Guide to Exponential Functions & e](https://betterexplained.com/articles/an-intuitive-guide-to-exponential-functions-e/) (explains *why* e matters, not just how to use it)

---

## 3. Calculus

This is the most important math prerequisite. If you only brush up on one thing, make it this.

### Derivatives (Single Variable)
- **What a derivative means** — the rate of change of a function. If f(x) = x^2, the derivative f'(x) = 2x tells you "how fast is f changing at this point?" At x=3, f is changing at a rate of 6
- **Key derivatives you'll see** — `d/dx(x^n) = n*x^(n-1)`, `d/dx(e^x) = e^x`, `d/dx(ln(x)) = 1/x`
- **The sigmoid derivative** — the tutorial derives that if `sigma(z) = 1/(1+e^(-z))`, then `sigma'(z) = sigma(z) * (1 - sigma(z))`. This is elegant — the derivative is expressed in terms of the function itself

### Chain Rule
- **The idea** — when functions are nested, multiply derivatives of each layer. If `y = f(g(x))`, then `dy/dx = f'(g(x)) * g'(x)`
- **Concrete example** — if `f(z) = z^2` and `g(x) = 3x + 1`, then `f(g(x)) = (3x+1)^2`. The derivative is `2(3x+1) * 3 = 6(3x+1)`. You differentiate the outer function, then multiply by the derivative of the inner
- **Why this is critical** — backpropagation IS the chain rule applied repeatedly. A neural network is a chain of nested functions: `output = f3(f2(f1(input)))`. To find how the input affects the output, you multiply derivatives through each layer. This is literally what `src/network.py:84-116` computes

### Partial Derivatives and Gradients
- **Partial derivatives** — when a function has multiple inputs, take the derivative with respect to one variable while treating the rest as constants. If `f(x, y) = x^2 + 3xy`, then `∂f/∂x = 2x + 3y` (treat y as a constant) and `∂f/∂y = 3x` (treat x as a constant)
- **Why this matters** — a neural network's cost depends on thousands of weights. To improve the network, you need to know "how does changing *this specific weight* affect the total cost?" That's a partial derivative
- **Gradient** — the vector of all partial derivatives: `∇f = [∂f/∂x, ∂f/∂y, ...]`. It points in the direction of steepest increase
- **Gradient descent** — to minimize the cost, step in the opposite direction of the gradient: `w_new = w_old - learning_rate * ∂C/∂w`. This one equation is the core of all training in the tutorial

### Where it's used
- `src/network.py:84-116` — backpropagation applies the chain rule to compute gradients layer by layer
- `src/network.py:126` — `cost_derivative` computes `∂C/∂a` (partial derivative of cost w.r.t. output)
- `src/network.py:137-139` — sigmoid derivative: `sigmoid(z) * (1 - sigmoid(z))`
- `src/network.py:77-78` — gradient descent update: `w - (eta / len(mini_batch)) * nw`
- Gradient descent is the core optimization loop in every network file

### Recommended resources
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) (YouTube, visual and intuitive — start here)
- [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (directly relevant — covers backpropagation visually)
- [Khan Academy — Calculus 1](https://www.khanacademy.org/math/calculus-1) (free — for derivatives and chain rule)
- [Khan Academy — Multivariable Calculus (partial derivatives section only)](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) (free — you only need the first unit on partial derivatives, not the full course)

---

## 4. Linear Algebra (Basics)

You don't need a full course. Neural networks use a small but essential slice of linear algebra.

### Vectors
- **What they are** — an ordered list of numbers. In this tutorial, a single MNIST image is a vector of 784 numbers (one per pixel), and a bias is a vector with one number per neuron
- **Column vectors** — written vertically, shape is (n, 1). The code creates them with `np.random.randn(y, 1)`. Most vectors in the tutorial are column vectors
- **Addition and scalar multiplication** — `[1,2] + [3,4] = [4,6]` and `3 * [1,2] = [3,6]`. Weight updates add scaled gradient vectors to weight vectors

### Matrices
- **What they are** — a 2D grid of numbers with rows and columns. A weight matrix connects one layer to the next. If layer 1 has 784 neurons and layer 2 has 30, the weight matrix is 30 rows x 784 columns
- **Dimensions** — always described as (rows x columns). A 30x784 matrix has 30 rows and 784 columns = 23,520 individual weights

### Dot Product
- **What it is** — multiply matching elements, then add them up: `[a, b, c] · [d, e, f] = a*d + b*e + c*f`. The result is a single number
- **Why it matters** — a single neuron computes a dot product: `output = sigmoid(weights · inputs + bias)`. Every "score" a neuron produces is one dot product. Matrix-vector multiplication (next section) is just doing many dot products at once — one per row of the matrix

### Matrix-Vector Multiplication
- **How it works** — each row of the matrix is "dotted" with the input vector (multiply corresponding elements, then sum) to produce one output number. A 30x784 matrix times a 784x1 vector gives a 30x1 vector
- **Dimension rule** — the inner dimensions must match: (30x**784**) times (**784**x1) works. (30x784) times (30x1) does not
- **Why this matters** — the entire feedforward pass is this operation repeated: `output = W * input + bias`. One matrix multiplication replaces 23,520 individual multiplications and additions

### Transpose
- **What it does** — flips rows and columns. A 30x784 matrix becomes a 784x30 matrix. Written as A^T
- **Why it's needed** — backpropagation needs to pass errors backwards through the network. Going forward uses W, going backward uses W^T (the transpose). See `src/network.py:113`: `np.dot(self.weights[-l+1].transpose(), delta)`

### Norms (Vector Size)
- **L2 norm** — the "length" of a vector: square each element, sum them, take the square root. `||[3, 4]|| = sqrt(9 + 16) = 5`
- **L1 norm** — sum of absolute values: `|3| + |4| = 7`
- **Why this matters** — L1 and L2 regularization penalize large weights by adding the norm of the weight vector to the cost function

### Tensors (N-Dimensional Arrays)
- **What they are** — a generalization of vectors (1D) and matrices (2D) to any number of dimensions. A 3D tensor is a stack of matrices; a 4D tensor is a stack of stacks
- **Why this matters for Chapter 6** — a single MNIST image is a 28×28 matrix, but PyTorch processes images in *batches* with a *channel* dimension, so the input shape becomes `(batch_size, channels, height, width)` — a 4D tensor. A batch of 10 grayscale 28×28 digits has shape `(10, 1, 28, 28)`. You don't manipulate these by hand; PyTorch handles them. Recognizing the shape just saves confusion when you see `image_shape=(10, 1, 28, 28)` in the code

### Where it's used
- `src/network.py:36` — feedforward: `np.dot(w, a) + b` multiplies weight matrix by activation vector
- `src/network.py:103` — backprop: `np.dot(delta, activations[-2].transpose())`
- `src/network2.py:390` — L2 regularization: `np.linalg.norm(w)**2`
- Neural networks are fundamentally sequences of matrix multiplications followed by nonlinear functions

### Recommended resources
- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (YouTube — best visual intuition for what matrix multiplication *means*, watch at least videos 1-4)
- [Khan Academy — Linear Algebra](https://www.khanacademy.org/math/linear-algebra) (free — more practice problems if you want to drill the mechanics)

---

## 5. Probability & Statistics (Basics)

A light understanding is enough — you don't need a full statistics course.

### Probability
- **What it means** — a number between 0 and 1. 0 = impossible, 1 = certain, 0.8 = 80% chance
- **Probabilities sum to 1** — if a digit classifier outputs `[0.01, 0.02, 0.05, 0.7, 0.1, ...]` for digits 0-9, all values should add up to 1.0. The softmax function (Chapter 6) enforces this

### Normal (Gaussian) Distribution
- **The bell curve** — most values cluster near the center (mean), with fewer values further away
- **Mean (average)** — the center of the bell curve. A mean of 0 means values are centered around zero
- **Standard deviation** — how spread out the values are. Small = tightly clustered, large = widely spread
- **Why this matters** — all weights and biases in the tutorial start as random numbers drawn from a normal distribution with mean 0. `np.random.randn()` generates these. In Chapter 3, the tutorial shows that *how* you scale these initial random values (dividing by `sqrt(n)`) makes a big difference in whether the network learns well

### Variance
- **What it means** — standard deviation squared. Measures how "spread out" numbers are
- **Why it shows up** — the improved weight initialization in `network2.py` divides by `sqrt(number_of_inputs)` specifically to control the variance of each neuron's output, preventing signals from exploding or vanishing before training even starts

### Where it's used
- `src/network.py:30-31` — weights and biases initialized from a normal distribution: `np.random.randn(y, x)`
- `src/network2.py:93-96` — improved initialization divides by `sqrt(n)` to control variance
- `src/mnist_loader.py:100-107` — one-hot encoding: label "3" becomes `[0,0,0,1,0,0,0,0,0,0]`
- The softmax function (Chapter 6) converts raw scores into probabilities that sum to 1

### Recommended resources
- [Khan Academy — Statistics & Probability](https://www.khanacademy.org/math/statistics-probability) (free — focus on "Modeling distributions" and "Normal distributions" sections)
- [Seeing Theory — Basic Probability](https://seeing-theory.brown.edu/basic-probability/index.html) (interactive visualizations, very intuitive)

---

## 6. NumPy (Python Library)

The first 5 chapters use NumPy heavily. You should know:

- **Creating arrays** — `np.array()`, `np.zeros()`, `np.random.randn()`
- **Array shapes and reshaping** — `.shape`, `np.reshape()`
- **Element-wise operations** — adding, multiplying, dividing arrays of the same shape
- **Matrix operations** — `np.dot()` for matrix multiplication
- **Broadcasting** — how NumPy handles operations on arrays of different shapes
- **Aggregations** — `np.sum()`, `np.argmax()`, `np.linalg.norm()`

### Where it's used
Everywhere in `src/network.py` and `src/network2.py`. Every weight update, every feedforward pass, every gradient calculation uses NumPy arrays.

### Recommended resource
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html) (official, free)

---

## 7. PyTorch (for Chapter 6 only)

Chapter 6 switches from hand-coded NumPy networks to PyTorch. You can learn this as you go, but helpful to know:

- **Tensors** — PyTorch's version of NumPy arrays, with GPU support
- **nn.Module** — base class for defining neural network layers
- **Forward pass** — defining how data flows through the network
- **Optimizers** — `torch.optim.SGD` handles weight updates for you
- **Loss functions** — `F.nll_loss`, `F.log_softmax`
- **Training/eval modes** — `model.train()` vs `model.eval()` (affects dropout)

### Where it's used
- `src/network3.py` — the entire Chapter 6 implementation
- `exercises/ch6_*.py` — all Chapter 6 exercises

### Recommended resource
- [PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) (official, free)

---

## What You Do NOT Need Beforehand

The tutorial teaches these from scratch — don't worry about knowing them in advance:

- How neural networks work (that's the whole point!)
- Backpropagation algorithm
- Convolutional neural networks (CNNs)
- Activation functions (sigmoid, ReLU, softmax)
- Cost/loss functions (quadratic, cross-entropy)
- Regularization (L1, L2, dropout)
- Optimization tricks (momentum, learning rate schedules, early stopping)
- The vanishing gradient problem
- The universal approximation theorem

---

## Suggested Preparation Plan

If your math is rusty, here's a focused path (not everything — just what matters most):

| Priority | Topic | Resource |
|----------|-------|----------|
| **Must** | Exponents, logarithms, and e | Khan Academy Algebra 2 + Better Explained (e) |
| **Must** | Derivatives & chain rule | 3Blue1Brown Calculus (videos 1-4) |
| **Must** | Partial derivatives & gradients | Khan Academy Multivariable Calculus (just the partials unit) |
| **Must** | Matrix multiplication & transpose | 3Blue1Brown Linear Algebra (videos 1-4) |
| **Should** | NumPy basics | NumPy Quickstart Tutorial |
| **Should** | Normal distribution & variance | Khan Academy Stats or Seeing Theory |
| **Nice** | Python classes & zip/tuples | Python official tutorial (Chapters 5 & 9) |
| **Later** | PyTorch basics | PyTorch 60-Minute Blitz (before Chapter 6) |

---

## What Each Chapter Needs

You don't have to master everything before starting. Here's what each
chapter actually uses, so you can brush up just-in-time:

| Chapter | Math you'll need | Python/Library | New concepts the chapter introduces |
|---------|------------------|----------------|-------------------------------------|
| Ch 1: First network | Sigmoid (eˣ), dot product, gradient descent intuition | NumPy basics, Python classes | Neurons, layers, SGD, MNIST loading |
| Ch 2: Backpropagation | Chain rule, partial derivatives, matrix transpose | (just reading code) | The four BP equations |
| Ch 3: Improvements | ln(x), L1/L2 norms, variance | NumPy: `np.argmax`, `np.linalg.norm` | Cross-entropy, regularization, weight init, momentum |
| Ch 4: Universal approximation | (none — visual demos only) | (none) | Why neural nets can fit any function |
| Ch 5: Vanishing gradients | σ′(z) ≤ 0.25 (matters when the chain rule multiplies many of them) | (same as Ch 1-3) | Why deep networks were historically hard |
| Ch 6: Deep CNNs | 4D tensors, softmax, dropout (probability) | PyTorch basics (`nn.Module`, optimizer, `F.nll_loss`) | Convolution, pooling, ReLU, deep architecture |

If a chapter introduces a concept you haven't seen, the book explains
it — you don't need to pre-learn it. The "Math you'll need" column is
the *foundation* the explanation builds on.

---

## Quick Self-Test

Can you answer these? If yes, you're ready to start.

1. **Algebra** — Simplify `e^(ln(5))` *(Answer: 5 — e and ln are inverses)*
2. **Algebra** — What is `ln(a*b)` in terms of `ln(a)` and `ln(b)`? *(Answer: ln(a) + ln(b))*
3. **Calculus** — What is the derivative of `f(x) = x^2`? *(Answer: 2x)*
4. **Calculus** — If `f(x) = e^(3x)`, what is `f'(x)`? *(Answer: 3e^(3x) — chain rule)*
5. **Calculus** — If `f(x, y) = x^2 + 3xy`, what is `∂f/∂x`? *(Answer: 2x + 3y — treat y as a constant)*
6. **Linear Algebra** — If A is a 3x784 matrix and v is a 784x1 vector, what is the shape of Av? *(Answer: 3x1)*
7. **Linear Algebra** — What does transposing a 30x784 matrix give you? *(Answer: a 784x30 matrix)*
8. **Probability** — If weights are initialized with `np.random.randn(30, 784)`, what distribution are they drawn from? *(Answer: normal/Gaussian distribution with mean 0 and standard deviation 1)*
9. **Python/NumPy** — What does `list(zip([1,2], [3,4]))` produce? *(Answer: [(1,3), (2,4)])*
10. **Python/NumPy** — What does `np.dot(A, B)` do? *(Answer: matrix multiplication)*

If some of these are fuzzy, spend time on the "Must" items in the preparation plan above before diving in. The tutorial will make much more sense with these foundations solid.

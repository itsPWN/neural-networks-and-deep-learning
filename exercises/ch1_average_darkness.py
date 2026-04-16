"""Chapter 1 -- Average darkness baseline (the simplest possible approach).

Classifies digits by how dark they are overall: "1" is light, "8" is dark.
This is a terrible classifier, but it sets the floor for comparison.

Run:  uv run python exercises/ch1_average_darkness.py

Expected result: 2225 / 10000 correct (22.25%).

This is barely better than random guessing (10%), showing why we
need neural networks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.mnist_average_darkness as m

m.main()

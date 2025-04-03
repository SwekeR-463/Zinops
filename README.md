# Zinops -> Einops from Scratch

A from-scratch implementation of einops-style tensor rearrangement in Python using NumPy, with unit tests.

## Overview

This project implements a subset of the `einops` library's `rearrange` function, supporting tensor operations like transpose, split, merge, repeat, and ellipsis-based transformations.

## Approach

- **Pattern Parsing**: A custom parser handles einops-style patterns (e.g., `'h w -> w h'`), preserving parentheses for split/merge operations using a state machine to track nesting.
- **Operation Detection**: Dynamically identifies operation types (split, merge, repeat, ellipsis) to apply specific validation and transformation logic.
- **Shape Management**: Uses an `OrderedDict` to map axis names to sizes, ensuring consistent ordering and supporting inferred dimensions.
- **Transformations**: Combines `reshape`, `transpose`, and `broadcast_to` in a single pass, minimizing intermediate copies.

## Design Decisions

1. **Simplicity**: Focused on core operations, excluding advanced features like `reduce` for clarity.
2. **Error Handling**: Explicit `Error` messages for invalid inputs, inspired by debugging needs.
3. **NumPy Only**: Leverages NumPy's efficiency without additional dependencies.
4. **Type Hints**: Added for clarity and static analysis.
5. **Single-Pass**: Combines transformations in one function, optimizing for small to medium tensors.

## Requirements

- Python 3.10+
- NumPy (`pip install numpy`)

## Installation

Clone or download this repository:
```bash
git clone https://github.com/SwekeR-463/Zinops.git
```

Install dependency:
```bash
pip install numpy
```

## Usage
```python
import numpy as np
from zinops import rearrange

# Transpose
x = np.arange(12).reshape(3, 4)
result = rearrange(x, 'h w -> w h')  # Shape: (4, 3)

# Split
x = np.arange(12).reshape(12, 1)
result = rearrange(x, '(h w) c -> h w c', h=3)  # Shape: (3, 4, 1)

# Merge with reorder
x = np.random.rand(2, 3, 4, 5)
result = rearrange(x, 'b h w c -> b (c h w)')  # Shape: (2, 60)
```

## Running Tests
Unit tests in `tests.py` cover basic operations, combinations and some edge cases taken from the [official einops docs](https://einops.rocks/api/rearrange/).

Run
```bash
python tests.py
```

## Errors Faced

The implementation evolved through debugging many errors:

1. **EinopsError: Transpose Axes Mismatch** - Misapplied transpose logic to splits (e.g., `'(h w) c -> h w c'`); you suggested distinct operation handling; fixed with operation-specific validation.
2. **ValueError: Split/Merge Shape Mismatch** - Reshape mismatches (e.g., 12 to `(3, 4, 4, 1)`); resolved by single reshape and component checks.
3. **ValueError: General Size Mismatch** - Incorrect split reshapes (e.g., 12 to `(3, 4, 4, 1)`); fixed with single-pass reshaping.
4. **AssertionError: Ellipsis Shape Mismatch** - Wrong ellipsis shape (`(4, 5, 6)` vs `(2, 3, 20)`); adjusted ellipsis handling.
5. **AxisError: Axis 4 Out of Bounds** - Invalid permutation for `'b h w c -> b (c h w)'`; fixed with input-order permutation.

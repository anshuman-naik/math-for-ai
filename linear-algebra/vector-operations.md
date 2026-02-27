# Vector Operations

Vector operations allow us to combine and scale vectors.
These operations are fundamental in machine learning, where parameters
and data are manipulated mathematically.

## Vector Addition

Two vectors of the same dimension can be added component-wise.

If:
v = (v₁, v₂, ..., vₙ)
w = (w₁, w₂, ..., wₙ)

Then:
v + w = (v₁ + w₁, v₂ + w₂, ..., vₙ + wₙ)

### Geometric Interpretation
Vector addition corresponds to placing one vector at the end of another.
The resulting vector represents the combined displacement.

## Scalar Multiplication

A vector can be multiplied by a scalar (a real number).

If:
v = (v₁, v₂, ..., vₙ)

Then for scalar c:
c·v = (c v₁, c v₂, ..., c vₙ)

### Interpretation
- If |c| > 1 → the vector stretches
- If 0 < |c| < 1 → the vector shrinks
- If c < 0 → direction reverses

## Why This Matters in Machine Learning

- Model parameters are updated using scalar multiplication and addition
- Gradient descent modifies vectors using these operations
- Data transformations rely on linear combinations of vectors

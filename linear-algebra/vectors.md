# Vectors

A vector is a mathematical object that has both **magnitude** (size) and
**direction**. Vectors are fundamental in linear algebra and form the basic
language used to represent data and parameters in machine learning.

## Intuitive Understanding
Geometrically, a vector can be visualized as an arrow in space.
The length of the arrow represents its magnitude, and the orientation of the
arrow represents its direction.

This geometric view is useful for building intuition before working with
higher-dimensional vectors that cannot be directly visualized.

## Types of Vectors
Some commonly encountered vectors include:

- **Zero (null) vector**:  
  A vector with zero magnitude. It represents the absence of direction and is
  usually denoted by **0**.

- **Unit vector**:  
  A vector with magnitude equal to 1. Unit vectors are often used to represent
  direction independently of scale.

- **Position vector**:  
  A vector that represents the position of a point relative to the origin.

- **Direction vector**:  
  A vector that indicates direction without being tied to a specific position.

## Algebraic Representation
A vector in an n-dimensional space can be written as an ordered list of numbers:

v = (v₁, v₂, ..., vₙ)

Each component represents how much the vector extends along a particular axis.

## Importance of Vectors in AI and Machine Learning
Vectors are central to AI and machine learning because:
- Data points are represented as vectors of features
- Model parameters (such as weights) are vectors
- Predictions and errors are computed using vector operations
- Many algorithms can be understood as transformations of vectors in space

Understanding vectors makes it easier to visualize data, interpret model
behavior, and reason about learning algorithms.

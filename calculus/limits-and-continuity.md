# Limits and Continuity

Limits describe the behavior of a function as its input approaches a particular
value. They form the foundation of calculus and are essential for defining
derivatives and understanding smooth change.

## Intuitive Idea of a Limit
Instead of asking for the exact value of a function at a point, a limit asks:
*What value does the function approach as we get closer and closer to that
point?*

This idea allows us to reason about functions even at points where they may not
be directly defined.

## Graphical Interpretation
From a graphical perspective, a limit corresponds to the value that the graph
approaches as the input approaches a specific value from the left and the right.

If the function approaches the same value from both directions, the limit
exists.

## Continuity
A function is said to be **continuous** at a point if:
- The function is defined at that point
- The limit exists at that point
- The value of the function equals the value of the limit

Intuitively, a continuous function can be drawn without lifting the pen.

## Relevance to Machine Learning
Continuity and limits are important in machine learning because:
- Learning algorithms assume smooth changes in loss functions
- Gradients rely on the idea of limits
- Discontinuities can make optimization unstable or undefined

## 25 February

- Initialized mathematics repository for AI
- Documented foundational ideas in calculus
- Focused on intuition before formal definitions

## 26 February
- Began probability section
- Wrote basic probability terminology and concepts

Key realization:
Linear algebra, calculus, and probability form the core mathematical foundation of AI. Building them slowly and clearly is more valuable than rushing into projects.

---

# 📘 Learning Log — 27 February 2026

## Focus Areas
- Conceptual Revision  
- Vector Space Extensions into Probabilistic Frameworks  
- Project Planning: Linear Regression Implementation (Python + Matplotlib)

---

## 1. Revision
Revisited previously covered material in:
- Linear Algebra (vector operations, linear combinations)
- Probability fundamentals

Clarified connections between vector spaces and probabilistic representations.

---

## 2. Extending Vectors into Probability

Explored how:
- Random variables can be viewed as elements in vector spaces  
- Expectation acts as a linear operator  
- Variance relates to inner-product structure  

Connected:
- Vector representation → Feature space  
- Probability distributions → Data representation  

This establishes foundational understanding for statistical learning models.

---

## 3. Linear Regression Project Planning

Designed implementation roadmap for:

- Simple Linear Regression from scratch
- Python implementation (NumPy for computation)
- Visualization using Matplotlib
- Plotting regression line against dataset
- Computing Mean Squared Error (MSE)

Defined next steps:
- Implement regression model using either:
  - Gradient Descent, or  
  - Normal Equation  
- Begin implementation on 28 February 2026

---

## Reflection
Today focused on conceptual consolidation and project structuring rather than intensive problem-solving. The transition from pure mathematics (vectors, probability) toward applied machine learning (regression modeling) is becoming structurally clear.

## Current Focus

- Strengthen conceptual clarity
- Maintain consistent daily progress
- Prioritize understanding over speed


## 11 March

### Focus
Implementation of Linear Regression from mathematical foundations.

### Work Completed
- Structured the machine-learning section of the repository
- Implemented linear regression using the **Normal Equation**
- Implemented **Gradient Descent** framework for optimizing regression parameters
- Created the main script integrating:
  - dataset loading
  - regression computation
  - visualization using matplotlib
  - user input prediction
- Added dataset (`data.csv`) for regression experimentation
- Documented calculus concepts required for optimization:
  - derivatives
  - partial derivatives
  - gradient vector
  - chain rule
  - gradient descent
  - mean squared error cost function

### Key Mathematical Insight
Linear regression can be solved in two fundamental ways:

**Closed-form solution (Normal Equation)**

\beta = (X^T X)^{-1} X^T y

**Iterative optimization (Gradient Descent)**

\theta = \theta - \alpha \nabla J(\theta)

Both methods aim to minimize the **Mean Squared Error** cost function.

### Reflection
Today marked the transition from pure mathematical study (linear algebra and calculus) to practical machine learning implementation. Understanding how matrix operations and derivatives translate directly into algorithms made the learning process more concrete.

## 12 March

Focus: Optimization behavior of Linear Regression.

- Implemented loss tracking during gradient descent
- Visualized learning curve (loss vs iterations)
- Compared regression parameters from normal equation and gradient descent
- Expanded dataset to improve regression demonstration
- Strengthened understanding of optimization in machine learning

Key insight:
Gradient descent gradually approaches the same solution as the normal equation while minimizing the loss function over iterations.

## 13 March

- Expanded regression dataset to 30 samples
- Improved visualization of regression trend
- Observed more stable gradient descent convergence
- Compared regression outputs from normal equation and gradient descent on larger dataset

Focused primarily on expanding the linear regression implementation in the machine-learning repository.

Activities included:
- expanding the dataset
- visualizing convergence of gradient descent
- comparing normal equation and gradient descent solutions

Reflected on how previously studied mathematical topics connect to the implementation:

Linear Algebra → matrix operations for regression  
Calculus → gradient descent optimization  
Probability → future modeling of uncertainty




## 20 March

Focus: Optimization analysis and model behavior.

- Conducted experiments with different learning rates in gradient descent
- Visualized convergence behavior for multiple learning rates
- Introduced noise into dataset to simulate real-world conditions
- Compared model performance using Mean Squared Error
- Observed stability and convergence differences across learning rates

Key insight:
The learning rate critically affects convergence. Too small leads to slow learning, while too large causes instability. Proper tuning is essential for effective optimization.

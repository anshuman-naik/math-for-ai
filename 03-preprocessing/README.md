# Feature Scaling

## Standardization (Z-score normalization)

Transforms feature:

x' = (x - μ) / σ

Where:
- μ = mean
- σ = standard deviation

## Why use it?

- Speeds up gradient descent
- Prevents one feature from dominating
- Enables higher learning rates

## Important Notes

- Scaling must be applied during both training and inference
- Models trained on scaled data must use same scaling parameters for prediction

## Observations

- Mini-batch GD performed better with scaling
- Higher learning rates became stable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression_math import normal_equation, predict

# Load dataset
df = pd.read_csv("data.csv")

X = df["Hours"].values
y = df["Score"].values

# Add bias term
X = np.column_stack((np.ones(len(X)), X))

# Compute parameters
beta = normal_equation(X, y)

# Predictions
y_pred = predict(X, beta)

# Plot
plt.scatter(df["Hours"], df["Score"])
plt.plot(df["Hours"], y_pred)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression (Normal Equation)")

plt.show()

# User prediction
new_hour = float(input("Enter study hours: "))

prediction = beta[0] + beta[1] * new_hour

print("Predicted Score:", prediction)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
In this problem we run the logistic regression with binary classificatoin
without the regularization term
"""

# Load and prepare data
data = {
    'X': [
        0.2, 0.5, 0.8, 1.1, 1.4,
        1.7, 2.0, 2.3, 2.6, 2.9,
        3.2, 3.5, 3.8, 4.1, 4.4,
        4.7, 5.0, 5.3, 5.6, 5.9,
        6.2, 6.5, 6.8, 7.1, 7.4,
        7.7, 8.0, 8.3, 8.6, 8.9
    ],
    'Y': [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# Reshape X to (10, 1) then add bias column
X = df['X'].values.reshape(-1, 1)  # shape (10, 1)
ones_col = np.ones((X.shape[0], 1))
X = np.hstack([ones_col, X])  # shape (10, 2)

# Reshape Y to (10, 1)
Y = df['Y'].values.reshape(-1, 1)

print("X shape:", X.shape)  # (10, 2)
print("Y shape:", Y.shape)  # (10, 1)


# Logistic Regression with Gradient Descent
def Logistic_Regression(x, y, alpha, iterations, theta):
    m = y.shape[0]
    costs = []

    for i in range(iterations):
        z = np.dot(x, theta)
        pred_y = 1 / (1 + np.exp(-z))  # sigmoid
        

        cost = (-1 / m) * np.sum(y * np.log(pred_y + 1e-8) + (1 - y) * np.log(1 - pred_y + 1e-8))
        costs.append(cost)

        gradient = (1 / m) * np.dot(x.T, (pred_y - y))
        theta = theta - alpha * gradient

    return theta, costs

# Initialize theta
theta = np.zeros((2, 1))
iterations = 100
alpha = 0.1

# Train
theta, costs = Logistic_Regression(X, Y, alpha, iterations, theta)

print("Trained theta:\n", theta)

# Plot cost over iterations
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations")
plt.show()





















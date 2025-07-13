import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
data = pd.read_csv('data-for-P-3.txt', header=None, names=['Size', 'Bedrooms', 'Price'])
data.insert(0, 'Ones', 1)  # Add intercept term

# Separate features and target
cols = data.shape[1]
X = data.iloc[:, 0:cols-1].values  # as numpy array
Y = data.iloc[:, cols-1:cols].values  # as numpy array

print("X shape:", X.shape)  # should be (m, n)
print("Y shape:", Y.shape)  # should be (m, 1)


# Gradient Descent for multivariable linear regression
def linear_regression_GD(x, y, alpha, iterations, theta):
    m = y.shape[0]
    costs = []

    for i in range(iterations):
        pred_y = np.dot(x, theta)  # shape (m, 1)
        cost = (1 / (2 * m)) * np.sum((pred_y - y) ** 2)
        costs.append(cost)
        gradient = (1 / m) * np.dot(x.T, (pred_y - y))  # shape (n, 1)
        theta = theta - alpha * gradient

    # Plotting the cost over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, iterations + 1), costs, color='red', marker='.')
    plt.title('Cost vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.grid(True)
    plt.show()

    return theta, costs

# Initialize theta (should match number of features = 3 here)
theta = np.array([[1], [1.5], [3.25]])
iterations = 10
alpha = 0.01

# Run Gradient Descent
theta, _ = linear_regression_GD(X, Y, alpha, iterations, theta)

print("Trained theta:\n", theta)



new_data = np.array([[1, 2100, 3]])  # 1 for bias, size=2100, bedrooms=3
predicted_price = np.dot(new_data, theta)
print("Predicted Price:", predicted_price[0][0])

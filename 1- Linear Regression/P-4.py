import numpy as np
import matplotlib.pyplot as plt


# Set random seed for reproducibility
np.random.seed(42)

# Generate 100 samples
X1 = np.random.randint(1, 10, size=200)
X2 = np.random.randint(10, 50, size=200)
X3 = np.random.randint(100, 500, size=200)

# Stack into a feature matrix X
X = np.column_stack((X1, X2, X3))

# Generate Y using a linear combination + some noise
noise = np.random.normal(0, 10, size=200)  # Add Gaussian noise
Y = 3 * X1 + 2 * X2 + 4 * X3 + noise
Y = Y.reshape(-1, 1)



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
iterations = 30
alpha = 0.01

# Run Gradient Descent
theta, _ = linear_regression_GD(X, Y, alpha, iterations, theta)

print("Trained theta:\n", theta)



X1 = np.random.randint(1, 10, size=1)
X2 = np.random.randint(10, 50, size=1)
X3 = np.random.randint(100, 500, size=1)

# Stack into a feature matrix X
new_data = np.column_stack((X1, X2, X3))
predicted_price = np.dot(new_data, theta)
print("Predicted Price:", predicted_price[0][0])
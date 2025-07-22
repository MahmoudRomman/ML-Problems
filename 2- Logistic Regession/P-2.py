import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
In this problem we run the logistic regression with binary classificatoin
without the regularization term
"""



# Load and prepare data
data = {
    'X1': [0.2, 1.1, 1.5, 2.3, 3.0, 3.3, 4.5, 4.8, 5.2, 6.1],
    'X2': [0.1, 0.9, 1.4, 2.0, 2.7, 3.1, 3.9, 4.4, 5.0, 5.9],
    'X3': [0.3, 1.0, 1.6, 2.4, 3.3, 3.6, 4.8, 5.1, 5.7, 6.3],
    'Y' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
data = pd.DataFrame(data)


# Separate features and target
cols = data.shape[1]
X = data.iloc[:, 0:cols-1].values  # as numpy array

ones_col = np.ones((X.shape[0], 1))
X = np.hstack([ones_col, X])  # shape (10, 4)

# Reshape Y to (10, 1)
Y = data.iloc[:, cols-1:cols].values  # as numpy array

print("X shape:", X.shape)  # (10, 4)
print("Y shape:", Y.shape)  # (10, 1)




# Logistic Regression with Gradient Descent
def Logistic_Regression(x, y, alpha, iterations):
    
    m = y.shape[0]
    n = x.shape[1]
    theta = np.zeros((n, 1))
    costs = []

    for i in range(iterations):
        z = np.dot(x, theta)
        pred_y = 1 / (1 + np.exp(-z))  # sigmoid
        

        cost = (-1 / m) * np.sum(y * np.log(pred_y + 1e-8) + (1 - y) * np.log(1 - pred_y + 1e-8))
        costs.append(cost)

        gradient = (1 / m) * np.dot(x.T, (pred_y - y))
        theta = theta - alpha * gradient

    return theta, costs


iterations = 1000
alpha = 0.1

# Train
theta, costs = Logistic_Regression(X, Y, alpha, iterations)

print("Trained theta:\n", theta)

# Plot cost over iterations
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations")
plt.show()





new_data = np.array([[1, 1.3, 6.002, 3.3030]])  # 1 for bias, size=2100, bedrooms=3


z = np.dot(new_data, theta)
predicted_output = 1 / (1 + np.exp(-z))  # sigmoid


print("Predicted Output:", predicted_output [0][0])















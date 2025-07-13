import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('data-for-P-2.txt', header=None, names=['Population', 'Profit'])

# Scatter the data ....
data.plot(kind='scatter', x='Population', y='Profit')

# Insert ones at the start of the data to fit the X0
data.insert(0, 'Ones', 1)

# Separate the data to get data for X and data for Y
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
Y = data.iloc[:,cols-1:cols]

print("X shape:", X.shape)  # should be (m, n)
print("Y shape:", Y.shape)  # should be (m, 1)

"""
The following function execute linear regression with one variable 
using gradient desecent algorithm 
"""

def linear_regression_GD(x, y, alpha, iterations, theta):
    costs = []
    m = int(y.shape[0])
    
    for itetaion in range(1, iterations):
        pred_y = np.dot(x, theta)  # shape (m, 1)
        cost = (1 / (2 * m)) * np.sum((pred_y - y) ** 2)
        costs.append(cost)
        gradient = (1 / m) * np.dot(x.T, (pred_y - y))  # shape (n, 1)
        theta = theta - alpha * gradient
        
        
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(1, iterations)), costs, marker='.', color='red', linestyle='-')  
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()
        
    return theta, costs





theta,_ = linear_regression_GD(X, Y, 0.01, 55, np.array([2, 2]).reshape(2, 1))


# Prediction points
new_x = np.array([5.6, 9.4, 13.8, 20.9, 23])
x_pred = new_x.reshape(-1, 1)
x_pred = np.hstack((np.ones((x_pred.shape[0], 1)), x_pred))
new_y = np.dot(x_pred, theta)

plt.figure(figsize=(8, 6))


orginal_data = pd.read_csv('data-for-P-2.txt', header=None, names=['Population', 'Profit'])

# Separate the data to get data for X and data for Y
cols = orginal_data.shape[1]
orginal_X = orginal_data.iloc[:, 0:cols-1]
orginal_Y = orginal_data.iloc[:,cols-1:cols]


plt.scatter(orginal_X, orginal_Y, marker='o', color='blue', label='Original Data')  # blue points
plt.plot(new_x, new_y, marker='o', color='green', linestyle='-', label='Predicted Line')  # green dashed line
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()




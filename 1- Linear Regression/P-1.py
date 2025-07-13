import numpy as np
import matplotlib.pyplot as plt


X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

Y = np.array([3, 4, 7, 9, 10, 12, 14, 15, 17, 19, 
              20, 22, 24, 25, 27, 30, 31, 33, 34, 36])


# The following function i execute it using the normal equation 
def linear_regression_NE(x, y):
    x = x.reshape(len(x), 1)
    # Add a column of ones to x
    ones = np.ones((x.shape[0], 1))
    x = np.hstack((ones, x))  # x becomes shape (n, d+1)

    # Apply the normal equation
    x_transpose = np.transpose(x)
    xTx = np.dot(x_transpose, x)

    try:
        xTx_inv = np.linalg.inv(xTx)
    except np.linalg.LinAlgError:
        raise ValueError("X^T X is not invertible (singular matrix).")

    theta = np.dot(np.dot(xTx_inv, x_transpose), y)


    return theta


# The following function i execute it using the normal equation 
def linear_regression_GD(x, y, alpha, iterations, theta):
    costs = []
    m = len(y)
        
    x = x.reshape(len(x), 1)
    # Add a column of ones to x
    ones = np.ones((x.shape[0], 1))
    x = np.hstack((ones, x))  # x becomes shape (n, d+1)
    
    x = x.T
    
    
    for itetaion in range(1, iterations):
        pred_y = np.dot(theta.T, x)
    
        cost = (np.sum(pred_y - y)) / 2*m
        costs.append(cost)
        
        theta = theta - ( alpha * ( (np.sum(pred_y - y)) / m ) )
        
    return theta, costs[-1]




num = int(input("Enter 1 Fo Using NE and 2 For using GD: "))
if num > 0:
    if num == 1:
        theta = linear_regression_NE(X, Y)
    elif num == 2:
        theta,_ = linear_regression_GD(X, Y, 0.01, 200, np.array([2, 2]).reshape(2, 1))
    else:
        print("Invalid Number...")
else:
    print("Input Error...")
    
if num in [1, 2]:
    # Prediction points
    new_x = np.array([1, 2, 9, 11, 13])
    x_pred = new_x.reshape(-1, 1)
    x_pred = np.hstack((np.ones((x_pred.shape[0], 1)), x_pred))
    new_y = np.dot(x_pred, theta)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, marker='o', color='blue', label='Original Data')  # blue points
    plt.plot(new_x, new_y, marker='o', color='green', linestyle='-', label='Predicted Line')  # green dashed line
    plt.xlabel('X values')
    plt.ylabel('Y values')
    if num == 1:
        plt.title('Linear Regression using Normal Equation')
    else:
        plt.title('Linear Regression using Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.show()
    


























import math
import numpy as np
import random

class LinearRegression:

    # Convergence threshold
    history_size = 100000

    def __init__(self, w=None, b=None, epsilon=1e-8) -> None:
        self.w = w
        self.b = random.randint(-100, 100) if b is None else b
        
        # Convergence threshold
        self.epsilon = epsilon

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, X, y, alpha=1.0e-5):
        """
        Fits the model to the data

        
        """

        # Initialize weights if not provided
        if self.w is None:
            if len(X.shape) == 1:
                self.w = np.random.rand(-100, 100)
            else:
                self.w = np.random.uniform(-100, 100, X.shape[1])

        if X.shape[0] != y.shape[0]:
            raise Exception('X and y must have same number of rows')

        J_history = []
        p_history = []

        # Begin training
        i = 0
        while True:
            # Check for convergence

            cost = self.cost(X, y)
            if i > 0 and math.isclose(J_history[-1], cost, abs_tol=self.epsilon):
                print(f'Final iteration {i}: Cost = {cost}, w = {self.w}, b = {self.b}')
                break

            dw, db = self.gradient(X, y)
            self.w -= alpha * dw
            self.b -= alpha * db

            # To prevent resource exhaustion
            if i > self.history_size:
                J_history.pop(0)
                p_history.pop(0)
                
            J_history.append(cost)
            p_history.append([self.w, self.b])

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % 1000 == 0:
                print(f'Iteration {i}: Cost = {cost}, w = {self.w}, b = {self.b}')

            i += 1
                
        return self.w, self.b, J_history, p_history
    
    # Function to calculate the cost
    def compute_cost_matrix(self, X, y, verbose=False):
        """
        Computes the gradient for linear regression
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters
        b (scalar)       : model parameter
        verbose : (Boolean) If true, print out intermediate value f_wb
        Returns
        cost: (scalar)
        """
        m = X.shape[0]

        # calculate f_wb for all examples.
        f_wb = X @ self.w + self.b
        # calculate cost
        total_cost = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)

        if verbose:
            print("f_wb:", f_wb)

        return total_cost


    def compute_gradient_matrix(self, X, y):
        """
        Computes the gradient for linear regression

        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters
        b (scalar)       : model parameter
        Returns
        dj_dw (ndarray (n,1)): The gradient of the cost w.r.t. the parameters w.
        dj_db (scalar):        The gradient of the cost w.r.t. the parameter b.

        """
        m, n = X.shape
        f_wb = X @ self.w + self.b
        e = f_wb - y
        dj_dw = (1 / m) * (X.T @ e)
        dj_db = (1 / m) * np.sum(e)

        return dj_db, dj_dw

    def cost(self, X, y):
        """
        J(w, b) = 1/2m * sum((w*X + b - y)^2)

        X: features
        y: target
        w: weights
        """
        m = len(y)
        J = 0
        for i in range(m):
            J += (np.dot(self.w, X[i]) + self.b - y[i])**2

        return J / (2 * m)

    def gradient(self, X, y):
        """
        dJ/dw = 1/m * sum((w*X + b - y)*X)
        dJ/db = 1/m * sum((w*X + b - y))

        X: features
        y: target
        w: weights
        """
        m = len(y)
        dw = 0
        db = 0
        for i in range(m):
            dw += (np.dot(self.w, X[i]) + self.b - y[i]) * X[i]
            db += (np.dot(self.w, X[i]) + self.b - y[i])
        
        return dw/m, db/m

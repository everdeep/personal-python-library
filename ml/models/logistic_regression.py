import math
import numpy as np
import random

class LogisticRegression:

    # Convergence threshold
    history_size = 100000

    def __init__(self, w=None, b=None, epsilon=1e-8) -> None:
        self.w = w
        self.b = random.randint(-100, 100) if b is None else b
        
        # Convergence threshold
        self.epsilon = epsilon

    def _model(self, X):
        return np.dot(X, self.w) + self.b

    def sigmoid(self, Z):
        z = np.clip( Z, -500, 500 )           # protect against overflow
        return 1.0 / (1.0 + np.exp(-z))
        
    def predict(self, X):
        return np.where(self.sigmoid(self._model(X)) >= 0.5, 1, 0)

    def fit(self, X, y, alpha=1.0e-5):
        """
        Performs batch gradient descent
        
        Args:
            X (ndarray (m,n)   : Data, m examples with n features
            y (ndarray (m,))   : target values
            w_in (ndarray (n,)): Initial values of model parameters  
            b_in (scalar)      : Initial values of model parameter
            alpha (float)      : Learning rate
            num_iters (scalar) : number of iterations to run gradient descent
        
        Returns:
            w (ndarray (n,))   : Updated values of parameters
            b (scalar)         : Updated value of parameter 
        """

        # Initialize parameters
        m, n = X.shape
        self.w = np.zeros((n,)) if self.w is None else self.w
       
        # An array to store cost J and w's at each iteration primarily for graphing later
        self.history = []
        
        i = 0
        while True:

            # Check for convergence
            if i > 1 and abs(self.history[-1] - self.history[-2]) < self.epsilon:
                break

            # Calculate the gradient and update the parameters
            db, dw = self.gradient(X, y)   

            # Update Parameters using w, b, alpha and gradient
            self.w -= alpha * dw               
            self.b -= alpha * db               
        
            # Save cost J at each iteration
            if i < self.history_size:      # prevent resource exhaustion 
                self.history.append( self.cost(X, y) )

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % 1000 == 0:
                print(f"Iteration {i:4d}: Cost {self.history[-1]}")
            
            i += 1
            
    def cost(self, X, y):
        """
        Sigmod cost function

        X: (m, n) matrix of features
        y: (m) targets
        
        J(w, b) = 1/m * sum(y*log(w*X + b) + (1-y)*log(1 - w*X + b))
        """
        m = len(y)
        cost = 0
        for i in range(m):
            z = self._model(X[i])
            cost += -y[i] * np.log(self.sigmoid(z)) - (1 - y[i]) * np.log(1 - self.sigmoid(z))
        
        return cost/m


    def gradient(self, X, y):
        """
        Computes the gradient for linear regression 
    
        Args:
            X (ndarray (m,n): Data, m examples with n features
            y (ndarray (m,)): target values

        Returns
            dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
            db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
        """
        m,n = X.shape
        dw = np.zeros((n,))                           #(n,)
        db = 0.

        for i in range(m):
            f_wb_i = self.sigmoid(self._model(X[i]))          #(n,)(n,)=scalar
            err_i  = f_wb_i  - y[i]                       #scalar
            for j in range(n):
                dw[j] += err_i * X[i,j]      #scalar
            db += err_i
            
        return db/m, dw/m

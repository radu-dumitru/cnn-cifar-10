import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)

    def forward(self, X):
        """
        X shape: (N, input_dim)
        Returns: (N, output_dim)
        """
        self.X = X
        out = X @ self.W + self.b
        return out

    def backward(self, d_out, learning_rate=0.01):
        """
        d_out shape: (N, output_dim)
        """
        dW = self.X.T @ d_out
        db = np.sum(d_out, axis=0)
        dX = d_out @ self.W.T

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dX

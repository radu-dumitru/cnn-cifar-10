import numpy as np

class Flatten:
    def forward(self, X):
        """
        Forward pass: reshape input into (N, -1)

        Parameters:
        -----------
        X : numpy array of shape (N, H, W, C)
            Batch of feature maps

        Returns:
        --------
        out : numpy array of shape (N, H*W*C)
        """

        self.input_shape = X.shape
        N = X.shape[0]
        out = X.reshape(N, -1)
        return out

    def backward(self, d_out):
        """
        Backward pass: reshape gradient back into input shape

        Parameters:
        -----------
        d_out : numpy array of shape (N, H*W*C)
            Gradient from next layer

        Returns:
        --------
        dX : numpy array of shape (N, H, W, C)
        """
        return d_out.reshape(self.input_shape)

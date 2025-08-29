import numpy as np

class MaxPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, X):
        self.X = X
        N, H, W, C = X.shape
        h_out = H // self.pool_size
        w_out = W // self.pool_size

        out = np.zeros((N, h_out, w_out, C))
        self.max_pos_h = np.zeros_like(out, dtype=np.int32)
        self.max_pos_w = np.zeros_like(out, dtype=np.int32)

        for n in range(N):
            for i in range(h_out):
                h_start = i * self.pool_size
                h_end = h_start + self.pool_size
                for j in range (w_out):
                    w_start = j * self.pool_size
                    w_end = w_start + self.pool_size
                    for c in range(C):
                        window = X[n, h_start:h_end, w_start:w_end, c]
                        flat_idx = np.argmax(window)
                        dh, dw = divmod(flat_idx, self.pool_size)
                        out[n, i, j, c] = window[dh, dw]
                        self.max_pos_h[n, i, j, c] = h_start + dh
                        self.max_pos_w[n, i, j, c] = w_start + dw

        return out

    def backward(self, d_out):
        N, H, W, C = self.X.shape
        dx = np.zeros((N, H, W, C))
        h_out, w_out = d_out.shape[1:3]

        for n in range(N):
            for i in range (h_out):
                for j in range(w_out):
                    for c in range(C):
                        h_idx = self.max_pos_h[n, i , j, c]
                        w_idx = self.max_pos_w[n, i , j, c]
                        dx[n, h_idx, w_idx, c] = d_out[n, i , j, c]
                        
        return dx





from convolutional import Convolutional
from max_pool import MaxPool
from dense import Dense
from relu import ReLU
from flatten import Flatten

class Network:
    def __init__(self):
        self.layers = [
            Convolutional(num_filters=8, kernel_size=(3,3,3), stride=1, padding=1),
            ReLU(),
            MaxPool(pool_size=2),
            Flatten(),
            Dense(32*16*16, 128),
            ReLU(),
            Dense(128, 10)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out, lr):
        for layer in reversed(self.layers):
            if isinstance(layer, (Dense, Convolutional)):
                d_out = layer.backward(d_out, lr)
            else:
                d_out = layer.backward(d_out)
                
        return d_out
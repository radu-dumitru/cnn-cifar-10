import numpy as np

class Convolutional:
	def __init__(self, num_filters, kernel_size, stride=1, padding=0):
		self.num_filters = num_filters
		self.k_h, self.k_w, self.c_in = kernel_size
		self.stride = stride
		self.padding = padding

		self.weights = np.random.randn(self.k_h, self.k_w, self.c_in, self.num_filters)
		self.bias = np.zeros(self.num_filters)

	def forward(self, input):
		self.input = input
		batch_size, h_in, w_in, _ = input.shape

		self.input_padded = np.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

		h_out = (h_in + 2 * self.padding - self.k_h) // self.stride + 1
		w_out = (w_in + 2 * self.padding - self.k_w) // self.stride + 1

		out = np.zeros((batch_size, h_out, w_out, self.num_filters))

		for n in range(batch_size):
			for f in range(self.num_filters):
				for i in range(h_out):
					for j in range (w_out):
						h_start = i * self.stride
						h_end = h_start + self.k_h
						w_start = j * self.stride
						w_end = w_start + self.k_w

						patch = self.input_padded[n, h_start:h_end, w_start:w_end, :]
						out[n, i, j, f] = np.sum(patch * self.weights[:, :, :, f]) + self.bias[f]

		self.out = out
		return out

	def backward(self, d_out, learning_rate=0.01):
		batch_size, h_out, w_out, _ = d_out.shape

		dx_padded = np.zeros_like(self.input_padded)
		dw = np.zeros_like(self.weights)
		db = np.zeros_like(self.bias)

		for n in range(batch_size):
			for f in range(self.num_filters):
				for i in range(h_out):
					for j in range(w_out):
						h_start = i * self.stride
						h_end = h_start + self.k_h
						w_start = j * self.stride
						w_end = w_start + self.k_w

						patch = self.input_padded[n, h_start:h_end, w_start:w_end, :]

						dw[:, :, :, f] += patch * d_out[n, i, j, f]
						dx_padded[n, h_start:h_end, w_start:w_end, :] += self.weights[:, :, :, f] * d_out[n, i, j, f]
						db[f] += d_out[n, i, j, f]

		if self.padding > 0:
			dx = dx_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
		else:
			dx = dx_padded

		self.weights -= learning_rate * dw
		self.bias -= learning_rate * db

		return dx


	

	

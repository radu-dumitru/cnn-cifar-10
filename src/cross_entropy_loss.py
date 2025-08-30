import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.logits = None
        self.labels = None
        self.probs = None

    def forward(self, logits, labels):
        """
        logits: (N - batch size, C - classes)
        labels: (N,)
        """
        self.logits = logits
        self.labels = labels

        # shift logits for numerical stability
        shifted_logits = logits - np.max(logits, axis = 1, keepdims = True)
        exp_scores = np.exp(shifted_logits)
        self.probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

        # pick the correct class probability
        N = logits.shape[0]
        correct_logprobs = -np.log(self.probs[np.arange(N), labels])
        loss = np.sum(correct_logprobs) / N
        
        return loss

    def backward(self):
        N = self.logits.shape[0]
        d_logits = self.probs.copy()
        d_logits[np.arange(N), self.labels] -= 1
        d_logits /= N
        
        return d_logits 
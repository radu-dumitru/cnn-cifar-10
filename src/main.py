import numpy as np
from data_loader import DataLoader
from cross_entropy_loss import CrossEntropyLoss
from network import Network

data_loader = DataLoader()
(x_train, y_train), (x_test, y_test) = data_loader.load_data()

network = Network()
cross_entropy_loss = CrossEntropyLoss()

lr = 0.01
batch_size = 64
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for x_batch, y_batch in data_loader.get_batches(x_train, y_train, batch_size=batch_size, shuffle=True):
        logits = network.forward(x_batch)
        loss = cross_entropy_loss.forward(logits, y_batch)
        total_loss += loss * x_batch.shape[0]

        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == y_batch)
        total += x_batch.shape[0]

        d_logits = cross_entropy_loss.backward()
        network.backward(d_logits, lr)

    train_loss = total_loss / total
    train_acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

total_loss = 0
correct = 0
total = 0

for x_batch, y_batch in data_loader.get_batches(x_test, y_test, batch_size=batch_size, shuffle=False):
    logits = network.forward(x_batch)
    loss = cross_entropy_loss.forward(logits, y_batch)
    total_loss += loss * x_batch.shape[0]

    preds = np.argmax(logits, axis=1)
    correct += np.sum(preds == y_batch)
    total += x_batch.shape[0]

test_loss = total_loss / total
test_acc = correct / total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
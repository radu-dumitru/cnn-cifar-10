from data_loader import DataLoader
from convolutional import Convolutional

data_loader = DataLoader()
(x_train, y_train), (x_test, y_test) = data_loader.load_data()

conv = Convolutional(num_filters=8, kernel_size=(3,3,3), stride=1, padding=1)

for x_batch, y_batch in data_loader.get_batches(x_train, y_train, batch_size=64):
    out = conv.forward(x_batch)
    print(out.shape)
    break
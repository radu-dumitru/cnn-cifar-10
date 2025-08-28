import pickle
import os
import urllib.request
import tarfile
import numpy as np

class DataLoader:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.data_dir = os.path.join(self.project_root, "cifar-10-batches-py")
        self.archive_name = os.path.join(self.project_root, "cifar-10-python.tar.gz")

    def download_and_extract(self):
        if os.path.exists(self.data_dir):
            print(f"CIFAR-10 data already exists in ./{self.data_dir}")
            return
        
        if not os.path.exists(self.archive_name):
            print(f"Downloading {self.cifar_url} ...")
            urllib.request.urlretrieve(self.cifar_url, self.archive_name)
            print("Download complete.")

        print("Extracting archive...")
        with tarfile.open(self.archive_name, "r:gz") as tar:
            tar.extractall(path=self.project_root)
        print("Extraction complete.")

        # Remove the archive file after extraction
        if os.path.exists(self.archive_name):
            os.remove(self.archive_name)
            print(f"Removed archive file: {self.archive_name}")

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    def load_data(self):
        self.download_and_extract()

        x_train, y_train = [], []
        for i in range(1, 6):
            batch = self.unpickle(os.path.join(self.data_dir, f"data_batch_{i}"))
            x_train.append(batch[b'data'])
            y_train.extend(batch[b'labels'])

        x_train = np.vstack(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
        y_train = np.array(y_train)

        test_batch = self.unpickle(os.path.join(self.data_dir, "test_batch"))
        x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_test = np.array(test_batch[b'labels'])

        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        return (x_train, y_train), (x_test, y_test)

    def get_batches(self, x, y, batch_size=64, shuffle=True):
        n = x.shape[0]
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield x[batch_idx], y[batch_idx]
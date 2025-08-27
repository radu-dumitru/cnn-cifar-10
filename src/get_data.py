import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import urllib.request
import tarfile

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_DIR = "cifar-10-batches-py"
ARCHIVE_NAME = "cifar-10-python.tar.gz"

def download_and_extract():
    if os.path.exists(DATA_DIR):
        print(f"CIFAR-10 data already exists in ./{DATA_DIR}")
        return
    
    if not os.path.exists(ARCHIVE_NAME):
        print(f"Downloading {CIFAR_URL} ...")
        urllib.request.urlretrieve(CIFAR_URL, ARCHIVE_NAME)
        print("Download complete.")

    print("Extracting archive...")
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall()
    print("Extraction complete.")

    # Remove the archive file after extraction
    if os.path.exists(ARCHIVE_NAME):
        os.remove(ARCHIVE_NAME)
        print(f"Removed archive file: {ARCHIVE_NAME}")

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

download_and_extract()
batch = unpickle(os.path.join(DATA_DIR, "data_batch_1"))

# Get the first image
img_flat = batch[b'data'][0]
img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)

# Show image in a window
plt.imshow(img)
plt.title(f"Label: {batch[b'labels'][0]}")
plt.show()


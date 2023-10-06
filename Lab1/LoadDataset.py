import urllib.request
import gzip
import numpy as np

urlMNIST = 'http://yann.lecun.com/exdb/mnist/'
savePath = 'Lab1/Dataset/'
filesName = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
         't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

def download_MNIST(url):
    for file in filesName:
        urllib.request.urlretrieve(url + file, savePath + file)

def read_mnist(images_file, labels_file):
    with gzip.open(savePath + labels_file, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(savePath + images_file, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

if __name__ == '__main__':
    train_images, train_labels = read_mnist(filesName[0], filesName[1])
    test_images, test_labels = read_mnist(filesName[2], filesName[3])

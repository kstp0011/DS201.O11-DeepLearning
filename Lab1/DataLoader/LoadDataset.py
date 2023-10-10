# import urllib.request
# import gzip
# import numpy as np
# from torch.utils.data import Dataset, DataLoader

# urlMNIST = 'http://yann.lecun.com/exdb/mnist/'
# savePath = 'Lab1/DataLoader/Dataset/'
# filesName = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
#          't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

# def read_mnist(images_file, labels_file):
#     with gzip.open(savePath + labels_file, 'rb') as f:
#         labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
#     with gzip.open(savePath + images_file, 'rb') as f:
#         images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
#     return images, labels

# class MnistDataset(Dataset):
#     def __init__(self, images, labels):
#         self.images = images
#         self.labels = labels
#         self.len = len(labels)

#     def __getitem__(self, index):
#         return self.images[index], self.labels[index]

#     def __len__(self):
#         return self.len

# def load_dataset(train_batch_size = 64, test_batch_size = 1000):
#     train_images, train_labels = read_mnist(filesName[0], filesName[1])
#     test_images, test_labels = read_mnist(filesName[2], filesName[3])
#     train_loader = DataLoader(MnistDataset(train_images, train_labels), batch_size=train_batch_size, shuffle=True)
#     test_loader = DataLoader(MnistDataset(test_images, test_labels), batch_size=test_batch_size, shuffle=True)
    # return train_loader, test_loader

import torchvision
import torch

Path = 'Lab1/DataLoader/Dataset/'

def load_dataset(train_batch_size = 64, test_batch_size = 1000):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(Path, train=True, download=True,
                                      transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                             (0.1307,), (0.3081,))
                                      ])),
                                      batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(Path, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
                                    atch_size=test_batch_size, shuffle=True)

if __name__ == '__main__':
    # download_MNIST(urlMNIST)
    train_loader, test_loader = load_dataset()


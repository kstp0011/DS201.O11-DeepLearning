import torch
from torch.utils.data import Dataset, DataLoader
import os
import struct
import numpy as np


class MNISTDataset(Dataset):
    def __init__(self, data_path, labels_path):
        with open(data_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            self.data = np.fromfile(
                file, dtype=np.uint8).reshape(size, 1, rows, cols)

        with open(labels_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            self.labels = np.fromfile(file, dtype=np.uint8)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_dataloader(args):
    train_dataset = MNISTDataset(args["train_images_path"], args["train_labels_path"])
    test_dataset = MNISTDataset(args["test_images_path"], args["test_labels_path"])

    train_loader = DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args["batch_size"], shuffle=False)

    return train_loader, test_loader
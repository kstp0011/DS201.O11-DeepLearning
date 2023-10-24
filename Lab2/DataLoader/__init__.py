from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
import idx2numpy
import torchvision.transforms as transforms


class MNISTCLOTHESDataset:
    def __init__(self: object, image_path: str, label_path: str, transform: None) -> None:
        self.data = idx2numpy.convert_from_file(image_path)
        self.target = idx2numpy.convert_from_file(label_path)
        self.transform = transform

    def __getitem__(self: object, index: int) -> tuple:
        image = self.data[index]
        label = self.target[index]
        if self.transform:
            image = self.transform(image)
        return self.data[index], self.target[index]

    def __len__(self: object) -> int:
        return len(self.target)


class load_data:
    def __init__(self, train_images, train_label, test_images, test_label, train_batch, test_batch) -> None:
        self.train_image_path = train_images
        self.train_label_path = train_label
        self.test_image_path = test_images
        self.test_label_path = test_label
        self.train_batch = train_batch
        self.test_batch = test_batch

    def get_dataloader(self, train=True):
        if train:
            image_path = self.train_image_path
            label_path = self.train_label_path
            batch_size = self.train_batch
        else:
            image_path = self.test_image_path
            label_path = self.test_label_path
            batch_size = self.test_batch

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = MNISTCLOTHESDataset(image_path, label_path, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
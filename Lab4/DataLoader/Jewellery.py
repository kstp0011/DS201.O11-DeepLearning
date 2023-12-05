import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# Define transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'data/Jewellery-Classification-master/dataset'
batch_size = 4
validation_split = .2
shuffle_dataset = False
random_seed = 42

# Load training data and apply transformations
train_data = datasets.ImageFolder(os.path.join(data_dir, 'training'), data_transforms['train'])

# Creating data indices for training and validation splits
dataset_size = len(train_data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                sampler=valid_sampler)

# Load test data and apply transformations
test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# Define dataloaders
Jew_dataloaders = {
    'train': train_loader,
    'val': validation_loader,
    'test': test_loader
}

# Calculate sizes of train, validation, and test sets
Jew_dataset_sizes = {
    'train': len(train_sampler),
    'val': len(valid_sampler),
    'test': len(test_data)
}

# Get class names
Jew_class_names = train_data.classes
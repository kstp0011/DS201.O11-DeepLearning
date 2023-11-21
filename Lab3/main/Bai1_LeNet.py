from torch.utils.data import DataLoader
from torch import optim
import torch
from Config import args
from DataLoader import create_dataloader
from model import LeNet
from task_utils import train, evaluate, get_confusion_matrix


def B1_Lenet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoaders
    train_loader, test_loader = create_dataloader(args)

    # Create model
    model = LeNet().to(device)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=args["lr"])

    # Train model
    for epoch in range(args["epoch"]):
        train(model, train_loader, optimizer, epoch, args["epoch"])

    # Evaluate model
    metrics = evaluate(model, test_loader, device)
    print("Result:")
    print(metrics)

    print("Confusion matrix:")
    num_classes = 10  # For MNIST, we have 10 classes (digits 0-9)
    conf_matrix = get_confusion_matrix(model, test_loader, device, num_classes)
    print(conf_matrix)

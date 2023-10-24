from Config import train_image_path, train_label_path, test_image_path, test_label_path, train_batch, test_batch, n_epochs, learning_rate, momentum
from DataLoader import load_data
from Model import MLP
from Task_utils import train, test, init_weights_W_GU_b_Z
from Metric import classification_Report
import torch

def main():
    # Load data
    data_loader = load_data(train_image_path, train_label_path, test_image_path, test_label_path, train_batch, test_batch)
    train_loader = data_loader.get_dataloader(train=True)
    test_loader = data_loader.get_dataloader(train=False)

    # Define model
    model = MLP()

    # Initialize weights and bias to ones
    init_weights_W_GU_b_Z(model)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    train(model, train_loader, optimizer, criterion, n_epochs, train_batch)

    # Test model
    test(model, test_loader, criterion)

    # Classification report
    classification_Report(model, test_loader)

if __name__ == '__main__':
    main()


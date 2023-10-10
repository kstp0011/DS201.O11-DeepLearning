from DataLoader import load_dataset
from Model import MnistModel, train
from Metric import validate

def main():
    # Load dataset
    train_loader, test_loader = load_dataset()

    # Create model
    network = MnistModel()

    # Train model
    # optimizer usse adam
    # criterion use cross entropy loss
    # num_epochs use 10
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    n_epochs = 10
    train(network, train_loader)

    # Validate model
    accuracy = validate(network, test_loader)
    print('Accuracy: ', accuracy)

if __name__ == '__main__':
    main()


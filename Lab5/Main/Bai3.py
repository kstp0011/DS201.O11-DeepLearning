from LoadData import UITvsfc_dataset, dataset_loader
from Models import gru_model
from task_utils import train_early_stopping, evaluation
import torch

config = {
    "path": "data/csv-files",
    "batch_size": 64,
    "num_epochs": 100,
    "lr": 0.001,
    "patience": 3,
    "hidden_size": 128,
    "embedding_dim": 300,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def main():
    # Load data
    train_dataset = UITvsfc_dataset(config["path"], "train")
    dev_dataset = UITvsfc_dataset(config["path"], "dev")
    test_dataset = UITvsfc_dataset(config["path"], "test")

    # Create dataloader
    train_loader = dataset_loader(train_dataset, config["batch_size"])
    dev_loader = dataset_loader(dev_dataset, config["batch_size"])
    test_loader = dataset_loader(test_dataset, config["batch_size"])

    # Create model
    model = gru_model(input_size=len(train_dataset.vocab),
                        hidden_size=config["hidden_size"],
                        output_size=len(train_dataset.label_vocab),
                        embedding_dim=config["embedding_dim"])

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    train_early_stopping(model, train_loader, dev_loader, config["num_epochs"],
                         optimizer, config["device"], config["patience"])

    # Evaluation
    evaluation(model, test_loader, config["device"], criterion)
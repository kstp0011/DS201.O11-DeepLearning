from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import torch


def evaluate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode

    # Initialize lists to store true and predicted labels
    true_labels = []
    pred_labels = []

    # Disable gradient calculation for evaluation to save memory and computations
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get the prediction (the class with the highest probability)
            _, predicted = torch.max(outputs, 1)

            # Extend the lists
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # Calculate metrics using true and predicted labels
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro')

    # Return the metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'F1-macro': f1_macro
    }

    return metrics


def get_confusion_matrix(model, dataloader, device, num_classes):
    model.eval()  # Set the model to evaluation mode

    # Initialize lists to store true and predicted labels
    true_labels = []
    pred_labels = []

    # Disable gradient calculation for evaluation to save memory and computations
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get the prediction (the class with the highest probability)
            _, predicted = torch.max(outputs, 1)

            # Extend the lists
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
    return cm

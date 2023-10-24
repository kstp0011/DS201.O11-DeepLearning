from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import torch


def evaluate(output, target):
    # Convert output probabilities to predicted class
    pred = output.argmax(dim=1)

    acc = accuracy_score(target, pred)
    precision = precision_score(target, pred, average='macro', zero_division=1)
    rec = recall_score(target, pred, average='macro', zero_division=1)
    f1 = f1_score(target, pred, average='macro', zero_division=1)
    return acc, precision, rec, f1


def print_confusion_matrix(output, target):
    pred = output.argmax(dim=1)
    print(confusion_matrix(target, pred))


def classification_Report(model, test_loader, criterion, ntest=1):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float(), target.long()
            output = model(data)
            pred = output.argmax(dim=1)
            print(classification_report(target, pred))
            print()
            ntest -= 1
            if ntest <= 0:
                break
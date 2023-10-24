from Metric import evaluate, print_confusion_matrix
import torch

def test(model, test_loader, criterion, ntest=1):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float(), target.long()
            output = model(data)
            loss = criterion(output, target)
            acc, pred, rec, f1 = evaluate(output, target)
            print_confusion_matrix(output, target)
            print(
                f'Accuracy: {acc}, Precision: {pred}, Recall: {rec}, F1: {f1}')
            ntest -= 1
            if ntest <= 0:
                break
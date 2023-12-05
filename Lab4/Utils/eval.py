from sklearn.metrics import classification_report
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, dataloaders, class_names, phase='test'):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_pred.extend(preds.view(-1).tolist())
            y_true.extend(labels.view(-1).tolist())

    print(classification_report(y_true, y_pred, target_names=class_names))
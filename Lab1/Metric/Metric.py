# import sklearn.metrics as metrics
import torch
import numpy as np


def test(model, test_loader):
    y_pred = []
    y_test = []
    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(labels)):
            y_pred.append(predicted[j])
            y_test.append(labels[j])
    return y_pred, y_test


def classification_report(y_true, y_pred, output_dict=False):
    """
    Generates a classification report for the given true labels and predicted labels,
    with macro-average and weighted-average metrics.

    Args:
      y_true: The true labels of the test set.
      y_pred: The predicted labels of the test set.
      output_dict: Whether to return the metrics as a dictionary or a string.

    Returns:
      A dictionary containing the metrics for each class or a string containing the
      classification report in a tabular format.
    """

    # Calculate the confusion matrix.
    confusion_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_true))))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1

    # Calculate the accuracy, precision, recall, and F1 score for each class.
    metrics = {}
    for i in range(len(np.unique(y_true))):
        metrics[i] = {
            "accuracy": confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]),
            "precision": confusion_matrix[i, i] / np.sum(confusion_matrix[:, i]),
            "recall": confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]),
            "f1-score": (2 * confusion_matrix[i, i])
            / (np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i])),
        }

    # Calculate the macro-average and weighted-average metrics.
    macro_avg = {
        "accuracy": np.mean(
            [metrics[i]["accuracy"] for i in range(len(np.unique(y_true)))]
        ),
        "precision": np.mean(
            [metrics[i]["precision"] for i in range(len(np.unique(y_true)))]
        ),
        "recall": np.mean(
            [metrics[i]["recall"] for i in range(len(np.unique(y_true)))]
        ),
        "f1-score": np.mean(
            [metrics[i]["f1-score"] for i in range(len(np.unique(y_true)))]
        ),
    }

    # Return the metrics as a dictionary or a string.
    if output_dict:
        return metrics, macro_avg
    else:
        report = ""
        for i in range(len(np.unique(y_true))):
            report += "Class {}: {}\n".format(i, metrics[i])
        report += "\nMacro average: {}\n".format(macro_avg)
        return report


def print_evaluation_each_class(
    classwise_accuracy, classwise_precision, classwise_recall, classwise_f1
):
    print("-" * 60)
    print("Evaluation Results")
    print("-" * 60)
    print("| Class | Accuracy | Precision | Recall |   F1.  |")
    print("|-------|----------|-----------|--------|--------|")
    for i in range(10):
        # print("|   {}   |  {:.4f}  |  {:.4f}   | {:.4f} | {:.4f} |".format(i, classwise_accuracy[i], classwise_precision[i], classwise_recall[i], classwise_f1[i]))
        print("|   {}   |".format(i), end="")
        print(f"  {classwise_accuracy[i]:.4f}  " + "|", end="", sep="")
        print(f"  {classwise_precision[i]:.4f}   " + "|", end="", sep="")
        print(f" {classwise_recall[i]:.4f} " + "|", end="", sep="")
        print(f" {classwise_f1[i]:.4f} " + "|", end="", sep="")
        print()
    print("-" * 60)
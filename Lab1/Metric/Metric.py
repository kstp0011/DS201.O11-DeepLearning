import torch

def evaluate(model, test_loader):
    classwise_correct = [0] * 10
    classwise_total = [0] * 10

    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for j in range(len(labels)):
            if labels[j] == i:
                classwise_total[i] += 1
                if predicted[j] == labels[j]:
                    classwise_correct[i] += 1

    classwise_accuracy = [classwise_correct[i] / classwise_total[i] for i in range(10)]
    classwise_precision = [classwise_correct[i] / (classwise_correct[i] + classwise_total[i] - classwise_correct[i]) for i in range(10)]
    classwise_recall = [classwise_correct[i] / classwise_total[i] for i in range(10)]
    classwise_f1 = [2 * classwise_precision[i] * classwise_recall[i] / (classwise_precision[i] + classwise_recall[i]) for i in range(10)]

    return classwise_accuracy, classwise_precision, classwise_recall, classwise_f1

def print_evaluation_each_class(classwise_accuracy, classwise_precision, classwise_recall, classwise_f1):
    print("-" * 60)
    print("Evaluation Results")
    print("-" * 60)
    print("| Class | Accuracy | Precision | Recall |   F1.  |")
    print("|-------|----------|-----------|--------|--------|")
    for i in range(10):
        # print("|   {}   |  {:.4f}  |  {:.4f}   | {:.4f} | {:.4f} |".format(i, classwise_accuracy[i], classwise_precision[i], classwise_recall[i], classwise_f1[i]))
        print("|   {}   |".format(i), end='')
        print(f"  {classwise_accuracy[i]:.4f}  "+'|', end='', sep='')
        print(f"  {classwise_precision[i]:.4f}   "+'|', end='', sep='')
        print(f" {classwise_recall[i]:.4f} "+'|', end='', sep='')
        print(f" {classwise_f1[i]:.4f} "+'|', end='', sep='')
        print()
    print("-" * 60)

def evaluate_all_class(classwise_accuracy, classwise_precision, classwise_recall, classwise_f1):
    overall_accuracy = sum(classwise_accuracy) / len(classwise_accuracy)
    overall_precision = sum(classwise_precision) / len(classwise_precision)
    overall_recall = sum(classwise_recall) / len(classwise_recall)
    overall_f1 = sum(classwise_f1) / len(classwise_f1)

    return overall_accuracy, overall_precision, overall_recall, overall_f1
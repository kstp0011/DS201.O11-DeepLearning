from DataLoader import load_dataset, download_mnist, plot_examples
from Model import Mnist_Model_B1, train, Mnist_Model_B2
from Metric import test, classification_report, print_evaluation_each_class

def main():
    # Load dataset
    dataset = download_mnist()
    train_loader, test_loader = load_dataset(dataset)

    # check dataset
    # examples = enumerate(train_loader)
    # batch_idx, (images, labels) = next(examples)
    # images_=images[0:10]
    # labels_=labels[0:10]
    # plot_examples(images_, labels_, rows=2);


    # Bai 1
    print("Bài 1: Xây dựng mô hình 1-layer MLP và hàm Softmax làm activation function.")

    # Create model
    network_1, optim_1 = Mnist_Model_B1()

    # Train model
    train(network_1, train_loader, optim_1)

    # Evaluate model
    y_pred, y_test = test(network_1, test_loader)
    report_each_class, macro_avg = classification_report(y_test, y_pred, output_dict=True)
    classwise_accuracy = [report_each_class[i]["accuracy"] for i in range(10)]
    classwise_precision = [report_each_class[i]["precision"] for i in range(10)]
    classwise_recall = [report_each_class[i]["recall"] for i in range(10)]
    classwise_f1 = [report_each_class[i]["f1-score"] for i in range(10)]
    print_evaluation_each_class(
        classwise_accuracy, classwise_precision, classwise_recall, classwise_f1
    )
    print("Overall Accuracy: {:.4f}".format(macro_avg["accuracy"]))
    print("Overall Precision: {:.4f}".format(macro_avg["precision"]))
    print("Overall Recall: {:.4f}".format(macro_avg["recall"]))
    print("Overall F1-score: {:.4f}".format(macro_avg["f1-score"]))


    # Bai 2
    print("")
    print("Bài 2: Xây dựng mô hình 3-layer MLP, hàm ReLU và hàm softmax làm activation function.")

    # Create model
    network_2, optim_2 = Mnist_Model_B2()

    # Train model
    train(network_2, train_loader, optim_2)

    # Evaluate model
    y_pred, y_test = test(network_2, test_loader)
    report_each_class, macro_avg = classification_report(y_test, y_pred, output_dict=True)
    classwise_accuracy = [report_each_class[i]["accuracy"] for i in range(10)]
    classwise_precision = [report_each_class[i]["precision"] for i in range(10)]
    classwise_recall = [report_each_class[i]["recall"] for i in range(10)]
    classwise_f1 = [report_each_class[i]["f1-score"] for i in range(10)]
    print_evaluation_each_class(
        classwise_accuracy, classwise_precision, classwise_recall, classwise_f1
    )
    print("Overall Accuracy: {:.4f}".format(macro_avg["accuracy"]))
    print("Overall Precision: {:.4f}".format(macro_avg["precision"]))
    print("Overall Recall: {:.4f}".format(macro_avg["recall"]))
    print("Overall F1-score: {:.4f}".format(macro_avg["f1-score"]))

if __name__ == '__main__':
    main()
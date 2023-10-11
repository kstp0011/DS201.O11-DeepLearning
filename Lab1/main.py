from DataLoader import load_dataset, download_mnist, plot_examples
from Model import Mnist_Model_B1, train, Mnist_Model_B2
from Metric import evaluate, print_evaluation_each_class, evaluate_all_class

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
    classwise_accuracy, classwise_precision, classwise_recall, classwise_f1 = evaluate(network_1, test_loader)
    print_evaluation_each_class(classwise_accuracy, classwise_precision, classwise_recall, classwise_f1)
    overall_accuracy, overall_precision, overall_recall, overall_f1 = evaluate_all_class(classwise_accuracy, classwise_precision, classwise_recall, classwise_f1)
    print("Overall Accuracy: {:.4f}".format(overall_accuracy))
    print("Overall Precision: {:.4f}".format(overall_precision))
    print("Overall Recall: {:.4f}".format(overall_recall))
    print("Overall F1: {:.4f}".format(overall_f1))


    # Bai 2
    print("")
    print("Bài 2: Xây dựng mô hình 3-layer MLP, hàm ReLU và hàm softmax làm activation function.")

    # Create model
    network_2, optim_2 = Mnist_Model_B2()

    # Train model
    train(network_2, train_loader, optim_2)

    # Evaluate model
    classwise_accuracy, classwise_precision, classwise_recall, classwise_f1 = evaluate(network_2, test_loader)
    print_evaluation_each_class(classwise_accuracy, classwise_precision, classwise_recall, classwise_f1)
    overall_accuracy, overall_precision, overall_recall, overall_f1 = evaluate_all_class(classwise_accuracy, classwise_precision, classwise_recall, classwise_f1)
    print("Overall Accuracy: {:.4f}".format(overall_accuracy))
    print("Overall Precision: {:.4f}".format(overall_precision))
    print("Overall Recall: {:.4f}".format(overall_recall))
    print("Overall F1: {:.4f}".format(overall_f1))

if __name__ == '__main__':
    main()
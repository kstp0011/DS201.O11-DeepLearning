from DataLoader import load_dataset, download_mnist, plot_examples
from Model import Mnist_Model_B1, train
from Metric import evaluate, print_evaluation_each_class, evaluate_all_class

def main():
    # Load dataset
    dataset = download_mnist()
    train_loader, test_loader = load_dataset(dataset)

    # Plot examples
    # examples = enumerate(train_loader)
    # batch_idx, (images, labels) = next(examples)
    # images_=images[0:10]
    # labels_=labels[0:10]
    # plot_examples(images_, labels_, rows=2);

    # Create model
    network, optim = Mnist_Model_B1()

    # Train model
    train(network, train_loader, optim)

    # Evaluate model
    classwise_accuracy, classwise_precision, classwise_recall, classwise_f1 = evaluate(network, test_loader)
    print_evaluation_each_class(classwise_accuracy, classwise_precision, classwise_recall, classwise_f1)
    overall_accuracy, overall_precision, overall_recall, overall_f1 = evaluate_all_class(classwise_accuracy, classwise_precision, classwise_recall, classwise_f1)
    print("Overall Accuracy: {:.4f}".format(overall_accuracy))

if __name__ == '__main__':
    main()


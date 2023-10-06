from Lab1 import download_MNIST, read_mnist, urlMNIST, filesName

download_MNIST(urlMNIST)
train_images, train_labels = read_mnist(filesName[0], filesName[1])
test_images, test_labels = read_mnist(filesName[2], filesName[3])

print(train_images.shape)
print(train_labels.shape)
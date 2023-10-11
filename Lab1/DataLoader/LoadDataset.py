from tqdm import tqdm
import requests
import gzip
import os
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset


data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
    ]
)

class MNISTCustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        image = self.transform(np.array(image))
        return image, label
    def __len__(self):
        return len(self.labels)


def clear_files():
    files = ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]

    # Move the files to the dataset directory
    for file in files:
        # Check if the dataset directory exists
        if not os.path.exists("Lab1/DataLoader/dataset"):
            # Create the dataset directory
            os.makedirs("Lab1/DataLoader/dataset")

        # Move the file to the dataset directory
        os.rename(file, os.path.join("Lab1/DataLoader/dataset", file))

def download_mnist():
    url_root = 'http://yann.lecun.com/exdb/mnist'

    file_dict={
        'train_images':'train-images-idx3-ubyte.gz',
        'train_labels':'train-labels-idx1-ubyte.gz',
        'test_images':'t10k-images-idx3-ubyte.gz',
        'test_labels':'t10k-labels-idx1-ubyte.gz'
    }

    if file_dict is not None:
        mnist_data=list()
        try:
            for i, key in enumerate(file_dict.keys()):    
                fname = file_dict[key]
                url = os.path.join(url_root,fname)                

                isExist = os.path.exists(fname)
                if not isExist:
                    response = requests.get(url, stream=True)
                    fsize=len(response.content)
                    print(url)
                    with open(fname, 'wb') as fout:
                        for data in tqdm(response.iter_content(), desc =fname, total=fsize):
                            fout.write(data)
                
                with gzip.open(fname, "rb") as f_in:                
                    if fname.find('idx3') != -1:        
                        mnist_data.append(np.frombuffer(f_in.read(), np.uint8, offset=16).reshape(-1, 28, 28)) #if images        
                    else:                               
                        mnist_data.append(np.frombuffer(f_in.read(), np.uint8, offset=8))  #if labels
            clear_files()
            #return mnist_data in a list format ==> [[train_images], [train_labels], [test_images], [test_labels]] 
            return mnist_data
        except Exception as e:
            print("Something went wrong:", e)
    else:
        print("file_dict cannot be None")

def load_dataset(dataset, train_batch_size=64, test_batch_size=1000):
    train_images=dataset[0]
    train_labels=dataset[1]
    test_images=dataset[2]
    test_labels=dataset[3]
    train_dataset = MNISTCustomDataset(train_images, train_labels,transform=data_transform)
    test_dataset = MNISTCustomDataset(test_images, test_labels,transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
    return train_loader, test_loader

def plot_examples(images, labels, rows=1):   #by default set rows=1
    fig = plt.figure()
    columns = len(images) // (rows)
    for i, (img, lbl) in enumerate(zip(images, labels)):
        if i < columns*rows:
            ax = fig.add_subplot(rows, columns, i+1)
            ax.imshow(img.reshape(28, 28),cmap="gray")
            ax.set_xticks([]) #set empty label for x axis
            ax.set_yticks([]) #set empty label for y axis
            ax.set_title("label: {}".format(lbl.item()))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataset= download_mnist()
    train_loader, test_loader = load_dataset(dataset)
    examples = enumerate(train_loader)
    batch_idx, (images, labels) = next(examples)
    images_=images[0:10]
    labels_=labels[0:10]
    plot_examples(images_, labels_, rows=2)
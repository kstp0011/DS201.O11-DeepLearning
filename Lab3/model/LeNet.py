import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # First convolutional layer: 1 input channel, 6 output channels, kernel size 5, padding 2
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # Second convolutional layer: 6 input channels, 16 output channels, kernel size 5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # First average pooling layer: kernel size 2, stride 2
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Second average pooling layer: kernel size 2, stride 2
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # First fully connected layer: 5*5*16 input features, 120 output features
        self.fc1 = nn.Linear(16*5*5, 120)
        # Second fully connected layer: 120 input features, 84 output features
        self.fc2 = nn.Linear(120, 84)
        # Third fully connected layer: 84 input features, 10 output features
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.avgpool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

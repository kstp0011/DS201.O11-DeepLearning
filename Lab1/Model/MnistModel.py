import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

def Mnist_Model_B1(lr=0.01):
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    return model, optimizer
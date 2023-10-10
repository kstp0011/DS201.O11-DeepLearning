import torch.nn as nn

class Mnist_model(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(784, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_stack(x)
        return x
    
def MnistModel():
    model = Mnist_model()
    return model
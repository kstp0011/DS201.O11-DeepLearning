import torch
from torch import nn
import torch.optim as optim
from .train import train
from .test import test

class MLP(nn.Module):
  def __init__(self:object) -> None:
      super(MLP, self).__init__()
      self.fc1 = nn.Linear(784,512)
      self.fc2= nn.Linear(512,256)
      self.fc3 = nn.Linear(256,10)
      self.relu = nn.ReLU()
  def forward(self:object, x:torch.Tensor) -> torch.Tensor:
      x = x.view(-1, 784)
      x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
      x = torch.softmax(x, dim=1)
      return x

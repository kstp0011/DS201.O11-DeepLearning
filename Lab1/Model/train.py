from torch.nn import CrossEntropyLoss
import torch

def train(self, dataloader, optimizer=None, criterion=CrossEntropyLoss(), num_epochs=10):
    if optimizer is None:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        self.train()
        for images, labels in dataloader:
            optimizer.zero_grad()
            output = self(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
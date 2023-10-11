import torch.nn as nn

def train(model, train_loader, optimizer, epochs=10):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch: {} Loss: {:.4f}".format(epoch + 1, loss.item()))
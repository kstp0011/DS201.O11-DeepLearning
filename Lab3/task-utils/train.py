from torch import nn


def train(model, data_loader, optimizer, epoch, n_epoch):
    model.train()
    for i, (images, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(
                f"Epoch {epoch}/{n_epoch}, Step {i}/{len(data_loader)}, Loss: {loss.item():.4f}")

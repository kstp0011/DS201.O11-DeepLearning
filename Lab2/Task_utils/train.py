from Metric import evaluate

def train(model, train_loader, optimizer, criterion, n_epochs, batch_size):
    model.train()
    for epoch in range(1, n_epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.float()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % batch_size == 0:
                # print epoch, batch_idx, loss.item(), accuracy(output, target)
                acc, prec, rec, f1 = evaluate(output, target)
                print(
                    f'epoch: {epoch}, batch: {batch_idx}/{len(train_loader)}, loss: {loss.item()}, acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}')
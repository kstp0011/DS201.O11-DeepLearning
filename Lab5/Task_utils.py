import torch
from sklearn.metrics import f1_score, precision_score, recall_score,  classification_report, confusion_matrix


def train(model, train_loader, dev_loader, num_epochs, optimizer, device):
    for epoch in range(num_epochs):
        model.train()
        for i, (sentences, labels) in enumerate(train_loader):
            sentences = sentences.to(device)
            labels = labels.to(device)
            outputs = model(sentences, labels)
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 30 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for sentences, labels in dev_loader:
                sentences = sentences.to(device)
                labels = labels.to(device)
                outputs = model(sentences)  # Don't provide labels here
                _, predicted = torch.max(outputs['logits'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Validation Accuracy: {100 * correct / total}%')

        

def evaluation(model, test_loader, device):
    labels_list = []
    predictions_list = []

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sentences, labels in test_loader:
            sentences = sentences.to(device)
            labels = labels.to(device)
            outputs = model(sentences)  # Don't provide labels here
            _, predicted = torch.max(outputs['logits'].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total}%')
        print("F1 score: ", f1_score(labels_list, predictions_list, average="macro"))
        print("Recall score: ", recall_score(
            labels_list, predictions_list, average="macro"))
        print("Precision score: ", precision_score(
            labels_list, predictions_list, average="macro"))
        print(classification_report(labels_list, predictions_list))
        print(confusion_matrix(labels_list, predictions_list))



gru_model_net = gru_model(input_size=len(train_dataset.vocab),
                          hidden_size=128, output_size=3, embedding_dim=100).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(gru_model_net.parameters(), lr=lr)


gru_model_net.to(device)


# Initialize variables for early stopping

best_val_loss = float('inf')

patience = 3

patience_counter = 0


for epoch in range(num_epochs):

    gru_model_net.train()

    for i, (sentences, labels) in enumerate(train_loader):
        sentences = sentences.to(device)

        labels = labels.to(device)

        outputs = gru_model_net(sentences)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if (i+1) % 30 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    gru_model_net.eval()

    val_loss = 0

    with torch.no_grad():

        correct = 0

        total = 0

        for sentences, labels in dev_loader:
            sentences = sentences.to(device)

            labels = labels.to(device)

            outputs = gru_model_net(sentences)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            # Calculate validation loss

            val_loss += criterion(outputs, labels).item()

        val_loss /= len(dev_loader)  # Average validation loss
        print(
            f'Validation Accuracy: {100 * correct / total}%, Validation Loss: {val_loss}')


        # Check for early stopping

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0  # Reset counter

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print("Early stopping")

            break



def train_early_stopping(model, train_loader, dev_loader, num_epochs, optimizer, device, patience=3):
    best_val_loss = 0
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (sentences, labels) in enumerate(train_loader):
            sentences = sentences.to(device)
            labels = labels.to(device)
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 30 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        model.eval()
        val_loss = 0
        with torch.no_grad():
            correct = 0
            total = 0
            for sentences, labels in dev_loader:
                sentences = sentences.to(device)
                labels = labels.to(device)
                outputs = model(sentences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()

            val_loss /= len(dev_loader)  # Average validation loss
            print(
                f'Validation Accuracy: {100 * correct / total}%, Validation Loss: {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset counter

            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break
from torch import nn
import torch.nn.functional as F
import torch


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, dropout, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size,
                          num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        # Embed the input sentences
        x = self.embedding(x)

        # Forward propagate the RNN
        out, _ = self.rnn(x)

        # Mean pooling
        out = torch.mean(out, dim=1)

        # Pass the output to the classifier
        out = self.fc(out)

        # softmax
        logits = F.log_softmax(out, dim=-1)

        output = {"logits": logits}
        if labels is not None:
            loss = self.criterion(logits, labels)
            output["loss"] = loss

        return output
    


class lstm_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim):
        super(lstm_model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Embed the input sentences
        x = self.embedding(x)

        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the RNN
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])

        return out
    


class gru_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim):
        super(gru_model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Embed the input sentences
        x = self.embedding(x)

        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the RNN
        out, _ = self.gru(x, h0)

        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])

        return out
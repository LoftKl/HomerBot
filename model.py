import torch
import torch.nn as nn

class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super(TinyLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        if hidden is None:
            output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

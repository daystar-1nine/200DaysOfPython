import torch
import torch.nn as nn

class StandardLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.emb(x)
        out, (h, c) = self.lstm(out)
        out = self.fc(h[-1])
        return self.sigmoid(out).squeeze(1)

if __name__ == "__main__":
    m = StandardLSTM(10, 8, 16)
    x = torch.randint(0, 10, (2, 5))
    print(m(x))

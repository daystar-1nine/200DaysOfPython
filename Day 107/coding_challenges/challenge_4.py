import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.emb(x)
        out, (h, c) = self.lstm(out)
        final_h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(final_h)).squeeze(1)

if __name__ == "__main__":
    m = BidirectionalLSTM(10, 8, 16)
    x = torch.randint(0, 10, (2, 5))
    print(m(x))

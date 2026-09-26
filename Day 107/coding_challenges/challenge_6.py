import torch
from torch.utils.data import Dataset, DataLoader

class CustomTextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

if __name__ == "__main__":
    seqs = [[1,2,3], [4,5,6], [7,8,0]]
    lbls = [0, 1, 0]
    ds = CustomTextDataset(seqs, lbls)
    dl = DataLoader(ds, batch_size=2)
    for b_x, b_y in dl:
        print(b_x)
        print(b_y)
        break

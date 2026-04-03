import pandas as pd
import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, path='data/ml-100k/u.data'):
        df = pd.read_csv(path, sep='\t', names=['user','item','rating','timestamp'])
        self.users = torch.tensor(df['user'].values - 1, dtype=torch.long)
        self.items = torch.tensor(df['item'].values - 1, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        self.n_users = int(df['user'].nunique())
        self.n_items = int(df['item'].nunique())

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
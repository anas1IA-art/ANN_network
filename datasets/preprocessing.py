import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

__all__= ['PrepaDataset',]


class PrepaDataset(Dataset):
    def __init__(self, path_data: Path):
        self.path = path_data
        self.data = self.read_data()
        self.X, self.y = self.prepare_data()

    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def prepare_data(self):
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values

        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        return X_normalized, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]
    

# if __name__ == "__main__":
    
#     prea = PrepaDataset('data/dataset.csv')

    
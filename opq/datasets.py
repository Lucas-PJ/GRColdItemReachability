import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.embeddings = pd.read_parquet(data_path)["embedding"].values
        self.embeddings = np.stack(self.embeddings, axis=0).astype(np.float32)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        return torch.from_numpy(emb)

    def __len__(self):
        return len(self.embeddings)

import numpy as np
import torch
from torch.utils.data import Dataset

class NCFRatingsDataset(Dataset):
    """
    Builds training examples by pairing each positive (u,i) with k negative items.
    Labels: 1 for positives, 0 for negatives.
    """
    def __init__(self, train_df, user_pos, num_items: int, neg_per_pos: int = 4, seed: int = 42):
        self.users = train_df["user_idx"].to_numpy(dtype=np.int64)
        self.items = train_df["item_idx"].to_numpy(dtype=np.int64)
        self.user_pos = user_pos
        self.num_items = int(num_items)
        self.neg_per_pos = int(neg_per_pos)
        self.rng = np.random.default_rng(seed)

        # We'll generate negatives on the fly in __getitem__
        # Dataset length is number of positives
        self.n_pos = len(train_df)

    def __len__(self):
        return self.n_pos

    def __getitem__(self, idx):
        u = int(self.users[idx])
        i_pos = int(self.items[idx])

        # One positive
        user_t = torch.tensor([u], dtype=torch.long)
        item_t = torch.tensor([i_pos], dtype=torch.long)
        y_t = torch.tensor([1.0], dtype=torch.float32)

        # k negatives
        negs = []
        pos_set = self.user_pos.get(u, set())
        while len(negs) < self.neg_per_pos:
            j = int(self.rng.integers(0, self.num_items))
            if j not in pos_set:
                negs.append(j)

        user_neg = torch.full((self.neg_per_pos,), u, dtype=torch.long)
        item_neg = torch.tensor(negs, dtype=torch.long)
        y_neg = torch.zeros(self.neg_per_pos, dtype=torch.float32)

        # Concatenate into one mini-batch sample
        users = torch.cat([user_t, user_neg])
        items = torch.cat([item_t, item_neg])
        labels = torch.cat([y_t, y_neg])

        return users, items, labels


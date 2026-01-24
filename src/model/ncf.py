import torch
import torch.nn as nn

class NeuralCF(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 32, hidden=(64, 32), dropout: float = 0.1):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        layers = []
        in_dim = emb_dim * 2
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)
        x = torch.cat([u, i], dim=-1)
        logits = self.mlp(x).squeeze(-1)
        return logits


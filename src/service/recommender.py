import json
import pandas as pd
import torch

from src.model.ncf import NeuralCF

class Recommender:
    def __init__(self, artifacts_dir="artifacts", device="cpu"):
        self.device = torch.device(device)

        with open(f"{artifacts_dir}/user2idx.json") as f:
            self.user2idx = {int(k): int(v) for k, v in json.load(f).items()}
        with open(f"{artifacts_dir}/item2idx.json") as f:
            # original item_id (as str) -> idx
            self.item2idx = {int(k): int(v) for k, v in json.load(f).items()}

        self.idx2item = {v: k for k, v in self.item2idx.items()}

        items = pd.read_csv(f"{artifacts_dir}/items.csv")
        self.item_id_to_title = dict(zip(items["item_id"].astype(int), items["title"].astype(str)))

        self.num_users = len(self.user2idx)
        self.num_items = len(self.item2idx)

        self.model = NeuralCF(self.num_users, self.num_items).to(self.device)
        state = torch.load(f"{artifacts_dir}/ncf.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def recommend(self, user_id: int, k: int = 10):
        if user_id not in self.user2idx:
            raise ValueError(f"unknown user_id: {user_id}. Try 1..943 for MovieLens-100k.")

        uidx = self.user2idx[user_id]
        users = torch.tensor([uidx] * self.num_items, dtype=torch.long, device=self.device)
        items = torch.arange(self.num_items, dtype=torch.long, device=self.device)

        scores = torch.sigmoid(self.model(users, items)).cpu().numpy()
        top_idx = scores.argsort()[::-1][:k]

        recs = []
        for item_idx in top_idx:
            item_id = int(self.idx2item[int(item_idx)])
            recs.append({
                "item_id": item_id,
                "title": self.item_id_to_title.get(item_id, "Unknown"),
                "score": float(scores[item_idx]),
            })
        return recs


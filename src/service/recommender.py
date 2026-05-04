import json
from pathlib import Path

import pandas as pd
import torch

from src.model.ncf import NeuralCF


class Recommender:
    def __init__(self, artifacts_dir="artifacts", device="cpu"):
        self.device = torch.device(device)

        base_dir = Path(__file__).resolve().parents[2]
        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.is_absolute():
            artifacts_path = (base_dir / artifacts_path).resolve()

        required_files = ["user2idx.json", "item2idx.json", "items.csv", "ncf.pt"]
        missing = [name for name in required_files if not (artifacts_path / name).exists()]
        if missing:
            missing_list = ", ".join(missing)
            raise FileNotFoundError(f"Missing artifacts in {artifacts_path}: {missing_list}")

        with open(artifacts_path / "user2idx.json") as f:
            self.user2idx = {int(k): int(v) for k, v in json.load(f).items()}
        with open(artifacts_path / "item2idx.json") as f:
            self.item2idx = {int(k): int(v) for k, v in json.load(f).items()}

        self.idx2item = {v: k for k, v in self.item2idx.items()}

        items = pd.read_csv(artifacts_path / "items.csv")
        self.item_id_to_title = dict(zip(items["item_id"].astype(int), items["title"].astype(str)))

        self.num_users = len(self.user2idx)
        self.num_items = len(self.item2idx)

        self.model = NeuralCF(self.num_users, self.num_items).to(self.device)
        state = torch.load(artifacts_path / "ncf.pt", map_location=self.device)
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

import os
import math
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.preprocess import load_data, build_mappings
from src.data.split import leave_one_out_split, build_user_pos_items, sample_negatives
from src.model.dataset import NCFRatingsDataset
from src.model.ncf import NeuralCF

def hit_rate_at_k(rank, k=10):
    return 1.0 if rank <= k else 0.0

def ndcg_at_k(rank, k=10):
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

@torch.no_grad()
def evaluate(model, test_df, user_pos, num_items, device, k=10, neg_eval=99, seed=42):
    model.eval()
    rng = np.random.default_rng(seed)
    hr, ndcg = [], []

    for row in test_df.itertuples(index=False):
        u = int(row.user_idx)
        pos_item = int(row.item_idx)

        negs = sample_negatives(user_pos, num_items, u, neg_eval, rng)
        candidates = [pos_item] + negs

        users_t = torch.tensor([u] * len(candidates), dtype=torch.long, device=device)
        items_t = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = torch.sigmoid(model(users_t, items_t)).cpu().numpy()

        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0]) + 1

        hr.append(hit_rate_at_k(rank, k))
        ndcg.append(ndcg_at_k(rank, k))

    return float(np.mean(hr)), float(np.mean(ndcg))

def main():
    device = torch.device("cpu")

    ratings, items = load_data()
    ratings, user2idx, item2idx = build_mappings(ratings)

    num_users = len(user2idx)
    num_items = len(item2idx)

    train_df, test_df = leave_one_out_split(ratings)
    user_pos = build_user_pos_items(train_df)

    ds = NCFRatingsDataset(train_df, user_pos, num_items=num_items, neg_per_pos=4, seed=42)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

    model = NeuralCF(
        num_users=num_users,
        num_items=num_items,
        emb_dim=32,
        hidden=(64, 32),
        dropout=0.1
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    epochs = 4
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for users, items_idx, labels in tqdm(dl, desc=f"epoch {ep}/{epochs}"):
            users = users.view(-1).to(device)
            items_idx = items_idx.view(-1).to(device)
            labels = labels.view(-1).to(device)

            logits = model(users, items_idx)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * labels.size(0)
            n += labels.size(0)

        train_loss = running / max(n, 1)
        hr, ndcg = evaluate(model, test_df, user_pos, num_items, device=device)

        print(f"epoch={ep} train_loss={train_loss:.4f} HR@10={hr:.4f} NDCG@10={ndcg:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/ncf.pt")

    with open("artifacts/user2idx.json", "w") as f:
        json.dump({str(k): int(v) for k, v in user2idx.items()}, f)

    with open("artifacts/item2idx.json", "w") as f:
        json.dump({str(k): int(v) for k, v in item2idx.items()}, f)

    items_out = items.copy()
    items_out.to_csv("artifacts/items.csv", index=False)

    print("Saved artifacts to ./artifacts")

if __name__ == "__main__":
    main()


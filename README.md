# Neural Collaborative Filtering Recommender (MovieLens 100K)

End-to-end recommender system: data → deep learning model → evaluation → API → demo UI.

## What it does
- Trains a **Neural Collaborative Filtering (NCF)** model in PyTorch on MovieLens-100k
- Evaluates using **leave-one-out** with 99 negative samples/user
- Serves recommendations via **FastAPI** (`/recommend`)
- Demo UI via **Streamlit**

## Results (CPU, 4 epochs)
- HR@10: ~0.54
- NDCG@10: ~0.31

## Run locally
### 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


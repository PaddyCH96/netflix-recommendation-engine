# Prudhvi Kadamuthuri
### Data & AI Engineer

I build ML systems that move from model training to real product experiences.
This project demonstrates production-style AI engineering: data pipeline, model training, API serving, and user-facing integration.

---

## Neural Collaborative Filtering Recommender System

### What it solves
Helps users discover relevant movies faster by ranking items based on historical user behavior.

### What I built
- Designed a MovieLens 100K data pipeline for ratings and item metadata
- Implemented a Neural Collaborative Filtering model in PyTorch
- Built the full training + evaluation workflow with ranking metrics
- Implemented a FastAPI inference backend (`/health`, `/recommend`)
- Integrated recommendations into a frontend experience (with this repo also containing a Streamlit demo app)

### Tech stack
- Python
- PyTorch
- FastAPI
- Next.js

### Output
- **Training pipeline**: data preprocessing, negative sampling, model training, evaluation
- **API layer**: production-style inference endpoints
- **UI layer**: interactive recommender demo

---

## Architecture Summary
**User → API → Model → Recommendations**

1. User submits `user_id`
2. FastAPI receives request at `/recommend`
3. Trained NCF model scores candidate items
4. Top recommendations are returned

---

## Live System (if available)
- Render Backend API: **Deploying** — [Add Render URL](https://example.com)
- Frontend UI (Vercel): **Deploying** — [Add Vercel URL](https://example.com)

---

## Key Achievements
- **HR@10 ~ 0.54**
- **NDCG@10 ~ 0.31**
- Production-style API for inference
- Artifact persistence for serving (`ncf.pt`, `user2idx.json`, `item2idx.json`, `items.csv`)

---

## Key Skills Demonstrated
- ML systems design
- PyTorch training pipeline development
- FastAPI backend engineering
- End-to-end deployment workflow

---

## Run Locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.train
uvicorn src.service.api:app --reload
streamlit run app/app.py
```

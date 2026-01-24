from fastapi import FastAPI, Query
from src.service.recommender import Recommender

app = FastAPI(title="NCF Recommender (MovieLens-100k)")
rec = Recommender()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend")
def recommend(user_id: int = Query(..., ge=1), k: int = Query(10, ge=1, le=50)):
    return {"user_id": user_id, "k": k, "recommendations": rec.recommend(user_id=user_id, k=k)}


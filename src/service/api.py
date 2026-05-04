from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.service.recommender import Recommender


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1)


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[int]


app = FastAPI(title="NCF Recommender (MovieLens-100k)")


@app.on_event("startup")
def load_model_once() -> None:
    app.state.recommender = Recommender(artifacts_dir="./artifacts")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, request: Request):
    recommender = request.app.state.recommender
    try:
        recs = recommender.recommend(user_id=payload.user_id, k=10)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    item_ids = [int(rec["item_id"]) for rec in recs]
    return {"user_id": payload.user_id, "recommendations": item_ids}


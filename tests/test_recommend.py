from src.service.recommender import Recommender


def test_recommender_loads_from_artifacts():
    rec = Recommender(artifacts_dir="./artifacts")
    assert rec.model is not None


def test_recommend_returns_item_ids():
    rec = Recommender(artifacts_dir="./artifacts")
    results = rec.recommend(user_id=1)

    assert isinstance(results, list)
    assert len(results) > 0
    item_ids = [entry["item_id"] for entry in results]
    assert all(isinstance(item_id, int) for item_id in item_ids)

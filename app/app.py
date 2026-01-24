import requests
import streamlit as st

st.set_page_config(page_title="NCF Recommender Demo", layout="centered")
st.title("🎬 Movie Recommender (Neural CF)")
st.caption("Deep-learning recommender trained on MovieLens-100k. Enter a user_id (1–943).")

API_URL = st.text_input("API URL", value="http://127.0.0.1:8000")
user_id = st.number_input("user_id", min_value=1, max_value=943, value=1, step=1)
k = st.slider("Top-K", min_value=1, max_value=20, value=10)

if st.button("Get recommendations"):
    try:
        r = requests.get(f"{API_URL}/recommend", params={"user_id": int(user_id), "k": int(k)}, timeout=10)
        r.raise_for_status()
        data = r.json()
        st.subheader(f"Top {data['k']} recommendations for user {data['user_id']}")
        for idx, rec in enumerate(data["recommendations"], start=1):
            st.write(f"{idx}. **{rec['title']}**  \nscore: `{rec['score']:.4f}`  | item_id: `{rec['item_id']}`")
    except Exception as e:
        st.error(f"API call failed: {e}")
        st.info("Make sure the FastAPI server is running on the API URL above.")

import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Get similar movie recommendations instantly!")

# -------------------------
# Load Files
# -------------------------
df = pickle.load(open("df.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
indices = pickle.load(open("indices.pkl", "rb"))

# -------------------------
# Recommendation Function
# -------------------------
def recommend(title, n=10):
    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]

    return df['title'].iloc[similar_idx]

# -------------------------
# UI Section
# -------------------------

movie_list = df['title'].values

selected_movie = st.selectbox("Select a Movie", movie_list)

num_recommendations = st.slider("Number of Recommendations", 1, 20, 10)

if st.button("Recommend"):
    recommendations = recommend(selected_movie, num_recommendations)

    st.subheader("Recommended Movies:")
    
    for movie in recommendations:
        st.write("👉", movie)

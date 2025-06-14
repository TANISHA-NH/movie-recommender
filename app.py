import streamlit as st
from recommender import get_recommendations, movies

st.title("🎬 Movie Recommendation Engine")
movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    st.subheader("Recommended Movies:")
    for m in get_recommendations(selected_movie):
        st.write("👉", m)

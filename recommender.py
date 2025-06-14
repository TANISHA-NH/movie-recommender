import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os

# ✅ Unzip the file if not already unzipped
if not os.path.exists("tmdb_5000_credits.csv"):
    with zipfile.ZipFile("tmdb_5000_credits.zip", "r") as zip_ref:
        zip_ref.extractall()
# Load dataset
movies = pd.read_csv(r"C:\Users\Tanisha\OneDrive\Desktop\Dayanand\tmdb_5000_credits.csv")

# Fill missing values just in case
movies = movies[['title', 'cast', 'crew']].dropna()

# Combine cast and crew into a single column
movies['content'] = movies['cast'] + ' ' + movies['crew']

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, num=5):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

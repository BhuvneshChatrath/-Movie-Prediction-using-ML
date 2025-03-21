from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load Dataset
movies = pd.read_csv("C:/Users/hp/movie.csv", usecols=["movieId", "title", "genres"])
movies['genres'] = movies['genres'].fillna('')
movies = movies.head(10000)  # Limit to 10,000 movies

# Convert Genres to Features
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Train Nearest Neighbors Model
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
model.fit(tfidf_matrix)

# Create Movie Index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        return []
    
    idx = indices[title]
    distances, indices_list = model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recommendations+1)
    recommended_movies = movies['title'].iloc[indices_list[0][1:]].tolist()
    
    return recommended_movies

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        movie_name = request.form["movie"]
        recommendations = recommend_movies(movie_name)

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)





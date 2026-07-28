"""Shared data loading utilities for the movie recommender.

Framework-agnostic (no Streamlit here) so it can be reused by app.py,
evaluation.py, and any offline experiment notebooks.
"""
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_data():
    """Load movies and ratings tables from the MovieLens ml-latest-small dataset."""
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

    # "(no genres listed)" means no genre info -> treat as empty list
    movies["genre_list"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    return movies, ratings


def build_genre_matrix(movies: pd.DataFrame):
    """One-hot encode each movie's genres.

    Returns:
        movie_ids: array of movieId in row order matching `matrix`
        matrix: (n_movies, n_genres) binary numpy array
        genre_names: list of genre column names matching matrix columns
    """
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(movies["genre_list"])
    movie_ids = movies["movieId"].to_numpy()
    return movie_ids, matrix.astype(float), list(mlb.classes_)


def build_user_item_matrix(ratings: pd.DataFrame, movie_ids: np.ndarray):
    """Build a sparse (movies x users) ratings matrix.

    Row order matches `movie_ids` so it lines up with the genre matrix,
    which makes it easy to combine signals in hybrid.py.
    """
    movie_id_to_row = {mid: i for i, mid in enumerate(movie_ids)}
    user_ids = np.sort(ratings["userId"].unique())
    user_id_to_col = {uid: i for i, uid in enumerate(user_ids)}

    rows = ratings["movieId"].map(movie_id_to_row)
    cols = ratings["userId"].map(user_id_to_col)
    valid = rows.notna()
    rows = rows[valid].astype(int).to_numpy()
    cols = cols[valid].astype(int).to_numpy()
    data = ratings.loc[valid, "rating"].to_numpy()

    matrix = csr_matrix((data, (rows, cols)), shape=(len(movie_ids), len(user_ids)))
    return matrix, movie_id_to_row, user_id_to_col


def get_popular_movies(movies: pd.DataFrame, ratings: pd.DataFrame, top_n: int = 10, min_ratings: int = 20):
    """Fallback list used when there isn't enough signal yet (cold start)."""
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_ratings]
    top_ids = stats.sort_values(["mean", "count"], ascending=False).head(top_n).index
    result = movies[movies["movieId"].isin(top_ids)][["movieId", "title", "genres"]].copy()
    result["score"] = result["movieId"].map(stats["mean"])
    return result.sort_values("score", ascending=False).reset_index(drop=True)

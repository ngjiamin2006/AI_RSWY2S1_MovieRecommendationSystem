"""Content-based recommender: recommends movies with similar genres.

OWNER: Member 1
This module solves cold start naturally -- a brand new user with zero
ratings can still get recommendations from their selected genre
preferences, since movies are represented purely by their own genre
features (no other users' behaviour required).

Ideas for extending this beyond the baseline:
- Bring in `tags.csv` (user-supplied free-text tags) via TF-IDF instead
  of / in addition to the one-hot genre vector.
- Weight recently liked movies more than earlier ones.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_user_profile(genre_matrix: np.ndarray, movie_id_to_row: dict, liked_movie_ids: list[int] | None,
                        genre_names: list[str] | None = None, selected_genres: list[str] | None = None):
    """Build a single vector representing what the user likes.

    Priority: average of liked movies' genre vectors if any exist,
    otherwise fall back to the genres picked during onboarding.
    """
    if liked_movie_ids:
        rows = [movie_id_to_row[mid] for mid in liked_movie_ids if mid in movie_id_to_row]
        if rows:
            return genre_matrix[rows].mean(axis=0, keepdims=True)

    if selected_genres and genre_names:
        vec = np.zeros((1, len(genre_names)))
        for g in selected_genres:
            if g in genre_names:
                vec[0, genre_names.index(g)] = 1.0
        if vec.sum() > 0:
            return vec

    return None


def recommend(movies, genre_matrix: np.ndarray, movie_ids: np.ndarray, movie_id_to_row: dict,
              genre_names: list[str], liked_movie_ids: list[int] | None = None,
              selected_genres: list[str] | None = None, top_n: int = 10):
    """Return top_n movies most similar to the user's profile.

    Returns a DataFrame with columns: movieId, title, genres, score
    """
    profile = build_user_profile(genre_matrix, movie_id_to_row, liked_movie_ids, genre_names, selected_genres)
    if profile is None:
        return None  # caller should fall back to popularity list

    scores = cosine_similarity(profile, genre_matrix)[0]
    exclude = set(liked_movie_ids or [])

    order = np.argsort(-scores)
    results = []
    for idx in order:
        mid = movie_ids[idx]
        if mid in exclude:
            continue
        results.append(mid)
        if len(results) >= top_n:
            break

    out = movies[movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: scores[movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)

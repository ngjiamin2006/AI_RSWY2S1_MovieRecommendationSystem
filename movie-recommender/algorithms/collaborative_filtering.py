"""Item-based collaborative filtering recommender.

OWNER: Member 2
Idea: two movies are "similar" if the same users rated them similarly,
regardless of genre. We never build the full (movies x movies)
similarity matrix up front (it would be ~9700x9700 -- too much memory
for a class laptop); instead we compute similarity only between the
user's liked movies and the rest, on demand, using sparse matrix ops.

This has a real cold-start limitation: a brand new user with zero
likes has no signal here, so `recommend` returns None and the caller
should fall back to a popularity list (see data_loader.get_popular_movies).

Ideas for extending this beyond the baseline:
- Try user-based CF instead of item-based and compare results.
- Try matrix factorization (e.g. sklearn's TruncatedSVD on the
  user-item matrix) instead of a similarity lookup.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def recommend(movies, user_item_matrix: csr_matrix, movie_ids: np.ndarray, movie_id_to_row: dict,
              liked_movie_ids: list[int] | None = None, top_n: int = 10):
    """Return top_n movies most similar to the user's liked movies.

    Returns a DataFrame with columns: movieId, title, genres, score
    or None if there isn't enough signal (caller should show a
    popularity-based fallback list instead).
    """
    if not liked_movie_ids:
        return None

    liked_rows = [movie_id_to_row[mid] for mid in liked_movie_ids if mid in movie_id_to_row]
    if not liked_rows:
        return None

    liked_vectors = user_item_matrix[liked_rows]  # (n_liked, n_users)
    sims = cosine_similarity(liked_vectors, user_item_matrix)  # (n_liked, n_movies)
    scores = sims.mean(axis=0)  # average similarity to each liked movie

    exclude = set(liked_movie_ids)
    order = np.argsort(-scores)
    results = []
    for idx in order:
        mid = movie_ids[idx]
        if mid in exclude or scores[idx] <= 0:
            continue
        results.append(mid)
        if len(results) >= top_n:
            break

    if not results:
        return None

    out = movies[movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: scores[movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def predict_rating(user_item_matrix: csr_matrix, movie_id_to_row: dict, user_col: int,
                    target_movie_id: int, k: int = 20) -> float | None:
    """Predict what a user would rate target_movie_id, used by evaluation.py for RMSE.

    Weighted average of the user's ratings on the k most similar movies
    to the target, weighted by item-item similarity.
    """
    if target_movie_id not in movie_id_to_row:
        return None
    target_row = movie_id_to_row[target_movie_id]

    user_rated_rows = user_item_matrix[:, user_col].nonzero()[0]
    if len(user_rated_rows) == 0:
        return None

    target_vector = user_item_matrix[target_row]
    candidate_vectors = user_item_matrix[user_rated_rows]
    sims = cosine_similarity(target_vector, candidate_vectors)[0]

    top_k_idx = np.argsort(-sims)[:k]
    top_sims = sims[top_k_idx]
    top_rows = user_rated_rows[top_k_idx]
    top_ratings = np.array([user_item_matrix[r, user_col] for r in top_rows])

    if top_sims.sum() <= 0:
        return None
    return float(np.dot(top_sims, top_ratings) / top_sims.sum())

"""Hybrid recommender: blends content-based and collaborative filtering scores.

OWNER: Member 3
Runs both underlying algorithms and combines their similarity scores
with a weight `alpha`. This directly addresses each one's weakness:
content-based alone never learns from other users' behaviour;
collaborative filtering alone can't handle a brand new user.

Ideas for extending this beyond the baseline:
- Learn `alpha` instead of hardcoding it (e.g. pick the value that
  maximizes precision@k on a validation split in evaluation.py).
- Switch the combination rule from weighted-average to rank fusion.
"""
import numpy as np

from algorithms import content_based, collaborative_filtering
from algorithms.ranking import select_top_n


def recommend(movies, genre_matrix, movie_ids, movie_id_to_row, genre_names,
              user_item_matrix, liked_movie_ids: list[int] | None = None,
              selected_genres: list[str] | None = None, top_n: int = 10, alpha: float = 0.5,
              allowed_ids: set | None = None, pool_size: int | None = None, sample_seed: int | None = None):
    """alpha=1.0 -> pure content-based, alpha=0.0 -> pure collaborative filtering.

    `allowed_ids` restricts candidates to this set (e.g. a year-range
    filter); `pool_size` + `sample_seed` turn a "Refresh" click into a
    re-roll instead of the same deterministic list -- see
    algorithms/ranking.py for details. Neither is used by evaluation.py.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cb_profile = content_based.build_user_profile(
        genre_matrix, movie_id_to_row, liked_movie_ids, genre_names, selected_genres
    )
    cb_scores = cosine_similarity(cb_profile, genre_matrix)[0] if cb_profile is not None else np.zeros(len(movie_ids))

    cf_scores = np.zeros(len(movie_ids))
    if liked_movie_ids:
        liked_rows = [movie_id_to_row[mid] for mid in liked_movie_ids if mid in movie_id_to_row]
        if liked_rows:
            liked_vectors = user_item_matrix[liked_rows]
            cf_scores = cosine_similarity(liked_vectors, user_item_matrix).mean(axis=0)

    def normalize(arr):
        span = arr.max() - arr.min()
        return (arr - arr.min()) / span if span > 0 else arr

    combined = alpha * normalize(cb_scores) + (1 - alpha) * normalize(cf_scores)

    exclude = set(liked_movie_ids or [])
    results = select_top_n(combined, movie_ids, exclude, allowed_ids, top_n, pool_size, sample_seed)

    if not results:
        return None

    out = movies[movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: combined[movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def recommend_tfidf(movies, tfidf_matrix, movie_ids, movie_id_to_row, vectorizer,
                     user_item_matrix, liked_movie_ids: list[int] | None = None,
                     selected_genres: list[str] | None = None, top_n: int = 10, alpha: float = 0.5,
                     allowed_ids: set | None = None, pool_size: int | None = None, sample_seed: int | None = None):
    """Same blend as recommend(), but scoring content-based with the richer TF-IDF matrix.

    See recommend() above for what `allowed_ids`/`pool_size`/`sample_seed` do.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cb_profile = content_based.build_tfidf_profile(
        tfidf_matrix, movie_id_to_row, liked_movie_ids, vectorizer, selected_genres
    )
    cb_scores = cosine_similarity(cb_profile, tfidf_matrix)[0] if cb_profile is not None else np.zeros(len(movie_ids))

    cf_scores = np.zeros(len(movie_ids))
    if liked_movie_ids:
        liked_rows = [movie_id_to_row[mid] for mid in liked_movie_ids if mid in movie_id_to_row]
        if liked_rows:
            liked_vectors = user_item_matrix[liked_rows]
            cf_scores = cosine_similarity(liked_vectors, user_item_matrix).mean(axis=0)

    def normalize(arr):
        span = arr.max() - arr.min()
        return (arr - arr.min()) / span if span > 0 else arr

    combined = alpha * normalize(cb_scores) + (1 - alpha) * normalize(cf_scores)

    exclude = set(liked_movie_ids or [])
    results = select_top_n(combined, movie_ids, exclude, allowed_ids, top_n, pool_size, sample_seed)

    if not results:
        return None

    out = movies[movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: combined[movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)

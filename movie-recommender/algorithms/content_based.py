"""Content-based recommender: recommends movies with similar genres.

OWNER: Member 1
This module solves cold start naturally -- a brand new user with zero
ratings can still get recommendations from their selected genre
preferences, since movies are represented purely by their own genre
features (no other users' behaviour required).

Three implementations live here on purpose:
- `build_user_profile` / `recommend` -- the original genre one-hot +
  cosine similarity baseline.
- `build_tfidf_profile` / `recommend_tfidf` -- richer TF-IDF features
  (TMDb overview/keywords/cast/director + genres, see data_loader.py's
  `build_tfidf_matrix`). Keeping both lets the report show a concrete
  before/after precision comparison between the two feature sets.
- `recommend_by_search` -- similarity from whatever a title search matched
  (genre + overview text, see data_loader.py's `build_cb_overview_matrix`),
  averaged across all matches rather than one picked movie. This is what
  the Content-Based tab in the app actually uses: no liked_movie_ids
  profile involved, per spec.

Ideas for extending this further:
- Weight recently liked movies more than earlier ones.
"""
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from algorithms.ranking import select_top_n


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


@st.cache_data(show_spinner=False)
def recommend(_movies, _genre_matrix: np.ndarray, _movie_ids: np.ndarray, _movie_id_to_row: dict,
              _genre_names: list[str], liked_movie_ids: list[int] | None = None,
              selected_genres: list[str] | None = None, top_n: int = 10,
              allowed_ids: set | None = None, pool_size: int | None = None, sample_seed: int | None = None):
    """Return top_n movies most similar to the user's profile.

    `allowed_ids` restricts candidates to this set (e.g. a year-range
    filter); `pool_size` + `sample_seed` turn a "Refresh" click into a
    re-roll instead of the same deterministic list -- see
    algorithms/ranking.py for details. Neither is used by evaluation.py.

    Returns a DataFrame with columns: movieId, title, genres, score
    """
    profile = build_user_profile(_genre_matrix, _movie_id_to_row, liked_movie_ids, _genre_names, selected_genres)
    if profile is None:
        return None  # caller should fall back to popularity list

    scores = cosine_similarity(profile, _genre_matrix)[0]
    exclude = set(liked_movie_ids or [])
    results = select_top_n(scores, _movie_ids, exclude, allowed_ids, top_n, pool_size, sample_seed)

    out = _movies[_movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: scores[_movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def recommend_by_search(_movies, _matrix, _movie_ids: np.ndarray, _movie_id_to_row: dict,
                         matched_movie_ids: list[int], top_n: int = 10, allowed_ids: set | None = None,
                         pool_size: int | None = None, sample_seed: int | None = None):
    """Content-based recommender driven by a title search, not a single
    picked movie and not the Like button. Averages the genre+overview rows
    (data_loader.py's `build_cb_overview_matrix`) of every movie the search
    matched into one profile, then ranks all other movies by cosine
    similarity to it -- so an ambiguous keyword (e.g. "marvel") is
    represented by everything it matched, instead of silently guessing one
    of them.

    Returns None if none of matched_movie_ids map to a known row (caller
    should treat like the other recommend()s' None -- fall back to
    popularity), so the UI never crashes on an empty/stale match set.
    """
    rows = [_movie_id_to_row[mid] for mid in matched_movie_ids if mid in _movie_id_to_row]
    if not rows:
        return None

    profile = np.asarray(_matrix[rows].mean(axis=0))
    scores = cosine_similarity(profile, _matrix)[0]
    exclude = set(matched_movie_ids)
    results = select_top_n(scores, _movie_ids, exclude, allowed_ids, top_n, pool_size, sample_seed)
    if not results:
        return None

    out = _movies[_movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: scores[_movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def build_tfidf_profile(tfidf_matrix, movie_id_to_row: dict, liked_movie_ids: list[int] | None,
                         vectorizer=None, selected_genres: list[str] | None = None):
    """TF-IDF equivalent of build_user_profile, using overview/keywords/cast/director/genres.

    Same priority as the genre one-hot version: average of liked movies'
    TF-IDF rows if any exist, otherwise embed the onboarding genre picks into
    the same feature space via the fitted vectorizer (genre words are part of
    the content soup, so this still works even before TMDb-only fields exist).
    """
    if liked_movie_ids:
        rows = [movie_id_to_row[mid] for mid in liked_movie_ids if mid in movie_id_to_row]
        if rows:
            return np.asarray(tfidf_matrix[rows].mean(axis=0))

    if selected_genres and vectorizer is not None:
        query = " ".join(selected_genres)
        vec = vectorizer.transform([query])
        if vec.nnz > 0:
            return vec.toarray()

    return None


@st.cache_data(show_spinner=False)
def recommend_tfidf(_movies, _tfidf_matrix, _movie_ids: np.ndarray, _movie_id_to_row: dict,
                     _vectorizer=None, liked_movie_ids: list[int] | None = None,
                     selected_genres: list[str] | None = None, top_n: int = 10,
                     allowed_ids: set | None = None, pool_size: int | None = None, sample_seed: int | None = None):
    """Same ranking logic as recommend(), scored against the richer TF-IDF content matrix.

    See recommend() above for what `allowed_ids`/`pool_size`/`sample_seed` do.
    """
    profile = build_tfidf_profile(_tfidf_matrix, _movie_id_to_row, liked_movie_ids, _vectorizer, selected_genres)
    if profile is None:
        return None

    scores = cosine_similarity(profile, _tfidf_matrix)[0]
    exclude = set(liked_movie_ids or [])
    results = select_top_n(scores, _movie_ids, exclude, allowed_ids, top_n, pool_size, sample_seed)

    out = _movies[_movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: scores[_movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)

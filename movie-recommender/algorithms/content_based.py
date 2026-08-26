"""Content-based recommender: recommends movies similar to a searched title.

OWNER: Member 1
Per tutor feedback, this module's own recommendation surface is driven
entirely by title/synopsis search -- never by the Like button or a
liked_movie_ids profile, which blurs content-based into looking like
collaborative filtering's user-profile pattern:

- `recommend_by_search` -- similarity from whatever a title search matched
  (genre + keywords + cast + director + overview text, with bigrams, see
  data_loader.py's `build_cb_overview_matrix`), averaged across all matches
  rather than one picked movie. This is what the Content-Based tab in the
  app actually uses.

`build_user_profile` / `build_tfidf_profile` below are liked_movie_ids
profile-builders kept here only because algorithms/hybrid.py (Member 3's
file, not touched by this change) imports them for its own Like-driven
recommend()/recommend_tfidf() -- evaluation.py's offline benchmark depends
on those staying available with their exact signature. Neither is used by
this file's own recommend_by_search(), and neither backs a Content-Based
tab in the app anymore.

Ideas for extending recommend_by_search further:
- Tune tag_weight/overview_weight in build_cb_overview_matrix per genre
  (e.g. weight cast/director higher for comedies, plot text higher for
  thrillers) instead of one fixed split for every movie.
"""
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from algorithms.ranking import select_top_n


def build_user_profile(genre_matrix: np.ndarray, movie_id_to_row: dict, liked_movie_ids: list[int] | None,
                        genre_names: list[str] | None = None, selected_genres: list[str] | None = None):
    """Build a single vector representing what the user likes -- used by
    hybrid.py's own Like-driven recommend(), not by anything in this file.

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


def build_tfidf_profile(tfidf_matrix, movie_id_to_row: dict, liked_movie_ids: list[int] | None,
                         vectorizer=None, selected_genres: list[str] | None = None):
    """TF-IDF equivalent of build_user_profile, using overview/keywords/cast/director/genres --
    used by hybrid.py's own Like-driven recommend_tfidf(), not by anything in this file.

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

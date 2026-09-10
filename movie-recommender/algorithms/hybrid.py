"""Hybrid recommender: blends content-based and collaborative filtering scores.

"""
import csv
import json
import os
import re
from datetime import datetime

import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from algorithms import content_based, collaborative_filtering
from algorithms.ranking import select_top_n

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
SEARCH_LOG_PATH = os.path.join(_LOG_DIR, "hybrid_search_log.csv")
BEST_ALPHA_PATH = os.path.join(_LOG_DIR, "hybrid_best_alpha.json")
FALLBACK_ALPHA = 0.15


def get_best_alpha(default: float = FALLBACK_ALPHA) -> float:
    """Return the alpha selected by offline evaluation, or a safe fallback.

    The value is intentionally loaded from a small local file so the live
    Streamlit app does not have to rerun the expensive evaluation every time.
    """
    try:
        with open(BEST_ALPHA_PATH, "r", encoding="utf-8") as f:
            value = float(json.load(f).get("best_alpha", default))
        return min(max(value, 0.0), 1.0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return float(default)


@st.cache_data(show_spinner=False)
def recommend(_movies, _genre_matrix, _movie_ids, _movie_id_to_row, _genre_names,
              _user_item_matrix, liked_movie_ids: list[int] | None = None,
              selected_genres: list[str] | None = None, top_n: int = 10, alpha: float = 0.5,
              allowed_ids: set | None = None, pool_size: int | None = None, sample_seed: int | None = None,
              _cf_scores: np.ndarray | None = None):
    """alpha=1.0 -> pure content-based, alpha=0.0 -> pure collaborative filtering.

    `allowed_ids` restricts candidates to this set (e.g. a year-range
    filter); `pool_size` + `sample_seed` turn a "Refresh" click into a
    re-roll instead of the same deterministic list -- see
    algorithms/ranking.py for details. Neither is used by evaluation.py.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cb_profile = content_based.build_user_profile(
        _genre_matrix, _movie_id_to_row, liked_movie_ids, _genre_names, selected_genres
    )
    cb_scores = cosine_similarity(cb_profile, _genre_matrix)[0] if cb_profile is not None else np.zeros(len(_movie_ids))

    if _cf_scores is not None:
        cf_scores = _cf_scores
    else:
        cf_scores = np.zeros(len(_movie_ids))
        if liked_movie_ids:
            liked_rows = [_movie_id_to_row[mid] for mid in liked_movie_ids if mid in _movie_id_to_row]
            if liked_rows:
                liked_vectors = _user_item_matrix[liked_rows]
                cf_scores = cosine_similarity(liked_vectors, _user_item_matrix).mean(axis=0)

    def normalize(arr):
        span = arr.max() - arr.min()
        return (arr - arr.min()) / span if span > 0 else arr

    combined = alpha * normalize(cb_scores) + (1 - alpha) * normalize(cf_scores)

    exclude = set(liked_movie_ids or [])
    results = select_top_n(combined, _movie_ids, exclude, allowed_ids, top_n, pool_size, sample_seed)

    if not results:
        return None

    out = _movies[_movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: combined[_movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def clean_text(text: str) -> str:
    """Lowercase, strip digits/punctuation, and drop English stop words.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # strip digits and punctuation
    tokens = [t for t in text.split() if t and t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


_TRAILING_ARTICLE_RE = re.compile(r"^(.*),\s*(the|a|an)\s*$")


def _normalize_title(text: str) -> str:
    """Lowercase, strip the trailing "(YYYY)" year and punctuation, and
    un-invert MovieLens's "Title, The (Year)" convention back to natural
    word order ("Matrix, The" -> "the matrix") so a normal-language query
    like "the matrix" can match it.

    Deliberately keeps stop words ("Back to the Future" relies on "back"
    and "to" and "the" -- clean_text's stop-word list would strip all
    three and collide it with "The Future"). Full stop-word/number
    stripping belongs to long-form content cleaning (clean_text above),
    not title lookup.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\(\d{4}\)\s*$", "", text).strip()  # drop trailing (year)
    m = _TRAILING_ARTICLE_RE.match(text)
    if m:
        text = f"{m.group(2)} {m.group(1)}"
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # strip remaining punctuation
    return re.sub(r"\s+", " ", text).strip()


def find_movie_by_search(movies, search_title: str, popularity: np.ndarray | None = None,
                          movie_id_to_row: dict | None = None):
    """Look up a single movie by title.
    """
    query = _normalize_title(search_title)
    if not query:
        return None, None, None

    normalized_titles = movies["title"].apply(_normalize_title)

    def _pick_best(candidates):
        if len(candidates) == 1 or popularity is None or movie_id_to_row is None:
            return candidates.loc[candidates["title"].str.len().idxmin()]
        pop_values = candidates["movieId"].map(
            lambda mid: popularity[movie_id_to_row[mid]] if mid in movie_id_to_row else 0
        )
        if pop_values.max() == 0:
            return candidates.loc[candidates["title"].str.len().idxmin()]
        return candidates.loc[pop_values.idxmax()]

    exact = movies[normalized_titles == query]
    if not exact.empty:
        best = _pick_best(exact)
        return int(best["movieId"]), best["title"], best["genres"]

    # Whole-word match only -- a raw substring match would let "up" match
    # inside "Supercon" or "it" match inside "Ittefaq".
    pattern = r"\b" + re.escape(query) + r"\b"
    matches = movies[normalized_titles.str.contains(pattern, na=False, regex=True)]
    if not matches.empty:
        best = _pick_best(matches)
        return int(best["movieId"]), best["title"], best["genres"]
        
    # If no whole-word match, try a fuzzy match for typos
    import difflib
    all_norm_titles = normalized_titles.dropna().tolist()
    close = difflib.get_close_matches(query, all_norm_titles, n=1, cutoff=0.6)
    if close:
        fuzzy_matches = movies[normalized_titles == close[0]]
        best = _pick_best(fuzzy_matches)
        return int(best["movieId"]), best["title"], best["genres"]

    return None, None, None


def _normalize(arr: np.ndarray) -> np.ndarray:
    span = arr.max() - arr.min()
    return (arr - arr.min()) / span if span > 0 else arr


def _item_based_cf_score(user_item_matrix, movie_id_to_row: dict, movie_ids: np.ndarray,
                          target_movie_id: int) -> np.ndarray:
    """Item-based CF: cosine similarity between the target movie's rating
    vector (its column of ratings across every user) and every other
    movie's rating vector. High score = the same users tended to rate
    both movies similarly, independent of content/genre.
    """
    if target_movie_id not in movie_id_to_row:
        return np.zeros(len(movie_ids))
    row = movie_id_to_row[target_movie_id]
    return cosine_similarity(user_item_matrix[row:row + 1], user_item_matrix)[0]


def _user_based_cf_score(user_item_matrix, movie_id_to_row: dict, movie_ids: np.ndarray,
                          target_movie_id: int, min_rating: float = 4.0, max_fans: int = 500) -> np.ndarray:
    """User-based CF: find users who rated the target movie >= min_rating
    ("fans"), then score every other movie by the average rating those
    same fans gave it. This is the "what did people who liked this movie
    also rate highly" half of collaborative filtering, complementing the
    item-based signal above.
    """
    if target_movie_id not in movie_id_to_row:
        return np.zeros(len(movie_ids))
    target_ratings = user_item_matrix[movie_id_to_row[target_movie_id]].toarray().ravel()
    fan_cols = np.where(target_ratings >= min_rating)[0]
    if len(fan_cols) == 0:
        return np.zeros(len(movie_ids))
    if len(fan_cols) > max_fans:
        # Cap crowd size for speed on ml-25m's scale -- a random sample of
        # a blockbuster's fans is still representative of what they liked.
        rng = np.random.default_rng(0)
        fan_cols = rng.choice(fan_cols, size=max_fans, replace=False)

    fan_ratings = user_item_matrix[:, fan_cols]
    sums = np.asarray(fan_ratings.sum(axis=1)).ravel().astype(float)
    counts = np.asarray((fan_ratings > 0).sum(axis=1)).ravel()
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def _personalized_user_based_cf_score(user_item_matrix, movie_id_to_row: dict, movie_ids: np.ndarray,
                                       liked_movie_ids: list, k: int = 200, min_support: int = 3,
                                       like_rating: float = 5.0) -> np.ndarray:
    if not liked_movie_ids:
        return np.zeros(len(movie_ids))

    pseudo_vector = np.zeros(len(movie_ids))
    for mid in liked_movie_ids:
        if mid in movie_id_to_row:
            pseudo_vector[movie_id_to_row[mid]] = like_rating
    if not pseudo_vector.any():
        return np.zeros(len(movie_ids))  # none of the liked ids are in this dataset

    pseudo_sparse = csr_matrix(pseudo_vector)
    # user_item_matrix is (movies, users); transpose (cheap -- CSR->CSC
    # format flip, no data copy) so each row is one user to compare against.
    sims = cosine_similarity(pseudo_sparse, user_item_matrix.T)[0]

    neighbor_cols = np.argsort(sims)[-k:]
    neighbor_cols = neighbor_cols[sims[neighbor_cols] > 0]
    if len(neighbor_cols) == 0:
        return np.zeros(len(movie_ids))

    weights = sims[neighbor_cols]
    neighborhood = user_item_matrix[:, neighbor_cols]
    weighted_sum = np.asarray(neighborhood.multiply(weights).sum(axis=1)).ravel()
    support = np.asarray((neighborhood > 0).sum(axis=1)).ravel()
    weight_total = np.asarray((neighborhood > 0).multiply(weights).sum(axis=1)).ravel()

    scores = np.divide(weighted_sum, weight_total, out=np.zeros_like(weighted_sum), where=weight_total > 0)
    scores[support < min_support] = 0  # drop low-confidence single-fan matches
    return scores


def combined_collaborative_score(user_item_matrix, movie_id_to_row: dict, movie_ids: np.ndarray,
                                  target_movie_id: int, liked_movie_ids: list | None = None):
    
    item_scores = _item_based_cf_score(user_item_matrix, movie_id_to_row, movie_ids, target_movie_id)
    fan_scores = _user_based_cf_score(user_item_matrix, movie_id_to_row, movie_ids, target_movie_id)
    signals = [_normalize(item_scores), _normalize(fan_scores)]

    personalized_used = False
    if liked_movie_ids:
        personalized_scores = _personalized_user_based_cf_score(user_item_matrix, movie_id_to_row, movie_ids,
                                                                   liked_movie_ids)
        if personalized_scores.any():
            signals.append(_normalize(personalized_scores))
            personalized_used = True

    return np.mean(signals, axis=0), personalized_used


def log_search(query: str, matched_title: str | None, path: str = SEARCH_LOG_PATH) -> None:

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        is_new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "query", "matched_title"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), query, matched_title or ""])
    except OSError:
        pass  # logging is a nice-to-have, never worth breaking the request over


def recommend_by_search(
    movies, content_matrix, user_item_matrix, movie_ids: np.ndarray,
    movie_id_to_row: dict, search_title: str, top_n: int = 10, alpha: float = 0.5,
    liked_movie_ids: list | None = None, allowed_ids: set | None = None,
    pool_size: int | None = None, sample_seed: int | None = None
):
    """Search-driven hybrid: searched movie for content, liked movies for CF."""
    alpha = min(max(alpha, 0.0), 1.0)
    liked_movie_ids = liked_movie_ids or []

    popularity = np.asarray((user_item_matrix > 0).sum(axis=1)).ravel()
    target_movie_id, matched_title, matched_genres = find_movie_by_search(
        movies, search_title, popularity=popularity, movie_id_to_row=movie_id_to_row
    )
    log_search(search_title, matched_title)

    if target_movie_id is None:
        return None, {
            "error": f"No movie found matching '{search_title}'. Try a different spelling or a shorter title."
        }
    if target_movie_id not in movie_id_to_row:
        return None, {
            "error": f"'{matched_title}' has no content features available for comparison."
        }

    target_row = movie_id_to_row[target_movie_id]
    content_scores = cosine_similarity(
        content_matrix[target_row:target_row + 1], content_matrix
    )[0]
    content_scores = _normalize(content_scores)

    valid_liked_movie_ids = [
        mid for mid in liked_movie_ids if mid in movie_id_to_row
    ]

    if valid_liked_movie_ids:
        collaborative_scores = _personalized_user_based_cf_score(
            user_item_matrix, movie_id_to_row, movie_ids, valid_liked_movie_ids
        )
        collaborative_scores = _normalize(collaborative_scores)
        personalized_used = True
    else:
        collaborative_scores = np.zeros(len(movie_ids))
        personalized_used = False

    cf_available = bool(np.any(collaborative_scores > 0))

    if cf_available:
        effective_content_weight = alpha
        effective_cf_weight = 1.0 - alpha
    else:
        effective_content_weight = 1.0
        effective_cf_weight = 0.0

    content_contribution = effective_content_weight * content_scores
    collaborative_contribution = effective_cf_weight * collaborative_scores
    combined = content_contribution + collaborative_contribution

    exclude = {target_movie_id, *valid_liked_movie_ids}
    results = select_top_n(
        combined, movie_ids, exclude, allowed_ids, top_n, pool_size, sample_seed
    )

    if not results:
        return None, {
            "error": f"No recommendations found for '{matched_title}'.",
            "matched_title": matched_title,
            "matched_movie_id": target_movie_id,
        }

    analysis = movies[movies["movieId"].isin(results)][
        ["movieId", "title", "genres"]
    ].copy()

    analysis["content_score"] = analysis["movieId"].map(
        lambda mid: content_scores[movie_id_to_row[mid]]
    )
    analysis["collaborative_score"] = analysis["movieId"].map(
        lambda mid: collaborative_scores[movie_id_to_row[mid]]
    )
    analysis["content_contribution"] = analysis["movieId"].map(
        lambda mid: content_contribution[movie_id_to_row[mid]]
    )
    analysis["collaborative_contribution"] = analysis["movieId"].map(
        lambda mid: collaborative_contribution[movie_id_to_row[mid]]
    )
    analysis["score"] = analysis["movieId"].map(
        lambda mid: combined[movie_id_to_row[mid]]
    )

    analysis = analysis.sort_values("score", ascending=False).reset_index(drop=True)
    display = analysis[["movieId", "title", "genres"]].copy()

    meta = {
        "matched_title": matched_title,
        "matched_movie_id": target_movie_id,
        "matched_genres": matched_genres,
        "personalized": personalized_used,
        "liked_movie_count": len(valid_liked_movie_ids),
        "alpha": alpha,
        "effective_content_weight": effective_content_weight,
        "effective_collaborative_weight": effective_cf_weight,
        "cf_available": cf_available,
        "analysis": analysis,
    }
    return display, meta


@st.cache_data(show_spinner=False)
def recommend_tfidf(_movies, _tfidf_matrix, _movie_ids, _movie_id_to_row, _vectorizer,
                     _user_item_matrix, liked_movie_ids: list[int] | None = None,
                     selected_genres: list[str] | None = None, top_n: int = 10, alpha: float = 0.5,
                     allowed_ids: set | None = None, pool_size: int | None = None, sample_seed: int | None = None,
                     _cf_scores: np.ndarray | None = None):
    """Same blend as recommend(), but scoring content-based with the richer TF-IDF matrix.

    See recommend() above for what `allowed_ids`/`pool_size`/`sample_seed` do.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cb_profile = content_based.build_tfidf_profile(
        _tfidf_matrix, _movie_id_to_row, liked_movie_ids, _vectorizer, selected_genres
    )
    cb_scores = cosine_similarity(cb_profile, _tfidf_matrix)[0] if cb_profile is not None else np.zeros(len(_movie_ids))

    if _cf_scores is not None:
        cf_scores = _cf_scores
    else:
        cf_scores = np.zeros(len(_movie_ids))
        if liked_movie_ids:
            liked_rows = [_movie_id_to_row[mid] for mid in liked_movie_ids if mid in _movie_id_to_row]
            if liked_rows:
                liked_vectors = _user_item_matrix[liked_rows]
                cf_scores = cosine_similarity(liked_vectors, _user_item_matrix).mean(axis=0)

    def normalize(arr):
        span = arr.max() - arr.min()
        return (arr - arr.min()) / span if span > 0 else arr

    combined = alpha * normalize(cb_scores) + (1 - alpha) * normalize(cf_scores)

    exclude = set(liked_movie_ids or [])
    results = select_top_n(combined, _movie_ids, exclude, allowed_ids, top_n, pool_size, sample_seed)

    if not results:
        return None

    out = _movies[_movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: combined[_movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)
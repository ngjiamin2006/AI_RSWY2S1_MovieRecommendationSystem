"""Collaborative Filtering Recommender System.

This module implements both Item-Based Collaborative Filtering (for general recommendations)
and User-Based Collaborative Filtering (for local user profiles), utilizing both
Cosine Similarity and Pearson Correlation Similarity for scoring.
Maximal Marginal Relevance (MMR) is also used for diverse results.
"""
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from algorithms.ranking import select_top_n


# ---------------------------------------------------------------------------
# Similarity Metrics
# ---------------------------------------------------------------------------

def correlation_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient (Correlation Similarity) between two rating vectors.

    Also known as Pearson Correlation Similarity. Unlike cosine similarity,
    this first mean-centers each vector on their co-rated items, removing
    individual rating-scale bias. A harsh rater and a generous rater can still
    be matched correctly if they agree on which movies are better or worse.

    Only dimensions where BOTH vectors are non-zero (i.e., co-rated items) are
    used in the calculation, which is standard practice for CF systems.

    Returns:
        float in [-1.0, 1.0]. Returns 0.0 when fewer than 2 co-rated items
        exist, or when one vector has zero variance.
    """
    # Fast early exit: find co-rated dimensions (non-zero in both vectors)
    mask = (vec_a != 0) & (vec_b != 0)
    n_corated = mask.sum()
    if n_corated < 2:
        # Need at least 2 co-rated items to compute a meaningful correlation
        return 0.0

    a = vec_a[mask]
    b = vec_b[mask]

    # Mean-center each vector (subtract their personal average on co-rated items)
    a = a - a.mean()
    b = b - b.mean()

    # Denominator: product of L2 norms (standard deviations)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    denom = norm_a * norm_b

    if denom == 0.0:
        # Zero variance: the user gave identical ratings to all co-rated movies
        return 0.0

    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Core Score Computation
# ---------------------------------------------------------------------------

def compute_cf_scores(user_item_matrix: csr_matrix, movie_id_to_row: dict,
                       liked_movie_ids: list[int] | None) -> np.ndarray:
    """Compute Item-Based CF scores using Cosine Similarity (memory-efficient).

    Uses scikit-learn's vectorized cosine_similarity on the sparse matrix
    directly — no dense materialisation of the full matrix.

    Returns:
        np.ndarray of shape (n_movies,): average cosine similarity between
        each movie and the set of liked movies.

    Note:
        Correlation Similarity for displayed movies is computed separately via
        compute_correlation_for_rows(), which operates one row at a time to
        keep memory usage O(n_users) instead of O(n_movies × n_users).
    """
    n_movies = user_item_matrix.shape[0]
    liked_rows = [movie_id_to_row[mid] for mid in (liked_movie_ids or [])
                  if mid in movie_id_to_row]

    if not liked_rows:
        return np.zeros(n_movies)

    liked_vectors = user_item_matrix[liked_rows]  # shape: (n_liked, n_users)

    # --- Memory-safe cosine similarity (never calls .toarray() on full matrix) ---
    # sklearn's cosine_similarity() internally converts the full matrix to dense
    # on some scipy/numpy versions, which would require 37+ GB RAM. Instead, we
    # compute the dot product manually using sparse matrix operations:
    #
    #   cosine(A, B) = (A @ Bᵀ) / (||A|| × ||B||)
    #
    # Step 1: Compute L2 norms row-wise on the sparse matrices (O(nnz), no dense)
    liked_norms = np.sqrt(np.asarray(liked_vectors.power(2).sum(axis=1)).flatten())
    liked_norms = np.maximum(liked_norms, 1e-12)           # avoid division by zero

    movie_norms = np.sqrt(np.asarray(user_item_matrix.power(2).sum(axis=1)).flatten())
    movie_norms = np.maximum(movie_norms, 1e-12)

    # Step 2: Raw dot product — result is (n_liked, n_movies), safe to densify
    # liked_vectors: (n_liked, n_users), user_item_matrix.T: (n_users, n_movies)
    raw_dots = liked_vectors @ user_item_matrix.T          # sparse result
    if hasattr(raw_dots, "toarray"):
        raw_dots = raw_dots.toarray()                      # (n_liked, n_movies) — small!

    # Step 3: Normalise by row norms
    cosine_matrix = raw_dots / liked_norms[:, None] / movie_norms[None, :]

    # Step 4: Average over liked movies
    return cosine_matrix.mean(axis=0)


def compute_correlation_for_rows(user_item_matrix: csr_matrix,
                                  movie_id_to_row: dict,
                                  liked_movie_ids: list[int],
                                  target_movie_ids: list[int]) -> dict:
    """Compute Correlation Similarity (Pearson) for a small subset of movies.

    Unlike compute_cf_scores(), this function is memory-safe: it extracts only
    one sparse row at a time (~162 K floats) instead of materialising the full
    (62 K × 162 K) dense matrix (~38 GB).

    Key optimisation: liked movie vectors are extracted ONCE and reused for
    every target movie, reducing sparse-row extractions from
    O(n_target × n_liked) to O(n_target + n_liked).

    Args:
        user_item_matrix: Sparse (n_movies, n_users) rating matrix.
        movie_id_to_row:  {movieId -> row index} lookup.
        liked_movie_ids:  Movies the user has liked (used as the profile).
        target_movie_ids: The small set of displayed/candidate movies to score.

    Returns:
        dict {movieId -> correlation_score}
    """
    liked_rows = [movie_id_to_row[mid] for mid in liked_movie_ids
                  if mid in movie_id_to_row]
    if not liked_rows:
        return {mid: 0.0 for mid in target_movie_ids}

    # --- Optimisation: pre-extract all liked vectors ONCE ---
    # Each row is a 1-D dense array of length n_users (~162 K floats ≈ 0.6 MB).
    liked_vecs: list[np.ndarray] = [
        user_item_matrix[row].toarray().flatten() for row in liked_rows
    ]

    result: dict[int, float] = {}
    for target_mid in target_movie_ids:
        if target_mid not in movie_id_to_row:
            result[target_mid] = 0.0
            continue

        # Extract target row once and reuse across all liked vectors
        target_vec = user_item_matrix[movie_id_to_row[target_mid]].toarray().flatten()

        total_corr = sum(correlation_similarity(lv, target_vec) for lv in liked_vecs)
        result[target_mid] = total_corr / len(liked_vecs)

    return result


# ---------------------------------------------------------------------------
# Recommendation Functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def recommend(_movies, _user_item_matrix: csr_matrix, _movie_ids: np.ndarray,
              _movie_id_to_row: dict, liked_movie_ids: list[int] | None = None,
              top_n: int = 10, allowed_ids: set | None = None,
              pool_size: int | None = None, sample_seed: int | None = None,
              _cf_scores: np.ndarray | None = None):
    """Generate recommendations using Item-Based Collaborative Filtering.

    Ranking is based on Cosine Similarity (fast, memory-efficient on the full
    catalogue). Correlation Similarity is NOT computed here — it is expensive
    to compute for all 62 K movies. Instead it is added to the final display
    cards via compute_correlation_for_rows() in app.py.

    Args:
        _cf_scores: Optional pre-computed cosine score array. If provided,
                    the function skips the compute_cf_scores() call.
    """
    if not liked_movie_ids:
        return None

    liked_rows = [_movie_id_to_row[mid] for mid in liked_movie_ids
                  if mid in _movie_id_to_row]
    if not liked_rows:
        return None

    # Use pre-computed scores if available, otherwise compute cosine similarity
    scores = (_cf_scores if _cf_scores is not None
              else compute_cf_scores(_user_item_matrix, _movie_id_to_row, liked_movie_ids))

    # Build candidate pool: positive score, not already liked, within allowed filter
    exclude = set(liked_movie_ids)
    positive_mask = scores > 0
    candidate_set = (set(_movie_ids[positive_mask]) if allowed_ids is None
                     else allowed_ids & set(_movie_ids[positive_mask]))

    results = select_top_n(scores, _movie_ids, exclude, candidate_set,
                           top_n, pool_size, sample_seed)
    if not results:
        return None

    max_score = scores.max() if scores.max() > 0 else 1.0
    out = _movies[_movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["rating"] = out["movieId"].map(
        lambda mid: 3.5 + 1.5 * (scores[_movie_id_to_row[mid]] / max_score)
    )
    return out.sort_values("rating", ascending=False).reset_index(drop=True)


def predict_rating(user_item_matrix: csr_matrix, movie_id_to_row: dict,
                   user_col: int, target_movie_id: int, k: int = 20) -> float | None:
    """Predict the star rating a user would give to a target movie.

    Used strictly for offline evaluation metrics (RMSE/MAE).

    Algorithm (Item-Based KNN):
        1. Find all movies this user has rated.
        2. Compute cosine similarity between the target movie and each rated movie.
        3. Take the Top-K most similar rated movies.
        4. Return a similarity-weighted average of their actual ratings.

    Args:
        user_col:         Column index of the user in user_item_matrix.
        target_movie_id:  The movie to predict a rating for.
        k:                Number of nearest neighbours to use (default: 20).

    Returns:
        Predicted rating float, or None if prediction is not possible.
    """
    if target_movie_id not in movie_id_to_row:
        return None
    target_row = movie_id_to_row[target_movie_id]

    # Retrieve all movies this user has actually rated (non-zero entries)
    user_rated_rows = user_item_matrix[:, user_col].nonzero()[0]
    if len(user_rated_rows) == 0:
        return None

    # Compute cosine similarity between the target and all rated movies
    target_vector = user_item_matrix[target_row]            # shape: (1, n_users)
    candidate_vectors = user_item_matrix[user_rated_rows]   # shape: (n_rated, n_users)
    sims = cosine_similarity(target_vector, candidate_vectors)[0]

    # Select Top-K most similar neighbours
    top_k_idx = np.argsort(-sims)[:k]
    top_sims = sims[top_k_idx]

    if top_sims.sum() <= 0:
        return None

    top_rows = user_rated_rows[top_k_idx]
    # Retrieve actual ratings for the selected neighbours
    top_ratings = np.array([user_item_matrix[r, user_col] for r in top_rows], dtype=float)

    # Weighted average: sum(sim_i × rating_i) / sum(sim_i)
    return float(np.dot(top_sims, top_ratings) / top_sims.sum())


# ---------------------------------------------------------------------------
# MMR Diversification
# ---------------------------------------------------------------------------

def _apply_mmr(cf_scores: np.ndarray, _movie_ids: np.ndarray,
               _user_item_matrix: csr_matrix, candidate_indices: list,
               top_n: int = 10, lambda_param: float = 0.3,
               pool_size: int = 50) -> list:
    """Apply Maximal Marginal Relevance (MMR) to diversify recommendations.

    MMR re-ranks candidates by balancing:
        - Relevance: high cosine score w.r.t. the user's liked movies.
        - Diversity: low similarity to movies already selected in this round.

    MMR formula:
        score(i) = (1 - λ) × relevance(i) - λ × max_sim(i, selected)

    Args:
        lambda_param: Trade-off weight (0 = pure relevance, 1 = pure diversity).
        pool_size:    Maximum number of candidates to consider (for speed).
    """
    if not candidate_indices:
        return []

    # Trim the candidate pool to a manageable size for efficiency
    candidate_indices = candidate_indices[:pool_size]
    if len(candidate_indices) <= 1:
        return [_movie_ids[idx] for idx in candidate_indices][:top_n]

    # Pre-compute pairwise similarity among the candidate pool
    candidate_vectors = _user_item_matrix[candidate_indices]    # (pool, n_users)
    item_sim_matrix = cosine_similarity(candidate_vectors)      # (pool, pool)

    selected_positions: list[int] = []   # positions within candidate_indices
    selected_mids: list = []

    while len(selected_mids) < top_n and len(selected_mids) < len(candidate_indices):
        best_mmr = -np.inf
        best_pos = -1
        best_real_idx = -1

        for pos, real_idx in enumerate(candidate_indices):
            if pos in selected_positions:
                continue

            relevance = cf_scores[real_idx]
            # Maximum similarity to any already-selected item (diversity penalty)
            penalty = (np.max(item_sim_matrix[pos, selected_positions])
                       if selected_positions else 0.0)

            # MMR score
            mmr_score = (1.0 - lambda_param) * relevance - lambda_param * penalty

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_pos = pos
                best_real_idx = real_idx

        if best_pos == -1:
            break

        selected_positions.append(best_pos)
        selected_mids.append(_movie_ids[best_real_idx])

    return selected_mids


# ---------------------------------------------------------------------------
# User-Based Collaborative Filtering (Interactive Demo)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def recommend_user_based(_movies, _user_item_matrix: csr_matrix,
                          _movie_ids: np.ndarray, _movie_id_to_row: dict,
                          current_user: str, local_profiles: dict,
                          top_n: int = 10):
    """User-Based Collaborative Filtering for the Interactive Demo Tab.

    Phase 1 — User-User Similarity via Jaccard:
        Finds which other local simulated user (e.g., 'User 2') has the most
        overlapping likes with the current user using the Jaccard coefficient:
            J(A, B) = |A ∩ B| / |A ∪ B|
        Recommends movies that the best-matched user liked but the current
        user hasn't seen yet.

    Phase 2 — Item-Based CF Fallback via Cosine Similarity:
        If no matching user is found (or to pad remaining slots), falls back to
        Item-Based CF with MMR diversification.

    Returns:
        (DataFrame | None, explanation: str, jaccard_score: float)
    """
    my_likes = set(local_profiles.get(current_user, []))
    if not my_likes:
        return (None,
                "You haven't liked any movies yet. Like some movies to see collaborative recommendations!",
                0.0)

    # -----------------------------------------------------------------------
    # Phase 1: Jaccard similarity — find the best-matching local user
    # -----------------------------------------------------------------------
    best_match_user = None
    best_match_score = 0.0
    best_match_likes: list = []

    for other_user, their_likes in local_profiles.items():
        if other_user == current_user or not their_likes:
            continue

        their_set = set(their_likes)

        # Skip if this user has nothing new to offer
        if not (their_set - my_likes):
            continue

        # Jaccard similarity: Intersection over Union
        intersection = len(my_likes & their_set)
        union = len(my_likes | their_set)
        jaccard = intersection / union if union > 0 else 0.0

        if jaccard > best_match_score:
            best_match_score = jaccard
            best_match_user = other_user
            best_match_likes = their_likes

    # -----------------------------------------------------------------------
    # Phase 2: Cosine Similarity for item-based fallback / padding
    # -----------------------------------------------------------------------
    liked_rows = [_movie_id_to_row[mid] for mid in my_likes if mid in _movie_id_to_row]
    cf_scores = np.zeros(len(_movie_ids))

    if liked_rows:
        liked_vectors = _user_item_matrix[liked_rows]
        # Memory-safe cosine similarity: manual sparse dot product, never .toarray() full matrix
        liked_norms = np.sqrt(np.asarray(liked_vectors.power(2).sum(axis=1)).flatten())
        liked_norms = np.maximum(liked_norms, 1e-12)
        movie_norms = np.sqrt(np.asarray(_user_item_matrix.power(2).sum(axis=1)).flatten())
        movie_norms = np.maximum(movie_norms, 1e-12)
        raw_dots = liked_vectors @ _user_item_matrix.T
        if hasattr(raw_dots, "toarray"):
            raw_dots = raw_dots.toarray()              # safe: shape is (n_liked, n_movies)
        cf_scores = (raw_dots / liked_norms[:, None] / movie_norms[None, :]).mean(axis=0)
        # Note: Correlation Similarity is computed on-demand in app.py via
        # compute_correlation_for_rows() for just the ~30 displayed movies.

    # Candidate pool for item-based fallback: unseen movies with positive score
    order = np.argsort(-cf_scores)
    allowed_candidates = [idx for idx in order
                          if _movie_ids[idx] not in my_likes and cf_scores[idx] > 0]
    max_cf = cf_scores.max() if cf_scores.max() > 0 else 1.0

    # -----------------------------------------------------------------------
    # Case A: No similar user found → pure item-based CF with MMR
    # -----------------------------------------------------------------------
    if not best_match_user or best_match_score == 0:
        item_based_recs = _apply_mmr(cf_scores, _movie_ids, _user_item_matrix,
                                     allowed_candidates, top_n=top_n)
        if not item_based_recs:
            return None, "No recommendations available right now.", 0.0

        out = _movies[_movies["movieId"].isin(item_based_recs)][["movieId", "title", "genres"]].copy()
        out["rating"] = out["movieId"].map(
            lambda mid: 3.5 + 1.5 * (cf_scores[_movie_id_to_row[mid]] / max_cf)
        )
        out = out.sort_values("rating", ascending=False).head(top_n).reset_index(drop=True)
        return out, "💡 **Other movies you might like**", 0.0

    # -----------------------------------------------------------------------
    # Case B: User match found → user-based recs + MMR padding
    # -----------------------------------------------------------------------
    user_based_recs = [mid for mid in best_match_likes if mid not in my_likes]

    remaining_n = top_n - len(user_based_recs)
    padding_candidates = [idx for idx in allowed_candidates
                          if _movie_ids[idx] not in user_based_recs]
    item_based_recs = (_apply_mmr(cf_scores, _movie_ids, _user_item_matrix,
                                   padding_candidates, top_n=remaining_n)
                       if remaining_n > 0 else [])

    final_recs = (user_based_recs + item_based_recs)[:top_n]

    if not final_recs:
        return (None,
                f"Your taste perfectly matches **{best_match_user}**! "
                "But they haven't liked anything you haven't already seen.",
                best_match_score)

    # Build the output DataFrame
    out = _movies[_movies["movieId"].isin(final_recs)][["movieId", "title", "genres"]].copy()

    # Scale expected ratings:
    #   User-matched movies: 4.5 – 5.0 (based on Jaccard strength)
    #   Item-based padded:   3.5 – 4.5 (based on normalised cosine score)
    user_based_set = set(user_based_recs)

    def get_rating(mid: int) -> float:
        if mid in user_based_set:
            return 4.5 + 0.5 * best_match_score   # e.g., 80% Jaccard → 4.9★
        return 3.5 + 1.0 * (cf_scores[_movie_id_to_row[mid]] / max_cf)

    out["rating"] = out["movieId"].map(get_rating)
    out = out.sort_values("rating", ascending=False).head(top_n).reset_index(drop=True)
    return out, "💡 **Other movies you might like**", best_match_score
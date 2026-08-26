import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from algorithms.ranking import select_top_n


def compute_cf_scores(user_item_matrix: csr_matrix, movie_id_to_row: dict,
                       liked_movie_ids: list[int] | None) -> np.ndarray:

    n_movies = user_item_matrix.shape[0]
    liked_rows = [movie_id_to_row[mid] for mid in (liked_movie_ids or []) if mid in movie_id_to_row]
    if not liked_rows:
        return np.zeros(n_movies)

    liked_vectors = user_item_matrix[liked_rows]  # (n_liked, n_users)
    sims = cosine_similarity(liked_vectors, user_item_matrix)  # (n_liked, n_movies)
    return sims.mean(axis=0)  # average similarity to each liked movie


@st.cache_data(show_spinner=False)
def recommend(_movies, _user_item_matrix: csr_matrix, _movie_ids: np.ndarray, _movie_id_to_row: dict,
              liked_movie_ids: list[int] | None = None, top_n: int = 10,
              allowed_ids: set | None = None, pool_size: int | None = None, sample_seed: int | None = None,
              _cf_scores: np.ndarray | None = None):

    if not liked_movie_ids:
        return None

    liked_rows = [_movie_id_to_row[mid] for mid in liked_movie_ids if mid in _movie_id_to_row]
    if not liked_rows:
        return None

    scores = _cf_scores if _cf_scores is not None else compute_cf_scores(
        _user_item_matrix, _movie_id_to_row, liked_movie_ids
    )

    exclude = set(liked_movie_ids)
    positive_mask = scores > 0
    candidate_allowed = set(_movie_ids[positive_mask]) if allowed_ids is None else allowed_ids & set(_movie_ids[positive_mask])
    results = select_top_n(scores, _movie_ids, exclude, candidate_allowed, top_n, pool_size, sample_seed)

    if not results:
        return None

    out = _movies[_movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["rating"] = out["movieId"].map(lambda mid: scores[_movie_id_to_row[mid]])
    return out.sort_values("rating", ascending=False).reset_index(drop=True)


def predict_rating(user_item_matrix: csr_matrix, movie_id_to_row: dict, user_col: int,
                    target_movie_id: int, k: int = 20) -> float | None:

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


def _apply_mmr(cf_scores, _movie_ids, _user_item_matrix, candidate_indices, top_n=10, lambda_param=0.3, pool_size=50):
    if not candidate_indices:
        return []
    candidate_indices = candidate_indices[:pool_size]
    if len(candidate_indices) <= 1:
        return [_movie_ids[idx] for idx in candidate_indices][:top_n]
    candidate_vectors = _user_item_matrix[candidate_indices]
    item_sim_matrix = cosine_similarity(candidate_vectors)
    selected_indices = []
    selected_mids = []
    while len(selected_mids) < top_n and len(selected_mids) < len(candidate_indices):
        best_mmr = -float('inf')
        best_cand_idx = -1
        best_real_idx = -1
        for i, real_idx in enumerate(candidate_indices):
            if i in selected_indices:
                continue
            relevance = cf_scores[real_idx]
            penalty = np.max(item_sim_matrix[i, selected_indices]) if selected_indices else 0.0
            mmr_score = (1 - lambda_param) * relevance - lambda_param * penalty
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_cand_idx = i
                best_real_idx = real_idx
        if best_cand_idx == -1:
            break
        selected_indices.append(best_cand_idx)
        selected_mids.append(_movie_ids[best_real_idx])
    return selected_mids


@st.cache_data(show_spinner=False)
def recommend_user_based(_movies, _user_item_matrix: csr_matrix, _movie_ids: np.ndarray, _movie_id_to_row: dict,
                            current_user: str, local_profiles: dict, top_n: int = 10):
    my_likes = set(local_profiles.get(current_user, []))
    if not my_likes:
        return None, "You haven't liked any movies yet. Like some movies to see collaborative recommendations!"

    my_set = set(my_likes)
    
    # Calculate similarity (Jaccard) with other local users
    best_match_user = None
    best_match_score = 0.0
    best_match_likes = []
    
    for other_user, their_likes in local_profiles.items():
        if other_user == current_user or not their_likes:
            continue
            
        their_set = set(their_likes)
        
        # Skip this user if they don't have any movies I haven't already seen
        if not (their_set - my_set):
            continue
            
        intersection = len(my_set & their_set)
        union = len(my_set | their_set)
        jaccard = intersection / union if union > 0 else 0
        
        if jaccard > best_match_score:
            best_match_score = jaccard
            best_match_user = other_user
            best_match_likes = their_likes

    liked_rows = [_movie_id_to_row[mid] for mid in my_likes if mid in _movie_id_to_row]
    cf_scores = np.zeros(len(_movie_ids))
    if liked_rows:
        liked_vectors = _user_item_matrix[liked_rows]
        cf_scores = cosine_similarity(liked_vectors, _user_item_matrix).mean(axis=0)

    # Determine candidates for item-based fallback (must be unseen and have CF score > 0)
    order = np.argsort(-cf_scores)
    allowed_candidates = [idx for idx in order if _movie_ids[idx] not in my_set and cf_scores[idx] > 0]
    
    max_cf = np.max(cf_scores) if np.max(cf_scores) > 0 else 1.0

    if not best_match_user or best_match_score == 0:
        # No local user match: use MMR on item-based candidates for diverse results
        item_based_recs = _apply_mmr(cf_scores, _movie_ids, _user_item_matrix, allowed_candidates, top_n=top_n)
        
        if not item_based_recs:
            return None, "No recommendations available right now."
            
        out = _movies[_movies["movieId"].isin(item_based_recs)][["movieId", "title", "genres"]].copy()
        out["rating"] = out["movieId"].map(lambda mid: 3.5 + 1.5 * (cf_scores[_movie_id_to_row[mid]] / max_cf))
        out = out.sort_values("rating", ascending=False).head(top_n).reset_index(drop=True)
        return out, "💡 **Other movies you might like**"

    # User match found: Recommend what the best match liked (that I haven't seen)
    user_based_recs = [mid for mid in best_match_likes if mid not in my_set]

    # Pad with diverse item-based recommendations using MMR
    remaining_n = top_n - len(user_based_recs)
    padding_candidates = [idx for idx in allowed_candidates if _movie_ids[idx] not in user_based_recs]
    item_based_recs = _apply_mmr(cf_scores, _movie_ids, _user_item_matrix, padding_candidates, top_n=remaining_n) if remaining_n > 0 else []

    final_recs = user_based_recs + item_based_recs
    final_recs = final_recs[:top_n]
    
    if not final_recs:
        return None, f"Your taste perfectly matches **{best_match_user}**! But they haven't liked anything you haven't already seen."

    # Build the dataframe for recommendations
    out = _movies[_movies["movieId"].isin(final_recs)][["movieId", "title", "genres"]].copy()
    
    # Scale Expected Ratings (User match = ~4.5 to 5.0, Item fallback = 3.5 to 4.5)
    def get_rating(mid):
        if mid in user_based_recs:
            return 4.5 + (0.5 * best_match_score)  # e.g., 80% match -> 4.9 stars
        return 3.5 + 1.0 * (cf_scores[_movie_id_to_row[mid]] / max_cf)
        
    out["rating"] = out["movieId"].map(get_rating)
    out = out.sort_values("rating", ascending=False).head(top_n).reset_index(drop=True)
    
    explanation = "💡 **Other movies you might like**"
    return out, explanation

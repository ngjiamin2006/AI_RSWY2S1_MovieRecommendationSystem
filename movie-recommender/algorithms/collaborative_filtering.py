import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from algorithms.ranking import select_top_n


def recommend(movies, user_item_matrix: csr_matrix, movie_ids: np.ndarray, movie_id_to_row: dict,
              liked_movie_ids: list[int] | None = None, top_n: int = 10,
              allowed_ids: set | None = None, pool_size: int | None = None, sample_seed: int | None = None):

    if not liked_movie_ids:
        return None

    liked_rows = [movie_id_to_row[mid] for mid in liked_movie_ids if mid in movie_id_to_row]
    if not liked_rows:
        return None

    liked_vectors = user_item_matrix[liked_rows]  # (n_liked, n_users)
    sims = cosine_similarity(liked_vectors, user_item_matrix)  # (n_liked, n_movies)
    scores = sims.mean(axis=0)  # average similarity to each liked movie

    exclude = set(liked_movie_ids)
    positive_mask = scores > 0
    candidate_allowed = set(movie_ids[positive_mask]) if allowed_ids is None else allowed_ids & set(movie_ids[positive_mask])
    results = select_top_n(scores, movie_ids, exclude, candidate_allowed, top_n, pool_size, sample_seed)

    if not results:
        return None

    out = movies[movies["movieId"].isin(results)][["movieId", "title", "genres"]].copy()
    out["score"] = out["movieId"].map(lambda mid: scores[movie_id_to_row[mid]])
    return out.sort_values("score", ascending=False).reset_index(drop=True)


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


def recommend_user_based(movies, user_item_matrix: csr_matrix, movie_ids: np.ndarray, movie_id_to_row: dict,
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
        intersection = len(my_set & their_set)
        union = len(my_set | their_set)
        jaccard = intersection / union if union > 0 else 0
        
        if jaccard > best_match_score:
            best_match_score = jaccard
            best_match_user = other_user
            best_match_likes = their_likes

    if not best_match_user or best_match_score == 0:
        return None, "We couldn't find any other local users with similar tastes yet. Try switching to another user and liking the same movies!"

    # Recommend what the best match liked, but I haven't seen
    user_based_recs = [mid for mid in best_match_likes if mid not in my_set]

    # We also get the standard item-based recommendations to pad the list
    # so the user always sees 10 movies, making the UI look consistent.
    liked_rows = [movie_id_to_row[mid] for mid in my_likes if mid in movie_id_to_row]
    cf_scores = np.zeros(len(movie_ids))
    if liked_rows:
        liked_vectors = user_item_matrix[liked_rows]
        cf_scores = cosine_similarity(liked_vectors, user_item_matrix).mean(axis=0)
        
    order = np.argsort(-cf_scores)
    item_based_recs = []
    for idx in order:
        mid = movie_ids[idx]
        if mid not in my_set and mid not in user_based_recs and cf_scores[idx] > 0:
            item_based_recs.append(mid)
            if len(item_based_recs) >= top_n:
                break

    final_recs = user_based_recs + item_based_recs
    final_recs = final_recs[:top_n]
    
    if not final_recs:
        return None, f"Your taste perfectly matches **{best_match_user}**! But they haven't liked anything you haven't already seen."

    # Build the dataframe for recommendations
    out = movies[movies["movieId"].isin(final_recs)][["movieId", "title", "genres"]].copy()
    
    # Map scores: give a high dummy score (e.g. 1.0) to the user-based ones to keep them at the top,
    # and the actual cf_scores to the rest
    def get_score(mid):
        if mid in user_based_recs:
            return 1.0 + best_match_score  # Ensure they stay at the very top
        return cf_scores[movie_id_to_row[mid]]
        
    out["score"] = out["movieId"].map(get_score)
    out = out.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    
    explanation = f"💡 **Collaborative Effect:** We noticed your taste matches **{best_match_user}** (Similarity: {best_match_score*100:.0f}%). We've put their favorites at the top of your list!"
    return out, explanation

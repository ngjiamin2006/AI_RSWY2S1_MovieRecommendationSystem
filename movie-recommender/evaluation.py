"""Offline evaluation shared by all three algorithms.

This is what produces the metrics table required by the assignment
rubric (precision/recall/F1, RMSE) -- it is separate from the live
"Like" button demo in app.py. Run this file directly to print a
comparison table, or import `evaluate_all` from app.py's evaluation tab.

=== Evaluation Metrics Explained ===
1. Precision@K (准确率): 
   - Out of the Top-K movies recommended to the user, what percentage did the user ACTUALLY like? (Measures how accurate the recommendations are).
2. Recall@K (召回率): 
   - Out of ALL the hidden movies the user liked in the test set, what percentage did we successfully catch in our Top-K list? (Measures how many good movies we didn't miss).
3. F1@K (F1分数): 
   - The harmonic mean of Precision and Recall. A balanced single score to judge overall recommendation quality.
4. Coverage (覆盖率): 
   - What percentage of the entire movie database did the algorithm recommend across ALL users? (High coverage = it recommends a wide variety of movies, not just the top 10 blockbusters).
5. Diversity (多样性): 
   - How different are the movies within a single user's Top-K list? (Calculated via genre dissimilarity; prevents recommending 10 identical superhero movies).
6. Avg Time (平均耗时): 
   - How many seconds it takes to generate recommendations for one user.
7. RMSE / MSE / MAE (误差指标): 
   - Root Mean Squared Error / Mean Squared Error / Mean Absolute Error. 
   - Measures how far off our predicted 0-5 star rating was from the user's actual rating. (Lower is better).
8. Accuracy (±1 Star): 
   - What percentage of our rating predictions were within 1 star of the actual rating.
====================================
"""


import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import (
    load_data, build_genre_matrix, build_user_item_matrix,
    load_tmdb_metadata, attach_tmdb_metadata, build_tfidf_matrix, build_cb_overview_matrix,
)
from algorithms import content_based, collaborative_filtering, hybrid

LIKE_THRESHOLD = 4.0  # ratings >= this count as "the user liked it"


def train_test_split_ratings(ratings: pd.DataFrame, test_size: float = 0.2, min_ratings: int = 10, seed: int = 42):
    """Per-user split: each user keeps some ratings for training and holds out the rest for testing.

    Vectorized (no per-user Python loop): shuffle everything once, then use
    `groupby().cumcount()` to pick each user's first `test_size` share of
    their own shuffled rows as test. A per-user `.groupby().sample()` loop
    works fine on ml-latest-small (610 users) but is a genuine bottleneck at
    ml-25m's ~162k users -- cumcount/transform are vectorized in pandas,
    a per-group Python loop isn't.
    """
    shuffled = ratings.sample(frac=1, random_state=seed).reset_index(drop=True)
    counts = shuffled.groupby("userId")["userId"].transform("count")
    rank_within_user = shuffled.groupby("userId").cumcount()
    n_test = np.maximum(1, (counts * test_size).astype(int))

    is_test = (counts >= min_ratings) & (rank_within_user < n_test)
    test = shuffled[is_test]
    train = shuffled[~is_test]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def precision_recall_f1_at_k(recommended_ids: list, relevant_ids: set, k: int):
    if not recommended_ids:
        return 0.0, 0.0, 0.0
    recommended_ids = recommended_ids[:k]
    hits = len(set(recommended_ids) & relevant_ids)
    precision = hits / len(recommended_ids)
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def intra_list_diversity(recommended_ids: list, genre_matrix: np.ndarray, movie_id_to_row: dict) -> float | None:
    """1 - average pairwise genre similarity within a recommendation list.

    Low diversity means a list is genre-homogeneous (e.g. ten near-identical
    animated kids' movies); high diversity means the list spans different
    kinds of films. Undefined (None) for lists shorter than 2 items.
    """
    rows = [movie_id_to_row[mid] for mid in recommended_ids if mid in movie_id_to_row]
    if len(rows) < 2:
        return None
    vectors = genre_matrix[rows]
    sims = cosine_similarity(vectors)
    n = len(rows)
    off_diagonal_sum = sims.sum() - np.trace(sims)
    mean_similarity = off_diagonal_sum / (n * (n - 1))
    return 1.0 - mean_similarity


def evaluate_all(movies, train_ratings, test_ratings, k: int = 10, max_users: int = 100, seed: int = 42,
                  links: pd.DataFrame | None = None):
    """Compute precision/recall/F1@K, coverage, and diversity for all algorithms,
    plus RMSE/MAE and per-call recommendation time for collaborative filtering.

    If `links` (links.csv) is provided, also evaluates the TMDb-enriched
    TF-IDF hybrid variant (`hybrid_tfidf`) alongside the genre-only one,
    so the two feature sets can be compared side by side in the same table.

    content_based is search-driven only (per tutor feedback, never
    Like-button-driven) -- each test user's single highest-rated training
    movie stands in as "the movie they searched for," scored against the
    same held-out relevant set as everyone else.

    - Coverage: fraction of the whole catalog that appears at least once
      across every test user's recommendation list for that algorithm.
    - Diversity: mean intra-list genre dissimilarity (see
      `intra_list_diversity`), averaged across users.
    - Recommendation time: mean wall-clock seconds per `recommend()` call.

    Returns a DataFrame indexed by algorithm name.
    """
    movie_ids, genre_matrix, genre_names = build_genre_matrix(movies)
    user_item_matrix, movie_id_to_row, user_id_to_col = build_user_item_matrix(train_ratings, movie_ids)

    algorithm_names = ["collaborative", "content_based"]
    tfidf_matrix = vectorizer = None
    # content-based's own genre+overview search matrix -- built unconditionally
    # (degrades to genre-only without TMDb, same as the live Content-Based tab).
    enriched = movies
    if links is not None:
        enriched = attach_tmdb_metadata(movies, links)
        tfidf_movie_ids, tfidf_matrix, vectorizer = build_tfidf_matrix(enriched)
        assert (tfidf_movie_ids == movie_ids).all(), "TF-IDF row order must match genre_matrix row order"
        algorithm_names += ["hybrid_tfidf"]
    cb_search_movie_ids, cb_matrix, _ = build_cb_overview_matrix(enriched)
    assert (cb_search_movie_ids == movie_ids).all(), "content-based matrix row order must match genre_matrix row order"

    rng = np.random.default_rng(seed)
    test_users = test_ratings["userId"].unique()
    if len(test_users) > max_users:
        test_users = rng.choice(test_users, size=max_users, replace=False)

    metrics = {name: {"precision": [], "recall": [], "f1": [], "time": []} for name in algorithm_names}
    recommended_sets = {name: set() for name in algorithm_names}
    diversity_scores = {name: [] for name in algorithm_names}
    squared_errors = {name: [] for name in algorithm_names}
    absolute_errors = {name: [] for name in algorithm_names}

    for user_id in test_users:
        user_train = train_ratings[train_ratings["userId"] == user_id]
        liked = user_train[user_train["rating"] >= LIKE_THRESHOLD]["movieId"].tolist()
        if not liked:
            continue

        user_test = test_ratings[test_ratings["userId"] == user_id]
        relevant = set(user_test[user_test["rating"] >= LIKE_THRESHOLD]["movieId"])
        if not relevant:
            continue

        def timed(fn, *args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            return result, time.perf_counter() - start

        # Computed once and reused below -- collaborative/hybrid/hybrid_tfidf would
        # otherwise each redo this same expensive similarity call independently.
        cf_scores = collaborative_filtering.compute_cf_scores(user_item_matrix, movie_id_to_row, liked)

        cf, cf_t = timed(collaborative_filtering.recommend, movies, user_item_matrix, movie_ids, movie_id_to_row,
                          liked_movie_ids=liked, top_n=k, _cf_scores=cf_scores)
        recs_by_name = {"collaborative": (cf, cf_t)}

        # content-based: search-driven only, never a liked_movie_ids profile.
        # Stand-in for "the user searched for a movie" -- their single
        # highest-rated training movie (the one they'd most plausibly search
        # for) becomes the query, scored against the same held-out relevant
        # set as everyone else.
        seed_movie_id = int(user_train.loc[user_train["rating"].idxmax(), "movieId"])
        cb, cb_t = timed(content_based.recommend_by_search, movies, cb_matrix, movie_ids, movie_id_to_row,
                          matched_movie_ids=[seed_movie_id], top_n=k)
        recs_by_name["content_based"] = (cb, cb_t)

        if tfidf_matrix is not None:
            hyt, hyt_t = timed(hybrid.recommend_tfidf, movies, tfidf_matrix, movie_ids, movie_id_to_row, vectorizer,
                               user_item_matrix, liked_movie_ids=liked, top_n=k, _cf_scores=cf_scores)
            recs_by_name["hybrid_tfidf"] = (hyt, hyt_t)

        for name, (recs, elapsed) in recs_by_name.items():
            ids = recs["movieId"].tolist() if recs is not None else []
            p, r, f1 = precision_recall_f1_at_k(ids, relevant, k)
            metrics[name]["precision"].append(p)
            metrics[name]["recall"].append(r)
            metrics[name]["f1"].append(f1)
            metrics[name]["time"].append(elapsed)
            recommended_sets[name].update(ids[:k])
            diversity = intra_list_diversity(ids[:k], genre_matrix, movie_id_to_row)
            if diversity is not None:
                diversity_scores[name].append(diversity)

        # hybrid's own liked-movies profile (content-based itself is
        # search-driven only and isn't scored via a profile -- these two
        # exist purely to feed the hybrid/hybrid_tfidf blend below).
        profile_cb = content_based.build_user_profile(genre_matrix, movie_id_to_row, liked)
        profile_cb_tfidf = None
        if tfidf_matrix is not None:
            profile_cb_tfidf = content_based.build_tfidf_profile(tfidf_matrix, movie_id_to_row, liked, vectorizer)
        seed_row_idx = movie_id_to_row.get(seed_movie_id)  # for content-based's search-driven rating prediction

        if user_id in user_id_to_col:
            col = user_id_to_col[user_id]
            for _, row in user_test.iterrows():
                target_mid = row["movieId"]
                actual_rating = row["rating"]
                
                if target_mid not in movie_id_to_row:
                    continue
                target_row_idx = movie_id_to_row[target_mid]
                
                preds = {}
                
                # CF prediction
                pred_cf = collaborative_filtering.predict_rating(user_item_matrix, movie_id_to_row, col, target_mid)
                preds["collaborative"] = pred_cf
                
                # CB prediction (search-driven): similarity between the seed
                # movie (this user's synthetic search query) and the held-out
                # movie -- same query used for the ranking metrics above.
                pred_cb_search = None
                if seed_row_idx is not None:
                    sim = cosine_similarity(
                        cb_matrix[seed_row_idx:seed_row_idx + 1], cb_matrix[target_row_idx:target_row_idx + 1]
                    )[0, 0]
                    pred_cb_search = max(0.0, min(5.0, sim * 5.0))
                preds["content_based"] = pred_cb_search

                # CB prediction (liked-profile, scale 0-1 similarity to 0-5 rating)
                # -- intermediate only, not the content_based row above; needed
                # here purely to feed the hybrid prediction below.
                pred_cb = None
                if profile_cb is not None:
                    target_vec = genre_matrix[target_row_idx].reshape(1, -1)
                    sim = cosine_similarity(profile_cb, target_vec)[0, 0]
                    pred_cb = max(0.0, min(5.0, sim * 5.0))

                # Basic hybrid prediction removed to only evaluate hybrid_tfidf
                alpha = 0.15
                
                # TF-IDF variants
                if tfidf_matrix is not None:
                    pred_cb_tfidf = None
                    if profile_cb_tfidf is not None:
                        target_vec = tfidf_matrix[target_row_idx]
                        if not isinstance(profile_cb_tfidf, np.ndarray):
                            profile_cb_tfidf = np.asarray(profile_cb_tfidf)
                        sim = cosine_similarity(profile_cb_tfidf.reshape(1, -1), target_vec)[0, 0]
                        pred_cb_tfidf = max(0.0, min(5.0, sim * 5.0))

                    if pred_cb_tfidf is not None and pred_cf is not None:
                        preds["hybrid_tfidf"] = alpha * pred_cb_tfidf + (1 - alpha) * pred_cf
                    elif pred_cb_tfidf is not None:
                        preds["hybrid_tfidf"] = pred_cb_tfidf
                    elif pred_cf is not None:
                        preds["hybrid_tfidf"] = pred_cf
                    else:
                        preds["hybrid_tfidf"] = None
                
                for name, pred in preds.items():
                    if pred is not None:
                        squared_errors[name].append((pred - actual_rating) ** 2)
                        absolute_errors[name].append(abs(pred - actual_rating))

    rmse = {name: float(np.sqrt(np.mean(errs))) if errs else None for name, errs in squared_errors.items()}
    mse = {name: float(np.mean(errs)) if errs else None for name, errs in squared_errors.items()}
    mae = {name: float(np.mean(errs)) if errs else None for name, errs in absolute_errors.items()}
    # Fraction of predictions within 1 star of actual -- not classification accuracy.
    accuracy_within_1star = {name: sum(1 for e in errs if e <= 1.0) / len(errs) if errs else None for name, errs in absolute_errors.items()}
    
    n_movies = len(movie_ids)

    rows = []
    for name, vals in metrics.items():
        rows.append({
            "algorithm": name,
            f"precision@{k}": np.mean(vals["precision"]) if vals["precision"] else 0.0,
            f"recall@{k}": np.mean(vals["recall"]) if vals["recall"] else 0.0,
            f"f1@{k}": np.mean(vals["f1"]) if vals["f1"] else 0.0,
            "coverage": len(recommended_sets[name]) / n_movies,
            "diversity": np.mean(diversity_scores[name]) if diversity_scores[name] else None,
            "avg_time_sec": np.mean(vals["time"]) if vals["time"] else None,
            "rmse": rmse.get(name),
            "mse": mse.get(name),
            "mae": mae.get(name),
            "accuracy_within_1star": accuracy_within_1star.get(name),
        })
    return pd.DataFrame(rows).set_index("algorithm")


if __name__ == "__main__":
    import argparse
    import os

    from data_loader import load_links

    parser = argparse.ArgumentParser(description="Run offline evaluation for the movie recommender.")
    parser.add_argument("--dataset", choices=["ml-latest-small", "ml-25m"], default="ml-25m")
    parser.add_argument("--max-users", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    movies, ratings = load_data(dataset=args.dataset)
    train_ratings, test_ratings = train_test_split_ratings(ratings)

    tmdb_path = os.path.join(os.path.dirname(__file__), "data", "tmdb", "movies_DB.csv")
    links = load_links(dataset=args.dataset) if os.path.exists(tmdb_path) else None

    results = evaluate_all(movies, train_ratings, test_ratings, k=args.k, max_users=args.max_users, links=links)
    pd.set_option("display.width", 120)
    print(f"\nDataset: {args.dataset}  (movies={len(movies)}, ratings={len(ratings)})\n")
    print(results)

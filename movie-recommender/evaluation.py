"""Offline evaluation shared by all three algorithms.

This is what produces the metrics table required by the assignment
rubric (precision/recall/F1, RMSE) -- it is separate from the live
"Like" button demo in app.py. Run this file directly to print a
comparison table, or import `evaluate_all` from app.py's evaluation tab.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import (
    load_data, build_genre_matrix, build_user_item_matrix,
    load_tmdb_metadata, attach_tmdb_metadata, build_tfidf_matrix,
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
    TF-IDF content-based and hybrid variants (`content_based_tfidf`,
    `hybrid_tfidf`) alongside the original genre-only ones, so the two
    feature sets can be compared side by side in the same table.

    - Coverage: fraction of the whole catalog that appears at least once
      across every test user's recommendation list for that algorithm.
    - Diversity: mean intra-list genre dissimilarity (see
      `intra_list_diversity`), averaged across users.
    - Recommendation time: mean wall-clock seconds per `recommend()` call.

    Returns a DataFrame indexed by algorithm name.
    """
    movie_ids, genre_matrix, genre_names = build_genre_matrix(movies)
    user_item_matrix, movie_id_to_row, user_id_to_col = build_user_item_matrix(train_ratings, movie_ids)

    algorithm_names = ["content_based", "collaborative", "hybrid"]
    tfidf_matrix = vectorizer = None
    if links is not None:
        enriched = attach_tmdb_metadata(movies, links)
        tfidf_movie_ids, tfidf_matrix, vectorizer = build_tfidf_matrix(enriched)
        assert (tfidf_movie_ids == movie_ids).all(), "TF-IDF row order must match genre_matrix row order"
        algorithm_names += ["content_based_tfidf", "hybrid_tfidf"]

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

        cb, cb_t = timed(content_based.recommend, movies, genre_matrix, movie_ids, movie_id_to_row, genre_names,
                          liked_movie_ids=liked, top_n=k)
        cf, cf_t = timed(collaborative_filtering.recommend, movies, user_item_matrix, movie_ids, movie_id_to_row,
                          liked_movie_ids=liked, top_n=k, _cf_scores=cf_scores)
        hy, hy_t = timed(hybrid.recommend, movies, genre_matrix, movie_ids, movie_id_to_row, genre_names,
                          user_item_matrix, liked_movie_ids=liked, top_n=k, _cf_scores=cf_scores)
        recs_by_name = {"content_based": (cb, cb_t), "collaborative": (cf, cf_t), "hybrid": (hy, hy_t)}

        if tfidf_matrix is not None:
            cbt, cbt_t = timed(content_based.recommend_tfidf, movies, tfidf_matrix, movie_ids, movie_id_to_row,
                               vectorizer, liked_movie_ids=liked, top_n=k)
            hyt, hyt_t = timed(hybrid.recommend_tfidf, movies, tfidf_matrix, movie_ids, movie_id_to_row, vectorizer,
                               user_item_matrix, liked_movie_ids=liked, top_n=k, _cf_scores=cf_scores)
            recs_by_name["content_based_tfidf"] = (cbt, cbt_t)
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

        profile_cb = content_based.build_user_profile(genre_matrix, movie_id_to_row, liked)
        profile_cb_tfidf = None
        if tfidf_matrix is not None:
            profile_cb_tfidf = content_based.build_tfidf_profile(tfidf_matrix, movie_id_to_row, liked, vectorizer)

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
                
                # CB prediction (scale 0-1 similarity to 0-5 rating)
                pred_cb = None
                if profile_cb is not None:
                    target_vec = genre_matrix[target_row_idx].reshape(1, -1)
                    sim = cosine_similarity(profile_cb, target_vec)[0, 0]
                    pred_cb = max(0.0, min(5.0, sim * 5.0))
                preds["content_based"] = pred_cb
                
                # HY prediction
                alpha = 0.5
                if pred_cb is not None and pred_cf is not None:
                    preds["hybrid"] = alpha * pred_cb + (1 - alpha) * pred_cf
                elif pred_cb is not None:
                    preds["hybrid"] = pred_cb
                elif pred_cf is not None:
                    preds["hybrid"] = pred_cf
                else:
                    preds["hybrid"] = None
                
                # TF-IDF variants
                if tfidf_matrix is not None:
                    pred_cb_tfidf = None
                    if profile_cb_tfidf is not None:
                        target_vec = tfidf_matrix[target_row_idx]
                        if not isinstance(profile_cb_tfidf, np.ndarray):
                            profile_cb_tfidf = np.asarray(profile_cb_tfidf)
                        sim = cosine_similarity(profile_cb_tfidf.reshape(1, -1), target_vec)[0, 0]
                        pred_cb_tfidf = max(0.0, min(5.0, sim * 5.0))
                    preds["content_based_tfidf"] = pred_cb_tfidf
                    
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

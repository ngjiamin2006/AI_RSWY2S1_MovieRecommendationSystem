"""Offline evaluation shared by all three algorithms.

This is what produces the metrics table required by the assignment
rubric (precision/recall/F1, RMSE) -- it is separate from the live
"Like" button demo in app.py. Run this file directly to print a
comparison table, or import `evaluate_all` from app.py's evaluation tab.
"""
import numpy as np
import pandas as pd

from data_loader import load_data, build_genre_matrix, build_user_item_matrix
from algorithms import content_based, collaborative_filtering, hybrid

LIKE_THRESHOLD = 4.0  # ratings >= this count as "the user liked it"


def train_test_split_ratings(ratings: pd.DataFrame, test_size: float = 0.2, min_ratings: int = 10, seed: int = 42):
    """Per-user split: each user keeps some ratings for training and holds out the rest for testing."""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for user_id, group in ratings.groupby("userId"):
        if len(group) < min_ratings:
            train_parts.append(group)  # not enough data to fairly test this user
            continue
        shuffled = group.sample(frac=1, random_state=rng.integers(1e9))
        n_test = max(1, int(len(shuffled) * test_size))
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])

    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def precision_recall_f1_at_k(recommended_ids: list, relevant_ids: set, k: int):
    if not recommended_ids:
        return 0.0, 0.0, 0.0
    recommended_ids = recommended_ids[:k]
    hits = len(set(recommended_ids) & relevant_ids)
    precision = hits / len(recommended_ids)
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_all(movies, train_ratings, test_ratings, k: int = 10, max_users: int = 100, seed: int = 42):
    """Compute precision/recall/F1@K for all three algorithms, plus RMSE for collaborative filtering.

    Returns a DataFrame indexed by algorithm name.
    """
    movie_ids, genre_matrix, genre_names = build_genre_matrix(movies)
    user_item_matrix, movie_id_to_row, user_id_to_col = build_user_item_matrix(train_ratings, movie_ids)

    rng = np.random.default_rng(seed)
    test_users = test_ratings["userId"].unique()
    if len(test_users) > max_users:
        test_users = rng.choice(test_users, size=max_users, replace=False)

    metrics = {name: {"precision": [], "recall": [], "f1": []} for name in ("content_based", "collaborative", "hybrid")}
    squared_errors = []

    for user_id in test_users:
        user_train = train_ratings[train_ratings["userId"] == user_id]
        liked = user_train[user_train["rating"] >= LIKE_THRESHOLD]["movieId"].tolist()
        if not liked:
            continue

        user_test = test_ratings[test_ratings["userId"] == user_id]
        relevant = set(user_test[user_test["rating"] >= LIKE_THRESHOLD]["movieId"])
        if not relevant:
            continue

        cb = content_based.recommend(movies, genre_matrix, movie_ids, movie_id_to_row, genre_names,
                                      liked_movie_ids=liked, top_n=k)
        cf = collaborative_filtering.recommend(movies, user_item_matrix, movie_ids, movie_id_to_row,
                                                liked_movie_ids=liked, top_n=k)
        hy = hybrid.recommend(movies, genre_matrix, movie_ids, movie_id_to_row, genre_names,
                               user_item_matrix, liked_movie_ids=liked, top_n=k)

        for name, recs in (("content_based", cb), ("collaborative", cf), ("hybrid", hy)):
            ids = recs["movieId"].tolist() if recs is not None else []
            p, r, f1 = precision_recall_f1_at_k(ids, relevant, k)
            metrics[name]["precision"].append(p)
            metrics[name]["recall"].append(r)
            metrics[name]["f1"].append(f1)

        if user_id in user_id_to_col:
            col = user_id_to_col[user_id]
            for _, row in user_test.iterrows():
                pred = collaborative_filtering.predict_rating(user_item_matrix, movie_id_to_row, col, row["movieId"])
                if pred is not None:
                    squared_errors.append((pred - row["rating"]) ** 2)

    rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else None

    rows = []
    for name, vals in metrics.items():
        rows.append({
            "algorithm": name,
            f"precision@{k}": np.mean(vals["precision"]) if vals["precision"] else 0.0,
            f"recall@{k}": np.mean(vals["recall"]) if vals["recall"] else 0.0,
            f"f1@{k}": np.mean(vals["f1"]) if vals["f1"] else 0.0,
            "rmse (collaborative only)": rmse if name == "collaborative" else None,
        })
    return pd.DataFrame(rows).set_index("algorithm")


if __name__ == "__main__":
    movies, ratings = load_data()
    train_ratings, test_ratings = train_test_split_ratings(ratings)
    results = evaluate_all(movies, train_ratings, test_ratings, k=10, max_users=100)
    pd.set_option("display.width", 120)
    print(results)

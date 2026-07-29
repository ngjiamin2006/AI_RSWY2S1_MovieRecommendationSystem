"""Shared data loading utilities for the movie recommender.

Framework-agnostic (no Streamlit here) so it can be reused by app.py,
evaluation.py, and any offline experiment notebooks.
"""
import ast
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TMDB_PATH = os.path.join(DATA_DIR, "tmdb", "movies_DB.csv")


def tmdb_available() -> bool:
    """Whether the (separately downloaded, gitignored) TMDb metadata file is present."""
    return os.path.exists(TMDB_PATH)

# "ml-latest-small" ships in the repo directly under data/. "ml-25m" is the
# larger benchmark -- too big for git, download separately (see README) into
# data/ml-25m/. Both directories use the same movies.csv/ratings.csv/links.csv
# schema, so everything downstream is agnostic to which one is active.
DATASET_DIRS = {
    "ml-latest-small": DATA_DIR,
    "ml-25m": os.path.join(DATA_DIR, "ml-25m"),
}


def load_data(dataset: str = "ml-latest-small"):
    """Load movies and ratings tables for the given MovieLens dataset.

    ratings.csv is loaded with narrower dtypes and without the (unused)
    timestamp column -- matters at ml-25m's scale (25M rows), harmless at
    ml-latest-small's.
    """
    data_dir = DATASET_DIRS[dataset]
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.csv"),
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )

    # "(no genres listed)" means no genre info -> treat as empty list
    movies["genre_list"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    return movies, ratings


def load_links(dataset: str = "ml-latest-small") -> pd.DataFrame:
    """Load links.csv (movieId -> imdbId/tmdbId) for the given dataset."""
    return pd.read_csv(os.path.join(DATASET_DIRS[dataset], "links.csv"))


def build_genre_matrix(movies: pd.DataFrame):
    """One-hot encode each movie's genres.

    Returns:
        movie_ids: array of movieId in row order matching `matrix`
        matrix: (n_movies, n_genres) binary numpy array
        genre_names: list of genre column names matching matrix columns
    """
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(movies["genre_list"])
    movie_ids = movies["movieId"].to_numpy()
    return movie_ids, matrix.astype(float), list(mlb.classes_)


def build_user_item_matrix(ratings: pd.DataFrame, movie_ids: np.ndarray):
    """Build a sparse (movies x users) ratings matrix.

    Row order matches `movie_ids` so it lines up with the genre matrix,
    which makes it easy to combine signals in hybrid.py.
    """
    movie_id_to_row = {mid: i for i, mid in enumerate(movie_ids)}
    user_ids = np.sort(ratings["userId"].unique())
    user_id_to_col = {uid: i for i, uid in enumerate(user_ids)}

    rows = ratings["movieId"].map(movie_id_to_row)
    cols = ratings["userId"].map(user_id_to_col)
    valid = rows.notna()
    rows = rows[valid].astype(int).to_numpy()
    cols = cols[valid].astype(int).to_numpy()
    data = ratings.loc[valid, "rating"].to_numpy()

    matrix = csr_matrix((data, (rows, cols)), shape=(len(movie_ids), len(user_ids)))
    return matrix, movie_id_to_row, user_id_to_col


def get_popular_movies(movies: pd.DataFrame, ratings: pd.DataFrame, top_n: int = 10, min_ratings: int = 20):
    """"Top Rated" -- highest average rating among movies with enough votes.

    Also used as the cold-start fallback when an algorithm has no signal yet.
    """
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_ratings]
    top_ids = stats.sort_values(["mean", "count"], ascending=False).head(top_n).index
    result = movies[movies["movieId"].isin(top_ids)][["movieId", "title", "genres"]].copy()
    result["score"] = result["movieId"].map(stats["mean"])
    return result.sort_values("score", ascending=False).reset_index(drop=True)


def get_most_rated_movies(movies: pd.DataFrame, ratings: pd.DataFrame, top_n: int = 10):
    """"Popular" -- highest rating COUNT (most-watched/most-discussed), regardless of average score."""
    counts = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    top_ids = counts.sort_values("count", ascending=False).head(top_n).index
    result = movies[movies["movieId"].isin(top_ids)][["movieId", "title", "genres"]].copy()
    result["score"] = result["movieId"].map(counts["count"])
    return result.sort_values("score", ascending=False).reset_index(drop=True)


def get_new_releases(movies_enriched: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """"New Releases" -- sorted by TMDb release_date, which is far more precise
    than parsing the year out of a MovieLens title string.
    """
    valid = movies_enriched.dropna(subset=["release_date"]).copy()
    valid = valid.sort_values("release_date", ascending=False)
    return valid[["movieId", "title", "genres", "release_date", "poster_url"]].head(top_n).reset_index(drop=True)


def _parse_stringified_list(value):
    """TMDb CSV stores lists (genres, keywords, cast) as Python-repr strings, e.g. "['a', 'b']"."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def load_tmdb_metadata(path: str = TMDB_PATH) -> pd.DataFrame:
    """Load the TMDb 100k metadata CSV (overview, keywords, cast, director, poster, release date).

    Row-level columns are parsed once here so downstream code works with real
    Python lists instead of stringified ones. Parsing ~95k stringified
    keywords/cast lists with ast.literal_eval takes ~40s, and this file
    doesn't change between runs, so the parsed result is cached to a pickle
    next to the source CSV -- reused (and re-validated by mtime) on every
    later call instead of re-parsing from scratch.

    Pickle is not portable across pandas/Python versions/environments, so a
    cache written by one machine/venv can fail to load on another (seen in
    practice: a pandas-internal `issubclass()` TypeError on version
    mismatch). Any read failure is treated as a cache miss -- fall back to
    re-parsing and overwrite it with a fresh, compatible cache.
    """
    cache_path = path + ".parsed.pkl"
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(path):
        try:
            return pd.read_pickle(cache_path)
        except Exception:
            pass  # stale/incompatible cache from a different environment -- re-parse below

    tmdb = pd.read_csv(path, usecols=[
        "movie_id", "overview", "genres", "keywords", "director",
        "cast", "release_date", "poster_url",
    ])
    tmdb["keywords_list"] = tmdb["keywords"].apply(_parse_stringified_list)
    tmdb["cast_list"] = tmdb["cast"].apply(_parse_stringified_list)
    tmdb["cast_names"] = tmdb["cast_list"].apply(
        lambda people: [p["name"] for p in people if isinstance(p, dict) and p.get("name")]
    )
    try:
        tmdb.to_pickle(cache_path)
    except OSError:
        pass  # caching is a pure optimization -- a failed write shouldn't break loading
    return tmdb


def attach_tmdb_metadata(movies: pd.DataFrame, links: pd.DataFrame, tmdb: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join MovieLens movies onto TMDb metadata via links.csv's tmdbId -> TMDb movie_id.

    Movies with no TMDb match (catalogs don't perfectly overlap) keep all
    their MovieLens columns (title/genres) with TMDb fields left as NaN --
    they still work via genre-only content signal, they just don't get the
    richer TF-IDF text or a poster.
    """
    if tmdb is None:
        tmdb = load_tmdb_metadata()
    tmdb = tmdb.drop_duplicates(subset="movie_id", keep="first")
    # TMDb's own "genres" is unused downstream (build_content_soup uses MovieLens's
    # genre_list) and collides with MovieLens's "genres" column name in the merge
    # below, which pandas would otherwise silently rename to genres_x/genres_y.
    tmdb = tmdb.drop(columns=["genres"])
    merged = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    merged = merged.merge(tmdb, left_on="tmdbId", right_on="movie_id", how="left")
    return merged


def build_content_soup(movies_enriched: pd.DataFrame) -> pd.Series:
    """Combine genres + keywords + top cast + director into one text blob per movie for TF-IDF.

    Genres always come from MovieLens (every row has them); the rest is only
    present where a TMDb match existed. Names are stripped of internal spaces
    ("Tom Hanks" -> "TomHanks") so a person is one token, not two that could
    spuriously match unrelated actors/directors sharing a first or last name.
    """
    def soup(row):
        tokens = list(row["genre_list"])
        if isinstance(row.get("keywords_list"), list):
            tokens += row["keywords_list"]
        if isinstance(row.get("cast_names"), list):
            tokens += row["cast_names"][:5]  # top-billed cast only
        if pd.notna(row.get("director")):
            tokens.append(row["director"])
        return " ".join(str(t).replace(" ", "") for t in tokens)

    return movies_enriched.apply(soup, axis=1)


def build_tfidf_matrix(movies_enriched: pd.DataFrame, max_features: int = 5000):
    """Vectorize the content soup with TF-IDF.

    Returns:
        movie_ids: array of movieId in row order matching `matrix`
        matrix: sparse (n_movies, n_features) TF-IDF matrix
        vectorizer: fitted TfidfVectorizer, reused to embed onboarding genre
            picks into the same feature space (see algorithms/content_based.py)
    """
    soup = build_content_soup(movies_enriched)
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(soup)
    movie_ids = movies_enriched["movieId"].to_numpy()
    return movie_ids, matrix, vectorizer

"""Streamlit app tying together onboarding, all three algorithms, and evaluation.

Run with: streamlit run app.py
"""
import os

import streamlit as st  # type: ignore[missing-import]

from data_loader import (
    load_data, build_genre_matrix, build_user_item_matrix, get_popular_movies,
    get_most_rated_movies, get_new_releases, load_links, attach_tmdb_metadata,
    build_tfidf_matrix, tmdb_available, DATASET_DIRS,
)
from algorithms import content_based, collaborative_filtering, hybrid
import evaluation

st.set_page_config(page_title="Movie Recommender", layout="wide")

TMDB_ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."


@st.cache_resource(show_spinner="Loading dataset and building recommendation features (first load can take a minute on the 25M dataset)...")
def load_pipeline(dataset: str):
    """Load everything the app needs for one dataset, once per dataset per session.

    st.cache_resource (not cache_data) is used deliberately: the return value
    holds large in-memory objects (a 62k x 162k sparse matrix at ml-25m
    scale) that should be cached by reference, not re-hashed/copied on every
    Streamlit rerun the way cache_data would try to do.
    """
    movies, ratings = load_data(dataset=dataset)
    movie_ids, genre_matrix, genre_names = build_genre_matrix(movies)
    user_item_matrix, movie_id_to_row, user_id_to_col = build_user_item_matrix(ratings, movie_ids)

    enriched = movies
    tfidf_matrix = vectorizer = None
    tmdb_error = None
    if tmdb_available():
        try:
            links = load_links(dataset=dataset)
            enriched = attach_tmdb_metadata(movies, links)
            _, tfidf_matrix, vectorizer = build_tfidf_matrix(enriched)
        except Exception as exc:  # malformed/partial TMDb download shouldn't crash the whole app
            tmdb_error = str(exc)
    else:
        tmdb_error = "TMDb dataset not found at movie-recommender/data/tmdb/ -- posters and rich content features are unavailable. See README for the download step."

    poster_lookup = (
        dict(zip(enriched["movieId"], enriched.get("poster_url", []))) if "poster_url" in enriched.columns else {}
    )

    return {
        "movies": movies, "ratings": ratings, "enriched": enriched,
        "movie_ids": movie_ids, "genre_matrix": genre_matrix, "genre_names": genre_names,
        "user_item_matrix": user_item_matrix, "movie_id_to_row": movie_id_to_row,
        "user_id_to_col": user_id_to_col, "tfidf_matrix": tfidf_matrix, "vectorizer": vectorizer,
        "poster_lookup": poster_lookup, "tmdb_error": tmdb_error,
    }


available_datasets = [name for name in DATASET_DIRS if os.path.exists(DATASET_DIRS[name])]

with st.sidebar:
    st.write("### Dataset")
    dataset = st.selectbox("MovieLens version", available_datasets, index=0)

pipeline = load_pipeline(dataset)
movies, ratings = pipeline["movies"], pipeline["ratings"]
enriched = pipeline["enriched"]
movie_ids, genre_matrix, genre_names = pipeline["movie_ids"], pipeline["genre_matrix"], pipeline["genre_names"]
user_item_matrix = pipeline["user_item_matrix"]
movie_id_to_row, user_id_to_col = pipeline["movie_id_to_row"], pipeline["user_id_to_col"]
tfidf_matrix, vectorizer = pipeline["tfidf_matrix"], pipeline["vectorizer"]
poster_lookup = pipeline["poster_lookup"]

if pipeline["tmdb_error"]:
    st.sidebar.warning(pipeline["tmdb_error"])

if "local_profiles" not in st.session_state:
    st.session_state.local_profiles = {
        "User 1": [],
        "User 2": [],
        "User 3": [],
        "User 4 (Target)": []
    }
if "current_user" not in st.session_state:
    st.session_state.current_user = "User 1"
if "liked_movie_ids" not in st.session_state:
    st.session_state.liked_movie_ids = st.session_state.local_profiles[st.session_state.current_user]
if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []

st.title(":clapper: Movie Recommender")


def render_recommendations(df, key_prefix, show_score=True, score_label="score"):
    if df is None or df.empty:
        st.info("Not enough signal yet for this algorithm -- showing popular movies instead.")
        df = get_popular_movies(movies, ratings, top_n=10)
    for _, row in df.iterrows():
        cols = st.columns([1, 5, 2, 1])
        poster = poster_lookup.get(row["movieId"])
        if isinstance(poster, str):
            cols[0].image(poster, width=70)
        else:
            cols[0].write(":clapper:")
        cols[1].write(f"**{row['title']}**  \n{row['genres']}")
        if show_score and "score" in row:
            cols[2].write(f"{score_label}: {row['score']:.3f}")
        elif "release_date" in row:
            cols[2].write(str(row["release_date"]))
        if cols[3].button(":+1: Like", key=f"{key_prefix}_{row['movieId']}"):
            if row["movieId"] not in st.session_state.liked_movie_ids:
                st.session_state.liked_movie_ids.append(row["movieId"])
            st.rerun()


with st.sidebar:
    st.write("### 👤 Select Local User")
    st.caption("Switch users to simulate a collaborative environment.")
    
    selected_user = st.selectbox(
        "Current User",
        list(st.session_state.local_profiles.keys()),
        index=list(st.session_state.local_profiles.keys()).index(st.session_state.current_user)
    )
    
    if selected_user != st.session_state.current_user:
        st.session_state.current_user = selected_user
        # Point the shared liked_movie_ids to the newly selected user's list
        st.session_state.liked_movie_ids = st.session_state.local_profiles[selected_user]
        st.rerun()

    st.divider()

    st.write("### Your liked movies")
    if st.session_state.liked_movie_ids:
        liked_titles = movies[movies["movieId"].isin(st.session_state.liked_movie_ids)]["title"]
        for t in liked_titles:
            st.write(f"- {t}")
    else:
        st.write("_None yet -- click Like on a recommendation._")
    if st.button("Reset all users"):
        st.session_state.clear()
        st.rerun()

(tab_home, tab_popular, tab_top_rated, tab_new, tab_cb, tab_cf, tab_hy, tab_eval) = st.tabs(
    ["Home", "Popular", "Top Rated", "New Releases", "Content-Based",
     "Collaborative Filtering", "Hybrid", "Evaluation"]
)

with tab_home:
    st.subheader("Welcome back!")
    c1, c2, c3 = st.columns(3)
    c1.metric("Movies", f"{len(movies):,}")
    c2.metric("Ratings", f"{len(ratings):,}")
    c3.metric("Users", f"{ratings['userId'].nunique():,}")
    
    st.divider()
    st.write("### What do you like to watch?")
    st.session_state.selected_genres = st.multiselect(
        "Pick a few genres you enjoy (Cold Start Preferences)", 
        options=genre_names, 
        default=st.session_state.selected_genres
    )
    
    st.caption(
        "Content-Based, Collaborative Filtering, and Hybrid tabs use TMDb-enriched TF-IDF features "
        "(overview/keywords/cast/director) when available, falling back to genre-only similarity otherwise."
    )

with tab_popular:
    st.caption("Most-watched movies (highest number of ratings), regardless of average score.")
    render_recommendations(get_most_rated_movies(movies, ratings, top_n=10), "pop", score_label="ratings")

with tab_top_rated:
    st.caption("Highest average rating among movies with enough votes to be reliable.")
    render_recommendations(get_popular_movies(movies, ratings, top_n=10), "top", score_label="avg rating")

with tab_new:
    st.caption("Most recently released movies with TMDb metadata (requires the TMDb dataset).")
    if tfidf_matrix is not None:
        render_recommendations(get_new_releases(enriched, top_n=10), "new", show_score=False)
    else:
        st.info("New Releases needs the TMDb dataset (for release dates) -- see the sidebar warning above.")

with tab_cb:
    st.caption("Recommends movies with similar content to what you've liked (or picked at onboarding).")
    if tfidf_matrix is not None:
        recs = content_based.recommend_tfidf(
            movies, tfidf_matrix, movie_ids, movie_id_to_row, vectorizer,
            liked_movie_ids=st.session_state.liked_movie_ids,
            selected_genres=st.session_state.selected_genres, top_n=10,
        )
    else:
        recs = content_based.recommend(
            movies, genre_matrix, movie_ids, movie_id_to_row, genre_names,
            liked_movie_ids=st.session_state.liked_movie_ids,
            selected_genres=st.session_state.selected_genres, top_n=10,
        )
    render_recommendations(recs, "cb")

with tab_cf:
    st.caption("Interactive Demo: Recommends movies liked by other Local Users with similar taste to you.")
    
    recs, explanation = collaborative_filtering.recommend_user_based(
        movies, user_item_matrix, movie_ids, movie_id_to_row,
        current_user=st.session_state.current_user,
        local_profiles=st.session_state.local_profiles, top_n=10
    )
    
    if explanation:
        st.info(explanation)
        
    if recs is not None and not recs.empty:
        render_recommendations(recs, "cf")
    else:
        # Fall back to item-based if no user matches, or if we got None
        st.warning("Falling back to standard Item-Based CF from the full dataset (no local users matched).")
        recs_item_based = collaborative_filtering.recommend(
            movies, user_item_matrix, movie_ids, movie_id_to_row,
            liked_movie_ids=st.session_state.liked_movie_ids, top_n=10,
        )
        render_recommendations(recs_item_based, "cf")

with tab_hy:
    st.caption("Blends the content-based and collaborative filtering scores.")
    alpha = st.slider("Weight towards content-based (alpha)", 0.0, 1.0, 0.5, 0.1)
    if tfidf_matrix is not None:
        recs = hybrid.recommend_tfidf(
            movies, tfidf_matrix, movie_ids, movie_id_to_row, vectorizer, user_item_matrix,
            liked_movie_ids=st.session_state.liked_movie_ids,
            selected_genres=st.session_state.selected_genres, top_n=10, alpha=alpha,
        )
    else:
        recs = hybrid.recommend(
            movies, genre_matrix, movie_ids, movie_id_to_row, genre_names, user_item_matrix,
            liked_movie_ids=st.session_state.liked_movie_ids,
            selected_genres=st.session_state.selected_genres, top_n=10, alpha=alpha,
        )
    render_recommendations(recs, "hy")

with tab_eval:
    st.caption("Offline evaluation on a held-out test split of real ratings -- this is what goes in your report.")
    if dataset == "ml-25m":
        st.info("ml-25m evaluation is slower (~5s per user) due to the size of the ratings matrix -- keep max users modest.")
    k = st.number_input("K (top-K for precision/recall/F1)", min_value=5, max_value=20, value=10, step=5)
    max_users = st.number_input("Max users to sample", min_value=10, max_value=200, value=30 if dataset == "ml-25m" else 100, step=10)
    if st.button("Run evaluation"):
        with st.spinner("Splitting data and scoring all algorithms..."):
            train_ratings, test_ratings = evaluation.train_test_split_ratings(ratings)
            links = load_links(dataset=dataset) if tmdb_available() else None
            results = evaluation.evaluate_all(movies, train_ratings, test_ratings, k=k, max_users=max_users, links=links)
        st.dataframe(results)
        st.bar_chart(results[[f"precision@{k}", f"recall@{k}", f"f1@{k}"]])
        st.bar_chart(results[["coverage", "diversity"]])

st.caption(TMDB_ATTRIBUTION)

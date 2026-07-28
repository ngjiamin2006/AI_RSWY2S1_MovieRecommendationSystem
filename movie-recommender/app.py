"""Streamlit app tying together onboarding, all three algorithms, and evaluation.

Run with: streamlit run app.py
"""
import streamlit as st

from data_loader import load_data, build_genre_matrix, build_user_item_matrix, get_popular_movies
from algorithms import content_based, collaborative_filtering, hybrid
import evaluation

st.set_page_config(page_title="Movie Recommender", layout="wide")


@st.cache_data
def get_movies_and_ratings():
    return load_data()


@st.cache_data
def get_genre_features(movies):
    return build_genre_matrix(movies)


@st.cache_data
def get_user_item_features(ratings, movie_ids):
    return build_user_item_matrix(ratings, movie_ids)


movies, ratings = get_movies_and_ratings()
movie_ids, genre_matrix, genre_names = get_genre_features(movies)
user_item_matrix, movie_id_to_row, user_id_to_col = get_user_item_features(ratings, movie_ids)

if "liked_movie_ids" not in st.session_state:
    st.session_state.liked_movie_ids = []
if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False

st.title(":clapper: Movie Recommender")

# ---- Onboarding (cold start) ----
if not st.session_state.onboarded:
    st.subheader("Welcome! What do you like to watch?")
    chosen = st.multiselect("Pick a few genres you enjoy", options=genre_names)
    if st.button("Get my recommendations", disabled=not chosen):
        st.session_state.selected_genres = chosen
        st.session_state.onboarded = True
        st.rerun()
    st.stop()


def render_recommendations(df, key_prefix):
    if df is None or df.empty:
        st.info("Not enough signal yet for this algorithm — showing popular movies instead.")
        df = get_popular_movies(movies, ratings, top_n=10)
    for _, row in df.iterrows():
        cols = st.columns([5, 2, 1])
        cols[0].write(f"**{row['title']}**  \n{row['genres']}")
        cols[1].write(f"score: {row['score']:.3f}")
        if cols[2].button(":+1: Like", key=f"{key_prefix}_{row['movieId']}"):
            if row["movieId"] not in st.session_state.liked_movie_ids:
                st.session_state.liked_movie_ids.append(row["movieId"])
            st.rerun()


with st.sidebar:
    st.write("### Your liked movies")
    if st.session_state.liked_movie_ids:
        liked_titles = movies[movies["movieId"].isin(st.session_state.liked_movie_ids)]["title"]
        for t in liked_titles:
            st.write(f"- {t}")
    else:
        st.write("_None yet — click Like on a recommendation._")
    if st.button("Reset onboarding"):
        st.session_state.clear()
        st.rerun()

tab_cb, tab_cf, tab_hy, tab_eval = st.tabs(["Content-Based", "Collaborative Filtering", "Hybrid", "Evaluation"])

with tab_cb:
    st.caption("Recommends movies with similar genres to what you've liked (or picked at onboarding).")
    recs = content_based.recommend(
        movies, genre_matrix, movie_ids, movie_id_to_row, genre_names,
        liked_movie_ids=st.session_state.liked_movie_ids,
        selected_genres=st.session_state.selected_genres, top_n=10,
    )
    render_recommendations(recs, "cb")

with tab_cf:
    st.caption("Recommends movies liked by other users with similar taste to you (needs at least one Like).")
    recs = collaborative_filtering.recommend(
        movies, user_item_matrix, movie_ids, movie_id_to_row,
        liked_movie_ids=st.session_state.liked_movie_ids, top_n=10,
    )
    render_recommendations(recs, "cf")

with tab_hy:
    st.caption("Blends the content-based and collaborative filtering scores.")
    alpha = st.slider("Weight towards content-based (alpha)", 0.0, 1.0, 0.5, 0.1)
    recs = hybrid.recommend(
        movies, genre_matrix, movie_ids, movie_id_to_row, genre_names, user_item_matrix,
        liked_movie_ids=st.session_state.liked_movie_ids,
        selected_genres=st.session_state.selected_genres, top_n=10, alpha=alpha,
    )
    render_recommendations(recs, "hy")

with tab_eval:
    st.caption("Offline evaluation on a held-out test split of real ratings — this is what goes in your report.")
    k = st.number_input("K (top-K for precision/recall/F1)", min_value=5, max_value=20, value=10, step=5)
    if st.button("Run evaluation"):
        with st.spinner("Splitting data and scoring all three algorithms..."):
            train_ratings, test_ratings = evaluation.train_test_split_ratings(ratings)
            results = evaluation.evaluate_all(movies, train_ratings, test_ratings, k=k, max_users=100)
        st.dataframe(results)
        st.bar_chart(results[[f"precision@{k}", f"recall@{k}", f"f1@{k}"]])

"""Streamlit app tying together onboarding, all three algorithms, and evaluation.

Run with: streamlit run app.py
"""
import streamlit as st  # type: ignore[missing-import]

from data_loader import (
    load_data, build_genre_matrix, build_user_item_matrix, get_popular_movies,
    get_most_rated_movies, get_new_releases, load_links, attach_tmdb_metadata,
    build_tfidf_matrix, build_cb_overview_matrix, build_year_lookup, tmdb_available,
)
from algorithms import content_based, collaborative_filtering, hybrid
import evaluation

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

/* Apply Inter font globally */
html, body, [class*="css"]  {   
    font-family: 'Inter', sans-serif !important;
}

/* Adjust top padding to prevent content from hiding under the top header */
.block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
}

/* Gradient Title */
h1 {
    background: -webkit-linear-gradient(45deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    padding-bottom: 10px;
}

/* Style the Tabs to look sleek */
button[data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #E2E8F0 !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* Make Tabs sticky (Always on top) */
div[data-testid="stTabs"] > div:first-of-type,
div[data-baseweb="tab-list"],
div[role="tablist"] {
    position: -webkit-sticky !important;
    position: sticky !important;
    top: 0px !important; /* Set to 0px to ensure it sticks to the very top */
    background-color: rgba(11, 15, 25, 0.98) !important; 
    backdrop-filter: blur(10px) !important;
    z-index: 999999 !important;
    padding-top: 15px !important;
    padding-bottom: 5px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Fix Streamlit parent overflow which breaks position: sticky */
.block-container, [data-testid="stVerticalBlock"], [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stAppViewBlockContainer"] {
    overflow: visible !important;
}

/* Input Fields (Search, numbers) */
input {
    background-color: #1a202c !important;
    border: 1px solid #2d3748 !important;
    color: white !important;
    border-radius: 8px !important;
}
input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f1 !important;
}

/* Metric Boxes */
div[data-testid="metric-container"] {
    background-color: rgba(21, 26, 38, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}

/* Target ONLY the movie columns using the :has selector so checkboxes and metrics aren't distorted */
div[data-testid="column"]:has(.movie-poster-card) {
    background-color: rgba(21, 26, 38, 0.7);
    padding: 15px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}

/* Hover effect ONLY on movie cards */
div[data-testid="column"]:has(.movie-poster-card):hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
}

/* Give posters rounded corners and shadow */
img {
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}

/* Sleek buttons */
div.stButton > button:first-child {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #E2E8F0;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
div.stButton > button:first-child:hover {
    background: rgba(99, 102, 241, 0.8);
    border: 1px solid #6366f1;
    color: white;
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

TMDB_ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."
DATASET = "ml-25m"


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

    _, cb_matrix, _ = build_cb_overview_matrix(enriched)

    poster_lookup = (
        dict(zip(enriched["movieId"], enriched.get("poster_url", []))) if "poster_url" in enriched.columns else {}
    )
    year_lookup = build_year_lookup(enriched)

    return {
        "movies": movies, "ratings": ratings, "enriched": enriched,
        "movie_ids": movie_ids, "genre_matrix": genre_matrix, "genre_names": genre_names,
        "user_item_matrix": user_item_matrix, "movie_id_to_row": movie_id_to_row,
        "user_id_to_col": user_id_to_col, "tfidf_matrix": tfidf_matrix, "vectorizer": vectorizer,
        "cb_matrix": cb_matrix,
        "poster_lookup": poster_lookup, "tmdb_error": tmdb_error, "year_lookup": year_lookup,
    }


pipeline = load_pipeline(DATASET)
movies, ratings = pipeline["movies"], pipeline["ratings"]
enriched = pipeline["enriched"]
movie_ids, genre_matrix, genre_names = pipeline["movie_ids"], pipeline["genre_matrix"], pipeline["genre_names"]
user_item_matrix = pipeline["user_item_matrix"]
movie_id_to_row, user_id_to_col = pipeline["movie_id_to_row"], pipeline["user_id_to_col"]
tfidf_matrix, vectorizer = pipeline["tfidf_matrix"], pipeline["vectorizer"]
cb_matrix = pipeline["cb_matrix"]
poster_lookup = pipeline["poster_lookup"]
year_lookup = pipeline["year_lookup"]

if pipeline["tmdb_error"]:
    st.sidebar.warning(pipeline["tmdb_error"])

if "local_profiles" not in st.session_state:
    st.session_state.local_profiles = {
        "User 1": [],
        "User 2": [],
        "User 3": [],
        "User 4": []
    }
if "current_user" not in st.session_state:
    st.session_state.current_user = "User 1"

known_years = sorted(set(year_lookup.values()))
year_bounds = (known_years[0], known_years[-1]) if known_years else (1900, 2025)

with st.sidebar:
    st.write("### Filter by year")
    col1, col2 = st.columns(2)
    min_yr = col1.number_input("From", min_value=year_bounds[0], max_value=year_bounds[1], value=year_bounds[0], step=1)
    max_yr = col2.number_input("To", min_value=year_bounds[0], max_value=year_bounds[1], value=year_bounds[1], step=1)
    year_range = (int(min_yr), int(max_yr))

# Only treat this as an active filter once the user narrows it -- at the
# full default range, filtering by year_lookup would also silently exclude
# the handful of movies with no known year at all (no TMDb match and no
# parseable year in the title).
year_filter_active = year_range != year_bounds
allowed_ids = (
    {mid for mid, y in year_lookup.items() if year_range[0] <= y <= year_range[1]}
    if year_filter_active else None
)
if year_filter_active:
    st.sidebar.caption(f"Showing only movies released {year_range[0]}-{year_range[1]} ({len(allowed_ids):,} eligible).")

if "liked_movie_ids" not in st.session_state:
    st.session_state.liked_movie_ids = st.session_state.local_profiles[st.session_state.current_user]
if "selected_genres" not in st.session_state:
    st.session_state.selected_genres = []
if "cb_matched_ids" not in st.session_state:
    st.session_state.cb_matched_ids = []

st.title("Movie Recommender")


def add_like(movie_id):
    if movie_id not in st.session_state.liked_movie_ids:
        st.session_state.liked_movie_ids.append(movie_id)
        st.toast("Movie liked! Updating algorithms in the background...", icon="🔄")
        st.session_state.just_liked = True

def render_recommendations(df, key_prefix, show_score=True, score_label="score"):
    if df is None or df.empty:
        if year_filter_active:
            st.info(
                f"No matches in {year_range[0]}-{year_range[1]} for this algorithm and your current taste -- "
                "showing popular movies from that range instead. Try widening the year filter."
            )
        else:
            st.info("Not enough signal yet for this algorithm -- showing popular movies instead.")
        df = get_popular_movies(movies, ratings, top_n=30, allowed_ids=allowed_ids)
        if df.empty:
            st.warning("No movies at all in this year range -- widen it in the sidebar.")
            return
            
    cols_per_row = 5
    
    for i in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        chunk = df.iloc[i:i+cols_per_row]
        
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                poster = poster_lookup.get(row["movieId"])
                title = row['title'] # we need title early for the fallback poster
                
                if isinstance(poster, str) and poster.strip() != "":
                    st.markdown(
                        f'<div class="movie-poster-card" style="height:350px; margin-bottom:12px;">'
                        f'<img src="{poster}" style="width:100%; height:100%; object-fit:cover; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.5);">'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Dynamic text-based fallback poster using a nice gradient and the movie title
                    st.markdown(
                        f'<div class="movie-poster-card" style="height:350px; display:flex; align-items:center; justify-content:center; '
                        f'background: linear-gradient(135deg, #1A202C, #0f131c); border-radius:12px; '
                        f'margin-bottom:12px; box-shadow:0 4px 12px rgba(0,0,0,0.5); padding: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.02);">'
                        f'<span style="color: #94a3b8; font-size: 1.2rem; font-weight: 600; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden;">'
                        f'{title}</span></div>', 
                        unsafe_allow_html=True
                    )
                
                score_str = ""
                if show_score and "score" in row:
                    # Using title() in case score_label is lowercase
                    label = score_label.title()
                    # If it's a count (like number of ratings), format as integer, else as a 2-decimal float
                    if label.lower() == "ratings" or "count" in label.lower():
                        score_str = f"Total Reviews: {int(row['score']):,}"
                    else:
                        score_str = f"{label}: {row['score']:.2f}/5.0"
                elif "release_date" in row:
                    score_str = f"Released: {row['release_date']}"
                
                genres = str(row['genres']).split('|')
                genre_str = ""
                if genres and genres[0] != 'nan':
                    genre_str = f"{', '.join(genres[:2])}"
                
                # Use a fixed-height container to ensure all cards are identical in height
                # margin-top: auto pushes the info block to the bottom, aligning it with the button
                st.markdown(
                    f'<div style="height: 100px; display: flex; flex-direction: column; margin-bottom: 10px;">'
                    f'<strong style="display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis; font-size: 1rem; line-height: 1.2;">'
                    f'{title}</strong>'
                    f'<div style="margin-top: auto;">'
                    f'<div style="color: #a0a0a0; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">'
                    f'{score_str}</div>'
                    f'<div style="color: #a0a0a0; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">'
                    f'{genre_str}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                is_liked = row["movieId"] in st.session_state.liked_movie_ids
                if is_liked:
                    st.button(
                        "Liked",
                        key=f"{key_prefix}_{row['movieId']}",
                        use_container_width=True,
                        disabled=True
                    )
                else:
                    st.button(
                        "Like",
                        key=f"{key_prefix}_{row['movieId']}",
                        use_container_width=True,
                        on_click=add_like,
                        args=(row["movieId"],)
                    )


def refresh_controls(key_prefix):
    """"Refresh" button for an algorithm tab: re-rolls the list instead of
    replaying the same deterministic ranking. Returns pool_size/sample_seed
    kwargs for the recommend() call -- empty on first render (plain
    deterministic top-N, same as evaluation.py), populated after the first
    click (samples top_n from a larger candidate pool using an
    incrementing seed, so each click gives a different-but-still-relevant list).
    """
    state_key = f"{key_prefix}_refresh_count"
    st.session_state.setdefault(state_key, 0)
    if st.button("Refresh", key=f"{key_prefix}_refresh_btn"):
        st.session_state[state_key] += 1
        st.rerun()
    count = st.session_state[state_key]
    return {"pool_size": 30, "sample_seed": count} if count > 0 else {}


def search_and_like(text_key, render_prefix, help_text="Click 'Like' to add them to your profile."):
    """Search-by-title + Like -- the original Home tab search pattern,
    parameterized so each tab's search box/results use independent
    session-state keys (each tab can be searched and tested separately).
    """
    query = st.text_input("Enter movie title...", placeholder="e.g. Inception or Toy Story", key=text_key)
    if query:
        results = movies[movies['title'].str.contains(query, case=False, na=False)].head(20)
        if results.empty:
            st.warning(f"No movies found matching '{query}'.")
        else:
            st.success(f"Found {len(results)} movies. {help_text}")
            render_recommendations(results, render_prefix, show_score=False)
    else:
        st.info("Type a movie name above to search for specific movies to like.")


def search_movies_cb(text_key="cb_search_input", render_prefix="cb_search"):
    """Content-Based tab's search bar: type a keyword and press Enter to see
    every matching title (never silently guesses a single one -- an
    ambiguous keyword like "marvel" would otherwise pick the wrong movie).
    All matches together become the profile for content_based.recommend_by_search(),
    stored as cb_matched_ids -- unlike search_and_like(), nothing is
    appended to liked_movie_ids.
    """
    query = st.text_input(
        "Search movies by title keyword...",
        placeholder="e.g. Inception or Marvel", key=text_key,
    )
    if query:
        matches = movies[movies['title'].str.contains(query, case=False, na=False)].head(20)
        if matches.empty:
            st.warning(f"No movies found matching '{query}'.")
            st.session_state.cb_matched_ids = []
        else:
            st.success(f"Found {len(matches)} movies matching '{query}'.")
            render_recommendations(matches, render_prefix, show_score=False)
            st.session_state.cb_matched_ids = matches['movieId'].tolist()
    else:
        st.session_state.cb_matched_ids = []
        st.info("Type a keyword above and press Enter to get content-based recommendations.")


with st.sidebar:
    st.write("### Select Local User")
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
        
        # Clear search and genre selections
        st.session_state.search_input = ""
        st.session_state.selected_genres = []
        for g in genre_names:
            if f"genre_chk_{g}" in st.session_state:
                st.session_state[f"genre_chk_{g}"] = False
                
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
    
    st.write("### Search Movies")
    search_and_like("search_input", "search")

    st.divider()
    st.write("### What do you like to watch?")
    st.caption("Pick a few genres you enjoy (Cold Start Preferences).")
    
    # Genre Checkbox Grid
    genre_cols = st.columns(4)
    new_selected_genres = []
    
    for i, g in enumerate(genre_names):
        with genre_cols[i % 4]:
            is_checked = st.checkbox(g, value=(g in st.session_state.selected_genres), key=f"genre_chk_{g}")
            if is_checked:
                new_selected_genres.append(g)
                
    st.session_state.selected_genres = new_selected_genres
    
    st.caption(
        "Content-Based now works by searching for a movie directly (see its tab) -- it no longer uses "
        "Likes or these genre picks. Collaborative Filtering and Hybrid still use your Likes/genre picks, "
        "with TMDb-enriched TF-IDF features (overview/keywords/cast/director) when available, falling "
        "back to genre-only similarity otherwise."
    )

with tab_popular:
    st.caption("Most-watched movies (highest number of ratings), regardless of average score.")
    recs = get_most_rated_movies(movies, ratings, top_n=30, allowed_ids=allowed_ids)
    render_recommendations(recs, "pop", score_label="ratings")

with tab_top_rated:
    st.caption("Highest average rating among movies with enough votes to be reliable.")
    render_recommendations(get_popular_movies(movies, ratings, top_n=30, allowed_ids=allowed_ids), "top", score_label="avg rating")

with tab_new:
    st.caption("Most recently released movies with TMDb metadata (requires the TMDb dataset).")
    if tfidf_matrix is not None:
        render_recommendations(get_new_releases(enriched, top_n=30, allowed_ids=allowed_ids), "new", show_score=False)
    else:
        st.info("New Releases needs the TMDb dataset (for release dates) -- see the sidebar warning above.")

def clear_cb_search():
    st.session_state.cb_search_input = ""
    st.session_state.cb_matched_ids = []

with tab_cb:
    st.caption(
        "Search movies by keyword -- every match found is used together (never just one "
        "guessed match) to find similar movies by genre + overview. This tab does not use "
        "the Like button as input."
    )
    if tfidf_matrix is None:
        st.info("TMDb overview data isn't available -- similarity is genre-only for this tab. See the sidebar warning above.")

    search_movies_cb()

    matched_ids = st.session_state.cb_matched_ids
    if matched_ids:
        st.divider()
        cc1, cc2 = st.columns([5, 1])
        with cc1:
            st.write("### You might like these...")
        with cc2:
            st.button("Clear", key="cb_clear_query", on_click=clear_cb_search)

        pool_kwargs = refresh_controls("cb")
        recs = content_based.recommend_by_search(
            movies, cb_matrix, movie_ids, movie_id_to_row,
            matched_movie_ids=matched_ids, top_n=30, allowed_ids=allowed_ids, **pool_kwargs,
        )
        render_recommendations(recs, "cb")

with tab_cf:
    st.caption("Interactive Demo: Recommends movies liked by other Local Users with similar taste to you.")
    search_and_like("cf_search_input", "cf_search")
    pool_kwargs = refresh_controls("cf")

    if not st.session_state.liked_movie_ids:
        # If the user hasn't liked anything, we don't need to try and fail multiple times.
        # Just pass None to trigger the standard popular movies fallback.
        render_recommendations(None, "cf")
    else:
        recs, explanation = collaborative_filtering.recommend_user_based(
            movies, user_item_matrix, movie_ids, movie_id_to_row,
            current_user=st.session_state.current_user,
            local_profiles=st.session_state.local_profiles, top_n=30
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
                liked_movie_ids=st.session_state.liked_movie_ids, top_n=30,
                allowed_ids=allowed_ids, **pool_kwargs,
            )
            render_recommendations(recs_item_based, "cf")

with tab_hy:
    st.caption("Blends the content-based and collaborative filtering scores.")
    search_and_like("hy_search_input", "hy_search")
    alpha = st.slider("Weight towards content-based (alpha)", 0.0, 1.0, 0.5, 0.1)
    pool_kwargs = refresh_controls("hy")
    if tfidf_matrix is not None:
        recs = hybrid.recommend_tfidf(
            movies, tfidf_matrix, movie_ids, movie_id_to_row, vectorizer, user_item_matrix,
            liked_movie_ids=st.session_state.liked_movie_ids,
            selected_genres=st.session_state.selected_genres, top_n=30, alpha=alpha,
            allowed_ids=allowed_ids, **pool_kwargs,
        )
    else:
        recs = hybrid.recommend(
            movies, genre_matrix, movie_ids, movie_id_to_row, genre_names, user_item_matrix,
            liked_movie_ids=st.session_state.liked_movie_ids,
            selected_genres=st.session_state.selected_genres, top_n=30, alpha=alpha,
            allowed_ids=allowed_ids, **pool_kwargs,
        )
    render_recommendations(recs, "hy")

with tab_eval:
    st.caption("Offline evaluation on a held-out test split of real ratings -- this is what goes in your report.")
    st.info("Evaluation on ml-25m is slower (~5s per user) due to the size of the ratings matrix -- keep max users modest.")
    k = st.number_input("K (top-K for precision/recall/F1)", min_value=5, max_value=20, value=10, step=5)
    max_users = st.number_input("Max users to sample", min_value=10, max_value=200, value=30, step=10)
    if st.button("Run evaluation"):
        with st.spinner("Splitting data and scoring all algorithms..."):
            train_ratings, test_ratings = evaluation.train_test_split_ratings(ratings)
            links = load_links(dataset=DATASET) if tmdb_available() else None
            results = evaluation.evaluate_all(movies, train_ratings, test_ratings, k=k, max_users=max_users, links=links)
        st.dataframe(results)
        st.bar_chart(results[[f"precision@{k}", f"recall@{k}", f"f1@{k}"]])
        st.bar_chart(results[["coverage", "diversity"]])

st.caption(TMDB_ATTRIBUTION)

with st.sidebar:
    with st.expander("👥 View All Users' Profiles", expanded=False):
        html_content = "<div style='max-height: 300px; overflow-y: auto; padding-right: 10px;'>"
        for p_user, p_likes in st.session_state.local_profiles.items():
            html_content += f"<strong style='color: white;'>{p_user}</strong><br>"
            if p_likes:
                liked_movies = movies[movies["movieId"].isin(p_likes)][["title", "genres"]]
                html_content += "<ul style='margin-top: 5px; padding-left: 20px; font-size: 0.85em;'>"
                for _, row in liked_movies.iterrows():
                    title = row['title']
                    genres = str(row['genres']).replace('|', ', ') if 'genres' in row and str(row['genres']) != 'nan' else ''
                    if genres:
                        html_content += f"<li style='margin-bottom: 5px;'><span style='color: #E2E8F0;'>{title}</span><br><span style='color: #94a3b8; font-size: 0.9em;'>{genres}</span></li>"
                    else:
                        html_content += f"<li style='margin-bottom: 5px;'><span style='color: #E2E8F0;'>{title}</span></li>"
                html_content += "</ul>"
            else:
                html_content += "<p style='color: #a0a0a0; font-size: 0.85em; margin-bottom: 15px;'>No movies liked yet.</p>"
        html_content += "</div>"
        
        st.markdown(html_content, unsafe_allow_html=True)

    st.divider()
    st.write("### Your liked movies")
    if st.session_state.liked_movie_ids:
        liked_data = movies[movies["movieId"].isin(st.session_state.liked_movie_ids)]
        for _, row in liked_data.iterrows():
            c1, c2 = st.columns([5, 1])
            title = row["title"]
            genres = str(row["genres"]).replace("|", ", ") if "genres" in row and str(row["genres"]) != "nan" else ""
            
            with c1:
                if genres:
                    st.markdown(f"**{title}**<br><span style='color: #a0a0a0; font-size: 0.85em;'>{genres}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{title}**", unsafe_allow_html=True)
            with c2:
                if st.button("❌", key=f"remove_{row['movieId']}", help="Remove movie"):
                    st.session_state.liked_movie_ids.remove(row["movieId"])
                    st.rerun()
    else:
        st.write("_None yet -- click Like on a recommendation._")
    if st.button("Reset all users"):
        st.session_state.clear()
        st.rerun()

if st.session_state.get("just_liked", False):
    st.toast("Update complete! Your recommendations are ready.", icon="✅")
    st.session_state.just_liked = False

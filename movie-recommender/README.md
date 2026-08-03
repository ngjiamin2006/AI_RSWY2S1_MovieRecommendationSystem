# Movie Recommender — AI Assignment (Recommender System)

Streamlit prototype for the Topic 3 group assignment. Datasets: [MovieLens ml-25m](https://grouplens.org/datasets/movielens/) (62,423 movies, 162,541 users, 25,000,095 ratings) and TMDb metadata (overview/keywords/cast/director/posters). Both are too large for git — see **Data Setup** below.

## Data Setup

The app needs two things in place before it will run, neither of which is in this repo (both are gitignored — too big for GitHub, which hard-rejects any file over 100MB):

```
movie-recommender/
└── data/
    ├── tmdb/
    │   └── movies_DB.csv
    └── ml-25m/
        ├── movies.csv
        ├── ratings.csv
        └── links.csv
```

**Easiest option: ask a team member for `movie-recommender-data.zip`** and extract it directly into your local `movie-recommender/` folder (same level as `app.py`) — it unpacks into exactly the structure above.

**From scratch:**
- MovieLens ml-25m: download [files.grouplens.org/datasets/movielens/ml-25m.zip](https://files.grouplens.org/datasets/movielens/ml-25m.zip) and unzip into `data/ml-25m/`.
- TMDb `movies_DB.csv`: source not yet documented here — get it from a team member (via the data zip above) until this is filled in.

Without TMDb data the app still runs (falls back to genre-only content-based, no posters); without MovieLens ml-25m it won't start at all.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## How the app is organized

- `data_loader.py` — loads MovieLens `ml-25m` + TMDb metadata, builds the genre matrix, TF-IDF matrix, and sparse user-item ratings matrix. Shared by everyone, no Streamlit code here.
- `algorithms/content_based.py` — **Member 1**: genre one-hot baseline, plus a richer TF-IDF variant over TMDb overview/keywords/cast/director. Handles cold start via picked genres.
- `algorithms/collaborative_filtering.py` — **Member 2**: item-based CF using rating similarity between movies, plus a local-user Jaccard-similarity variant (simulated multi-user demo). Item-based CF has no cold-start ability by design — falls back to a popularity list until the user has liked at least one movie (this is expected and worth mentioning in your report as a real limitation of CF).
- `algorithms/hybrid.py` — **Member 3**: weighted combination of the two above, in both genre-only and TF-IDF flavors.
- `algorithms/ranking.py` — shared top-N selection logic (year-range filtering, "Refresh" re-sampling) used by all three algorithm files.
- `evaluation.py` — train/test split + precision/recall/F1@K (all three algorithms) + RMSE/MAE (collaborative filtering) + coverage/diversity/timing. Run directly (`python evaluation.py`) to get a metrics table for your documentation, independent of the UI.
- `app.py` — the Streamlit UI: pick genres and/or like movies, then browse the Home, Popular, Top Rated, New Releases, Content-Based, Collaborative Filtering, and Hybrid tabs (posters, year filter, refresh, local-user switcher) and check the Evaluation tab for metrics.

## Suggested workflow per member

1. Run the app once as-is (`streamlit run app.py`) so everyone sees the baseline working end to end.
2. Each member improves **only their own file** in `algorithms/` — the function signature (`recommend(...)`) must stay the same so `app.py` and `evaluation.py` keep working.
3. Improvement ideas are noted as comments at the top of each algorithm file.
4. Re-run `python evaluation.py` after changes to see how your metrics moved.

## What still needs to be added for submission

- [ ] Try each member's improvement ideas and re-evaluate.
- [ ] Screenshot the Evaluation tab's table/chart for the Results & Discussion section of the documentation.
- [ ] Note in the documentation that this uses MovieLens data (not self-collected) and cite it properly (GroupLens Research, F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.), plus the required TMDb attribution ("This product uses the TMDB API but is not endorsed or certified by TMDB.").
- [ ] Fill in the AI Disclosure Statement (Appendix B) using `AI_DISCLOSURE_LOG.md` — keep that log updated every time the code changes with further AI assistance.

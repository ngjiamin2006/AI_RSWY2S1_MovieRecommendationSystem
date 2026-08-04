# AI_RSWY2S1_movieRecommendationSystem

Movie recommendation system built for the AI Group Assignment (Session 202605, Topic 3: Recommender System). It recommends movies using three approaches — content-based filtering (genre + TMDb-enriched TF-IDF), collaborative filtering (item-based + a local-user similarity demo), and a hybrid of both — served through a Streamlit web app with posters, year filtering, and refreshable results, and evaluated with standard recommender metrics (precision/recall/F1, RMSE, MAE, coverage, diversity).

Datasets: MovieLens **ml-25m** (62,423 movies, 162,541 users, 25,000,095 ratings) and TMDb metadata (overview/keywords/cast/director/posters). Both are too large for git and are **not included in this repo** — see `movie-recommender/README.md`'s **Data Setup** section for how to get them.

## Project structure

```
AI_RSWY2S1/
├── ASSIGNMENT.md              # assignment brief
└── movie-recommender/         # the app
    ├── app.py                 # Streamlit UI (entry point)
    ├── data_loader.py         # loads data, builds feature/ratings matrices
    ├── evaluation.py          # precision/recall/F1@K + RMSE, run standalone
    ├── algorithms/
    │   ├── content_based.py            # genre + TMDb TF-IDF recommenders
    │   ├── collaborative_filtering.py  # item-based + local-user CF
    │   ├── hybrid.py                   # weighted combination of the two
    │   └── ranking.py                  # shared top-N selection/refresh logic
    ├── data/                  # MovieLens ml-25m + TMDb CSVs (gitignored, download separately)
    └── requirements.txt
```

## How to run

Requires Python 3.10+, and the datasets downloaded per `movie-recommender/README.md`'s Data Setup section.

```bash
cd movie-recommender

# create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# install dependencies
pip install -r requirements.txt

# launch the app
streamlit run app.py
```

This opens the app in your browser (default `http://localhost:8501`). Pick a few favorite genres and/or like some recommendations, then browse each algorithm's tab (posters, year filter, refresh button) and check the **Evaluation** tab for metrics.

To run the offline evaluation (precision/recall/F1/RMSE) without the UI:

```bash
python evaluation.py
```

## Algorithms

- **Content-based** (`algorithms/content_based.py`) — genre one-hot + cosine similarity baseline, plus a richer TF-IDF variant over TMDb overview/keywords/cast/director (overview weighted separately so it doesn't drown out the tag signal); handles cold start via genre picks.
- **Collaborative filtering** (`algorithms/collaborative_filtering.py`) — item-based CF using rating similarity between movies, plus a local-user Jaccard-similarity demo; falls back to a popularity list until the user has liked at least one movie.
- **Hybrid** (`algorithms/hybrid.py`) — weighted combination of the content-based and collaborative filtering scores, in both genre-only and TF-IDF flavors.

## Documentation

- `movie-recommender/README.md` — data setup, implementation notes, and per-member workflow.
- `movie-recommender/SPEC_COMPLIANCE.md` — mapping of the assignment requirements to what's implemented.
- `movie-recommender/AI_DISCLOSURE_LOG.md` — log of AI tool usage, for the AI Disclosure Statement appendix.
- `ASSIGNMENT.md` — the assignment brief this project follows.

## Citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

This product uses the TMDB API but is not endorsed or certified by TMDB.

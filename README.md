# AI_RSWY2S1_movieRecommendationSystem

Movie recommendation system built for the AI Group Assignment (Session 202605, Topic 3: Recommender System). It recommends movies using three approaches — content-based filtering, collaborative filtering, and a hybrid of both — served through a Streamlit web app, and evaluated with standard recommender metrics (precision/recall/F1, RMSE).

Dataset: [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) (9,742 movies, 610 users, 100,836 ratings), included under `movie-recommender/data/`.

## Project structure

```
AI_RSWY2S1/
├── ASSIGNMENT.md              # assignment brief
└── movie-recommender/         # the app
    ├── app.py                 # Streamlit UI (entry point)
    ├── data_loader.py         # loads data, builds feature/ratings matrices
    ├── evaluation.py          # precision/recall/F1@K + RMSE, run standalone
    ├── algorithms/
    │   ├── content_based.py         # genre-similarity recommender
    │   ├── collaborative_filtering.py  # item-based collaborative filtering
    │   └── hybrid.py                # weighted combination of the two
    ├── data/                  # MovieLens ml-latest-small CSVs
    └── requirements.txt
```

## How to run

Requires Python 3.10+.

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

This opens the app in your browser (default `http://localhost:8501`). On first use you'll be asked to pick a few favorite genres (onboarding), then you can browse recommendations under each algorithm's tab, like movies, and check the **Evaluation** tab for metrics.

To run the offline evaluation (precision/recall/F1/RMSE) without the UI:

```bash
python evaluation.py
```

## Algorithms

- **Content-based** (`algorithms/content_based.py`) — recommends movies similar in genre to ones the user liked; also handles cold start via the onboarding genre picks.
- **Collaborative filtering** (`algorithms/collaborative_filtering.py`) — item-based CF using rating similarity between movies; falls back to a popularity list until the user has liked at least one movie.
- **Hybrid** (`algorithms/hybrid.py`) — weighted combination of the content-based and collaborative filtering scores.

## Documentation

- `movie-recommender/README.md` — implementation notes and per-member workflow.
- `movie-recommender/SPEC_COMPLIANCE.md` — mapping of the assignment requirements to what's implemented.
- `movie-recommender/AI_DISCLOSURE_LOG.md` — log of AI tool usage, for the AI Disclosure Statement appendix.
- `ASSIGNMENT.md` — the assignment brief this project follows.

## Citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

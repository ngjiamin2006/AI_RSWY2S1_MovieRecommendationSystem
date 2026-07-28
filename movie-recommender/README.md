# Movie Recommender — AI Assignment (Recommender System)

Streamlit prototype for the Topic 3 group assignment. Dataset: [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) (9,742 movies, 610 users, 100,836 ratings) — already downloaded into `data/`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## How the app is organized

- `data_loader.py` — loads `movies.csv` / `ratings.csv`, builds the genre feature matrix and the sparse user-item ratings matrix. Shared by everyone, no Streamlit code here.
- `algorithms/content_based.py` — **Member 1**: genre-similarity recommender. Also handles cold start (new user with no likes yet) using their onboarding genre picks.
- `algorithms/collaborative_filtering.py` — **Member 2**: item-based collaborative filtering using rating similarity between movies. Has no cold-start ability by design — falls back to popularity list until the user has liked at least one movie (this is expected and worth mentioning in your report as a real limitation of CF).
- `algorithms/hybrid.py` — **Member 3**: weighted combination of the two above.
- `evaluation.py` — train/test split + precision/recall/F1@K (all three algorithms) + RMSE (collaborative filtering). Run directly (`python evaluation.py`) to get a metrics table for your documentation, independent of the UI.
- `app.py` — the Streamlit UI: onboarding (genre picker) → tabs per algorithm with a Like button → Evaluation tab.

## Suggested workflow per member

1. Run the app once as-is (`streamlit run app.py`) so everyone sees the baseline working end to end.
2. Each member improves **only their own file** in `algorithms/` — the function signature (`recommend(...)`) must stay the same so `app.py` and `evaluation.py` keep working.
3. Improvement ideas are noted as comments at the top of each algorithm file.
4. Re-run `python evaluation.py` after changes to see how your metrics moved.

## What still needs to be added for submission

- [ ] Try each member's improvement ideas and re-evaluate.
- [ ] Screenshot the Evaluation tab's table/chart for the Results & Discussion section of the documentation.
- [ ] Note in the documentation that this is MovieLens data (not self-collected) and cite it properly (GroupLens Research, F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.).
- [ ] Fill in the AI Disclosure Statement (Appendix B) using `AI_DISCLOSURE_LOG.md` — keep that log updated every time the code changes with further AI assistance.

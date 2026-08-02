"""Shared top-N selection logic used by every algorithm's recommend() function.

Centralized here so year-range filtering and "Refresh" re-sampling behave
identically across content_based.py, collaborative_filtering.py, and
hybrid.py instead of being re-implemented five times.
"""
import numpy as np


def select_top_n(scores: np.ndarray, movie_ids: np.ndarray, exclude: set,
                  allowed_ids: set | None = None, top_n: int = 10,
                  pool_size: int | None = None, sample_seed: int | None = None) -> list:
    """Rank movie_ids by scores (descending) and return up to top_n movieIds.

    - `exclude`: movieIds never returned (e.g. already liked).
    - `allowed_ids`: if given, only these movieIds are eligible (e.g. a
      year-range filter). Unlike `exclude`, this can shrink the candidate
      pool a lot -- callers should handle a short or empty result explicitly
      rather than assume top_n items always come back.
    - `pool_size` + `sample_seed`: if both given, the top `pool_size` eligible
      candidates are gathered first, then `top_n` of them are randomly
      sampled using `sample_seed`. A "Refresh" button re-rolls the list by
      passing a new seed each click. Without these, selection is the plain
      deterministic top-N (unchanged behaviour for existing callers, e.g.
      evaluation.py, which never passes them).
    """
    fetch_n = pool_size if pool_size is not None else top_n
    order = np.argsort(-scores)
    results = []
    for idx in order:
        mid = movie_ids[idx]
        if mid in exclude:
            continue
        if allowed_ids is not None and mid not in allowed_ids:
            continue
        results.append(mid)
        if len(results) >= fetch_n:
            break

    if pool_size is not None and sample_seed is not None and len(results) > top_n:
        rng = np.random.default_rng(sample_seed)
        results = list(rng.choice(np.array(results), size=top_n, replace=False))

    return results[:top_n]

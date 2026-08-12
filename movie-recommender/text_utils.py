"""Explicit text-preprocessing helpers for content-based similarity.

Kept separate from data_loader.py so the preprocessing steps required for
the Content-Based tab (lowercase, strip numbers, strip stopwords) are
visible and independently testable, rather than relying on
TfidfVectorizer's implicit stop_words="english" default (which is the only
place any of this happened before -- see data_loader.build_tfidf_matrix's
overview_vectorizer).
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text) -> str:
    """Lowercase -> strip digits/punctuation -> remove English stopwords.

    Returns "" for non-string input (NaN overviews are floats in pandas).
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()                       # clear capital letters
    text = _NON_ALPHA_RE.sub(" ", text)        # clear numbers + punctuation
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]  # clear stop words
    return " ".join(tokens)

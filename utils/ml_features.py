"""ml_features.py

Converts a raw customer ticket string into the exact feature DataFrame
that the trained Random Forest pipeline expects.

The feature set mirrors what was used in the training notebook:

    Numeric:     text_length, word_count, exclamation_count,
                 question_count, caps_ratio, sentiment_score, has_number
    Categorical: hour_bucket, day_name

For live inference the time-based features use the current timestamp
(the ticket arrives now, not at training time).
"""

import re
import pandas as pd
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# VADER analyser — initialised once at module import.
_sia = SentimentIntensityAnalyzer()


def _caps_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are upper-case."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _hour_bucket(hour: int) -> str:
    """Map an hour (0-23) to the same buckets used during training."""
    if hour <= 5:
        return "night"
    if hour <= 11:
        return "morning"
    if hour <= 17:
        return "afternoon"
    return "evening"


def extract_features(text: str, timestamp: datetime | None = None) -> pd.DataFrame:
    """Return a single-row DataFrame of ML features for a raw ticket string.

    Args:
        text:      Raw customer ticket text.
        timestamp: The time the ticket arrived.  Defaults to now.

    Returns:
        pd.DataFrame with exactly one row and all required feature columns.
    """
    ts = timestamp or datetime.now()
    sentiment = _sia.polarity_scores(text)["compound"]

    row = {
        # Numeric
        "text_length":       len(text),
        "word_count":        len(text.split()),
        "exclamation_count": text.count("!"),
        "question_count":    text.count("?"),
        "caps_ratio":        _caps_ratio(text),
        "sentiment_score":   sentiment,
        "has_number":        int(bool(re.search(r"\d", text))),
        # Categorical — must be string dtype to match training
        "hour_bucket":       _hour_bucket(ts.hour),
        "day_name":          ts.strftime("%a"),   # Mon, Tue, …
    }

    return pd.DataFrame([row])

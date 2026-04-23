"""train_model.py

Trains the Random Forest priority classifier locally and saves it to
artifacts/priority_model.joblib in the exact format main.py expects.

Usage:
    source .venv/bin/activate
    python scripts/train_model.py
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load labeled data
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "labeled_tickets_cache.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "priority_model.joblib"

logger.info("Loading labeled data from %s", DATA_PATH)
df = pd.read_csv(DATA_PATH)
logger.info("Loaded %d rows", len(df))

# Drop rows with missing text or label
df = df.dropna(subset=["text", "priority_label"])
df["priority_label"] = df["priority_label"].astype(int)
logger.info("After dropping nulls: %d rows", len(df))

# ---------------------------------------------------------------------------
# Build feature matrix vectorised (fast — avoids row-by-row Python loop)
# ---------------------------------------------------------------------------
logger.info("Extracting features (vectorised)...")

import re as _re
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
_sia = SentimentIntensityAnalyzer()

texts = df["text"].astype(str)

X = pd.DataFrame({
    "text_length":       texts.str.len(),
    "word_count":        texts.str.split().str.len(),
    "exclamation_count": texts.str.count("!"),
    "question_count":    texts.str.count(r"\?"),
    "caps_ratio":        texts.apply(lambda t: sum(1 for c in t if c.isupper()) / max(sum(1 for c in t if c.isalpha()), 1)),
    "sentiment_score":   texts.apply(lambda t: _sia.polarity_scores(t)["compound"]),
    "has_number":        texts.str.contains(r"\d", regex=True).astype(int),
    "hour_bucket":       "afternoon",   # training-time constant (no timestamps in corpus)
    "day_name":          "Mon",         # training-time constant
})

y = df["priority_label"].values

logger.info("Feature matrix shape: %s", X.shape)
logger.info("Label distribution: %s", pd.Series(y).value_counts().to_dict())

# ---------------------------------------------------------------------------
# Define feature groups (must match extract_features output)
# ---------------------------------------------------------------------------
numeric_features = [
    "text_length", "word_count", "exclamation_count",
    "question_count", "caps_ratio", "sentiment_score", "has_number",
]
categorical_features = ["hour_bucket", "day_name"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
logger.info("Training Random Forest...")
pipeline.fit(X, y)
logger.info("Training complete.")

# ---------------------------------------------------------------------------
# Save in the exact format main.py expects
# ---------------------------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
artifact = {
    "pipeline": pipeline,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
}
joblib.dump(artifact, OUTPUT_PATH)
logger.info("Model saved to %s", OUTPUT_PATH)

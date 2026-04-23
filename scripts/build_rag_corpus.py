"""build_rag_corpus.py

Builds the RAG question-answer corpus from the raw Twitter Customer Support
dataset (twcs.csv) and saves it to data/processed/rag_corpus.csv.

Each row in the output represents one resolved support case:
  - question : cleaned customer tweet (thread starter)
  - answer   : all company replies concatenated in chronological order
  - priority_label     : 0 = normal, 1 = urgent
  - priority_label_str : "normal" / "urgent"

Labeling uses the exact same weak-supervision scoring function that was
used to label the ML training data in the notebook, so the corpus labels
are consistent with the rest of the project.

Usage:
    python scripts/build_rag_corpus.py \\
        --input  data/processed/twcs.csv \\
        --output data/processed/rag_corpus.csv
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Ensure the project root is on the path so we can import shared modules.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.text import clean_text  # shared cleaner used by corpus + backend

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Labeling constants — must match the notebook exactly
# ---------------------------------------------------------------------------
STRONG_PHRASES = [
    "fraud", "fraudulent",
    "hacked", "hack",
    "locked out", "account locked",
    "outage", "service down",
    "stolen", "unauthorized",
    "charged twice", "double charged",
    "payment failed", "refund not received",
    "cannot login", "cant login", "can't login",
]

MEDIUM_PHRASES = [
    "refund", "cancel", "cancelled", "canceled",
    "error", "not working", "broken", "failed",
]

WEAK_PHRASES = ["asap", "urgent", "immediately", "right now"]

STRONG_PATTERN = re.compile("|".join(re.escape(p) for p in STRONG_PHRASES), re.IGNORECASE)
MEDIUM_PATTERN = re.compile("|".join(re.escape(p) for p in MEDIUM_PHRASES), re.IGNORECASE)
WEAK_PATTERN   = re.compile("|".join(re.escape(p) for p in WEAK_PHRASES),   re.IGNORECASE)

# Deflection replies to filter out — short company replies that provide no
# resolution context and would poison retrieval results.
DEFLECTION_PHRASES = [
    "please dm", "send us a dm", "direct message",
    "please follow", "we need to follow",
    "can you dm", "could you dm",
]
DEFLECTION_PATTERN = re.compile(
    "|".join(re.escape(p) for p in DEFLECTION_PHRASES), re.IGNORECASE
)

URGENT_THRESHOLD = 2   # weak score >= 2 → urgent (same as notebook)
MIN_WORD_COUNT   = 4   # drop tweets with fewer than 4 words (same as notebook)
DEFLECTION_MAX_WORDS = 15  # only drop answers shorter than this


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def label_ticket(clean: str, raw: str, sia: SentimentIntensityAnalyzer) -> dict:
    """Compute weak-supervision scores and return the priority label.

    Args:
        clean: Cleaned tweet text.
        raw:   Original tweet text (used for punctuation signals).
        sia:   Initialised VADER analyser.

    Returns:
        Dict with keys: strong_hit, medium_hit, weak_hit, exclamation_hit,
        sentiment_score, sentiment_hit, weak_score, priority_label,
        priority_label_str.
    """
    strong_hit     = int(bool(STRONG_PATTERN.search(clean))) * 2
    medium_hit     = int(bool(MEDIUM_PATTERN.search(clean)))
    weak_hit       = int(bool(WEAK_PATTERN.search(clean)))
    exclamation_hit = int(raw.count("!") >= 2)
    sentiment_score = sia.polarity_scores(clean)["compound"]
    sentiment_hit   = int(sentiment_score <= -0.5)

    weak_score = (
        strong_hit + medium_hit + weak_hit + exclamation_hit + sentiment_hit
    )
    priority_label = int(weak_score >= URGENT_THRESHOLD)

    return {
        "strong_hit": strong_hit,
        "medium_hit": medium_hit,
        "weak_hit": weak_hit,
        "exclamation_hit": exclamation_hit,
        "sentiment_score": sentiment_score,
        "sentiment_hit": sentiment_hit,
        "weak_score": weak_score,
        "priority_label": priority_label,
        "priority_label_str": "urgent" if priority_label == 1 else "normal",
    }


def build_corpus(input_csv: Path, output_csv: Path, test_size: float = 0.1) -> None:
    """Run the full corpus construction pipeline and write the output CSVs.

    Produces two files:
      - <output_csv stem>_train.csv  — indexed into Chroma
      - <output_csv stem>_test.csv   — held out for RAG evaluation

    Args:
        input_csv:  Path to the raw twcs.csv file.
        output_csv: Base output path. Train/test suffixes are added automatically.
        test_size:  Fraction of the corpus to hold out as the RAG test set.
    """
    # The labeled tickets cache lives next to the raw CSV.
    # If it already exists we skip the slow VADER labeling step (~8 min).
    label_cache = input_csv.parent / "labeled_tickets_cache.csv"

    if label_cache.exists():
        logger.info("Loading labeled tickets from cache %s", label_cache)
        tickets = pd.read_csv(label_cache)
        logger.info("Cached labeled tickets: %d", len(tickets))
    else:
        # --- Load ---
        logger.info("Loading raw dataset from %s", input_csv)
        df = pd.read_csv(input_csv)
        logger.info("Raw rows: %d", len(df))

        # --- Isolate thread-starter customer tweets ---
        tickets = df[
            (df["inbound"] == True) &
            (df["in_response_to_tweet_id"].isna())
        ].copy()
        logger.info("Thread-starter customer tweets: %d", len(tickets))

        # --- Word count filter ---
        tickets["word_count"] = tickets["text"].str.split().str.len()
        tickets = tickets[tickets["word_count"] >= MIN_WORD_COUNT].copy()

        # --- Clean text ---
        tickets["clean_text"] = tickets["text"].apply(clean_text)
        tickets = tickets[tickets["clean_text"].str.len() > 0].copy()
        logger.info("After cleaning + short-text filter: %d", len(tickets))

        # --- Label ---
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()

        label_rows = [
            label_ticket(row["clean_text"], row["text"], sia)
            for _, row in tickets.iterrows()
        ]
        label_df = pd.DataFrame(label_rows, index=tickets.index)
        tickets = pd.concat([tickets, label_df], axis=1)

        logger.info(
            "Label distribution — normal: %d  urgent: %d",
            (tickets["priority_label"] == 0).sum(),
            (tickets["priority_label"] == 1).sum(),
        )

        # Save cache so the next run skips the labeling step entirely.
        tickets.to_csv(label_cache, index=False)
        logger.info("Labeled tickets cached to %s", label_cache)

    # Re-load the raw file just for the company replies (needed for the join).
    # This is fast (~20 s) compared to the labeling step.
    logger.info("Loading raw dataset for company replies from %s", input_csv)
    df = pd.read_csv(input_csv)

    # --- Join company replies ---
    company = df[df["inbound"] == False][
        ["tweet_id", "in_response_to_tweet_id", "text"]
    ].copy()

    # Convert IDs to clean integer strings so both sides match.
    # Reading CSVs with NaN in numeric columns produces floats (e.g. 123456.0),
    # so a plain .astype(str) would give "123456.0" vs "123456" and the join
    # would return 0 rows. Converting via Int64 first strips the decimal part.
    def to_id_str(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)

    tickets["tweet_id"] = to_id_str(tickets["tweet_id"])
    company["in_response_to_tweet_id"] = to_id_str(company["in_response_to_tweet_id"])
    company["tweet_id"] = to_id_str(company["tweet_id"])

    # Sort by tweet_id (chronological proxy) before grouping so concatenated
    # replies are in the correct order.
    company = company.sort_values("tweet_id")

    reply_map = (
        company
        .groupby("in_response_to_tweet_id")["text"]
        .apply(lambda texts: " | ".join(str(t).strip() for t in texts))
        .reset_index()
        .rename(columns={
            "in_response_to_tweet_id": "tweet_id",
            "text": "answer_raw",
        })
    )

    corpus = tickets.merge(reply_map, on="tweet_id", how="inner")
    logger.info("QA pairs after joining replies: %d", len(corpus))

    # --- Quality filter: drop pure deflection replies ---
    is_deflection_only = (
        corpus["answer_raw"].str.contains(DEFLECTION_PATTERN, regex=True, na=False) &
        (corpus["answer_raw"].str.split().str.len() < DEFLECTION_MAX_WORDS)
    )
    corpus = corpus[~is_deflection_only].copy()
    logger.info("QA pairs after deflection filter: %d", len(corpus))

    # --- Select and rename export columns ---
    export = corpus[[
        "tweet_id",
        "clean_text",
        "answer_raw",
        "priority_label",
        "priority_label_str",
        "weak_score",
        "sentiment_score",
    ]].rename(columns={
        "clean_text": "question",
        "answer_raw": "answer",
    })

    # --- Stratified train / test split ---
    # Stratify by priority_label so both splits preserve the urgent/normal ratio.
    train_df, test_df = train_test_split(
        export,
        test_size=test_size,
        random_state=42,
        stratify=export["priority_label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # --- Write output ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    train_path = output_csv.parent / (output_csv.stem + "_train" + output_csv.suffix)
    test_path  = output_csv.parent / (output_csv.stem + "_test"  + output_csv.suffix)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    logger.info(
        "Train corpus saved to %s  (%d rows)", train_path, len(train_df)
    )
    logger.info(
        "Test corpus saved to  %s  (%d rows)", test_path, len(test_df)
    )
    logger.info(
        "Train label split — normal: %d  urgent: %d",
        (train_df["priority_label"] == 0).sum(),
        (train_df["priority_label"] == 1).sum(),
    )
    logger.info(
        "Test label split  — normal: %d  urgent: %d",
        (test_df["priority_label"] == 0).sum(),
        (test_df["priority_label"] == 1).sum(),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the RAG QA corpus from twcs.csv."
    )
    parser.add_argument(
        "--input",
        default="data/processed/twcs.csv",
        help="Path to the raw twcs.csv file (default: data/processed/twcs.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/rag_corpus.csv",
        help="Base output path — _train and _test suffixes are added automatically",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of the corpus to hold out as the RAG test set (default: 0.1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_corpus(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        test_size=args.test_size,
    )

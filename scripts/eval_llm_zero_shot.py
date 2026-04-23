"""eval_llm_zero_shot.py

Zero-shot LLM priority classification evaluation script.

Loads a pre-exported test sample produced by the training notebook,
calls the OpenAI API for each ticket using a Pydantic structured-output
schema, measures wall-clock latency and token cost per call, then
compares the results against the ML baseline predictions already stored
in the CSV file.

Why structured outputs?
  Parsing free-form LLM text with regex fails in many small ways
  (extra whitespace, the model prefixes its answer, field types vary).
  Passing a Pydantic model directly to the OpenAI client forces the
  model to return a validated JSON object — or fail loudly.

Usage:
    python scripts/eval_llm_zero_shot.py \\
        --input  data/processed/test_export.csv \\
        --n-samples 300 \\
        --output logs/llm_eval_results.csv

Environment variables (set in .env):
    OPENAI_API_KEY — required
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

# Add the project root to Python's import path when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import pandas to load and work with the exported test dataset.
import pandas as pd

# Load environment variables such as the OpenAI API key from the .env file.
from dotenv import load_dotenv

# OpenAI client used to send each ticket to the LLM.
from openai import OpenAI

# Import the metrics we need to compare ML and LLM performance.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Import the structured schema that the LLM response must follow.
from schemas.priority import TicketPriority

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Configure logging so the script prints clear progress messages while it runs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a logger object for this file.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# gpt-4o-mini pricing (USD per token, as of 2024)
# Input:  $0.150 per 1,000,000 tokens
# Output: $0.600 per 1,000,000 tokens
# Convert the published input price into cost per single token.
GPT4O_MINI_INPUT_COST_PER_TOKEN = 0.150 / 1_000_000

# Convert the published output price into cost per single token.
GPT4O_MINI_OUTPUT_COST_PER_TOKEN = 0.600 / 1_000_000

# Store the default model name in one place so it is easy to change later.
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Save partial results every N rows so long runs are not lost if interrupted.
CHECKPOINT_EVERY = 25

# System prompt: role, output format, and invariant rules.
# Per-request data (the ticket text) goes in the user prompt only.
# This is the main instruction the LLM sees before classifying a ticket.
SYSTEM_PROMPT = (
    "You are a customer support triage assistant. "
    "Your job is to classify support tickets as either 'urgent' or 'normal'.\n\n"
    "Urgent tickets involve: service outages, billing emergencies, data loss, "
    "safety concerns, or extreme customer distress.\n"
    "Normal tickets involve: general questions, minor issues, feature requests, "
    "or routine account inquiries.\n\n"
    "Respond with your classification label, a confidence score between 0.0 and 1.0, "
    "and a brief one-sentence reasoning."
)


# ---------------------------------------------------------------------------
# Environment / setup
# ---------------------------------------------------------------------------
def load_env() -> None:
    """
    Load environment variables from the .env file in the project root.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is absent after loading.
    """
    # Read variables from the local .env file into the process environment.
    load_dotenv()

    # Stop early if the API key is missing, because the script cannot run without it.
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found. "
            "Copy .env.example to .env and fill in your key."
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_sample(csv_path: str, n_samples: int) -> pd.DataFrame:
    """
    Load the test export CSV and return a stratified sample.

    Stratified sampling preserves the urgent/normal class ratio so
    the evaluation metrics are comparable to the full test-set metrics
    reported by the notebook.

    Args:
        csv_path: Path to test_export.csv produced by the training notebook.
        n_samples: Total number of rows to sample.

    Returns:
        Shuffled DataFrame with at least these columns:
        clean_text, true_label_str, ml_label_str, ml_confidence.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    # Make sure the input file actually exists before trying to read it.
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Test export not found at '{csv_path}'. "
            "Run the Colab export cells first and place the file there."
        )

    # Load the exported test dataset into a pandas DataFrame.
    df = pd.read_csv(csv_path)

    # These are the minimum columns the script needs for a fair comparison.
    required_cols = {"clean_text", "true_label_str", "ml_label_str", "ml_confidence"}

    # Check whether any required columns are missing.
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Stratified sample: preserve class distribution
    # Split the dataset by class so we can sample each class separately.
    urgent_df = df[df["true_label_str"] == "urgent"]
    normal_df = df[df["true_label_str"] == "normal"]

    # Measure how common urgent tickets are in the full dataset.
    urgent_ratio = len(urgent_df) / len(df)

    # Choose how many urgent and normal rows to sample.
    n_urgent = max(1, round(n_samples * urgent_ratio))
    n_normal = n_samples - n_urgent

    # Cap at available rows
    # Prevent sampling more rows than the dataset actually has.
    n_urgent = min(n_urgent, len(urgent_df))
    n_normal = min(n_normal, len(normal_df))

    # Sample both classes separately, then shuffle the final sample.
    sample = pd.concat([
        urgent_df.sample(n=n_urgent, random_state=42),
        normal_df.sample(n=n_normal, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Log the final sample size so we can confirm the script loaded data correctly.
    logger.info(
        "Sampled %d tickets (%d urgent, %d normal) from %s",
        len(sample), n_urgent, n_normal, csv_path,
    )
    return sample


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------
def classify_ticket(
    client: OpenAI, text: str, model_name: str
) -> tuple[TicketPriority, float, int, int]:
    """
    Send a single ticket to the LLM and return a structured prediction.

    Uses client.beta.chat.completions.parse so the OpenAI SDK validates
    the JSON response against TicketPriority before returning it.

    Args:
        client: Authenticated OpenAI client instance.
        text: The cleaned ticket text to classify.

    Returns:
        Tuple of (TicketPriority, latency_ms, input_tokens, output_tokens).

    Raises:
        openai.OpenAIError: On API errors (rate limit, timeout, etc.).
    """
    # Start a timer so we can measure how long one LLM call takes.
    start = time.perf_counter()

    # Send the prompt and ticket text to the LLM and force a structured response.
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            # User prompt contains only the per-request data
            {"role": "user", "content": f"Support ticket:\n{text}"},
        ],
        response_format=TicketPriority,
    )

    # Convert elapsed time into milliseconds.
    latency_ms = (time.perf_counter() - start) * 1000

    # Read the validated structured response from the first choice.
    prediction: TicketPriority = response.choices[0].message.parsed

    # Read token usage so we can estimate API cost.
    usage = response.usage

    # Return the prediction plus timing and token details.
    return prediction, latency_ms, usage.prompt_tokens, usage.completion_tokens


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Compute the dollar cost of one OpenAI API call.

    Args:
        input_tokens: Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.

    Returns:
        Cost in USD as a float.
    """
    # Multiply token counts by the published prices to estimate cost.
    return (
        input_tokens * GPT4O_MINI_INPUT_COST_PER_TOKEN
        + output_tokens * GPT4O_MINI_OUTPUT_COST_PER_TOKEN
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(
    sample: pd.DataFrame,
    client: OpenAI,
    model_name: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Run LLM zero-shot classification on every row in sample.

    Logs progress every 25 rows. Rows that fail (e.g., due to a transient
    API error) are recorded with null prediction fields and a non-null
    error message — they are excluded from metric computation but kept
    in the output for debugging.

    Args:
        sample: DataFrame with clean_text, true_label_str, ml_label_str,
                ml_confidence columns.
        client: Authenticated OpenAI client.
        output_path: File path where per-row results CSV will be written.

    Returns:
        DataFrame with one row per input ticket, LLM predictions appended.
    """
    # Create the output folder if it does not already exist.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Store one result dictionary per processed ticket.
    rows = []

    # Go through the sampled tickets one by one.
    for _, row in sample.iterrows():
        try:
            # Ask the LLM to classify the current ticket.
            prediction, latency_ms, in_tok, out_tok = classify_ticket(
                client, str(row["clean_text"]), model_name
            )

            # Estimate the cost of this one API call.
            cost = compute_cost(in_tok, out_tok)

            # Save the successful prediction and all comparison details.
            rows.append({
                "text": row["clean_text"],
                "true_label": row["true_label_str"],
                "ml_label": row["ml_label_str"],
                "ml_confidence": row["ml_confidence"],
                "llm_label": prediction.label.value,
                "llm_confidence": prediction.confidence,
                "llm_reasoning": prediction.reasoning,
                "latency_ms": round(latency_ms, 2),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": round(cost, 8),
                "error": None,
            })

        except Exception as exc:
            # Log the error but continue — do not abort the whole run
            logger.warning("Row failed: %s", exc)

            # Save the failed row too, so we can inspect problems later.
            rows.append({
                "text": row["clean_text"],
                "true_label": row["true_label_str"],
                "ml_label": row["ml_label_str"],
                "ml_confidence": row["ml_confidence"],
                "llm_label": None,
                "llm_confidence": None,
                "llm_reasoning": None,
                "latency_ms": None,
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "error": str(exc),
            })

        # Track how many rows have been processed so far.
        completed = len(rows)

        # Periodically save a checkpoint so partial progress is not lost.
        if completed % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

        # Print progress every checkpoint interval for long runs.
        if completed % CHECKPOINT_EVERY == 0:
            logger.info("Progress: %d / %d", completed, len(sample))

    # Convert all collected rows into a DataFrame.
    results_df = pd.DataFrame(rows)

    # Save detailed per-ticket results for later review.
    results_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info("Per-row results saved to %s", output_path)
    return results_df


# ---------------------------------------------------------------------------
# Comparison reporting
# ---------------------------------------------------------------------------
def print_comparison(results_df: pd.DataFrame) -> None:
    """
    Print a side-by-side comparison table of ML vs LLM zero-shot metrics.

    Metrics are computed only on the rows where the LLM returned a valid
    prediction (no error). The ML metrics are also computed on this same
    subset so both numbers are directly comparable.

    Args:
        results_df: DataFrame produced by run_evaluation.
    """
    # Drop rows where LLM failed so both models are evaluated on the same set
    # Only keep rows where the LLM returned a valid label.
    valid = results_df.dropna(subset=["llm_label"])

    # Count how many rows failed.
    failed = len(results_df) - len(valid)
    if failed > 0:
        logger.warning("%d rows excluded from metrics due to API errors.", failed)

    # Convert string labels into 0/1 values for metric calculation.
    y_true = (valid["true_label"] == "urgent").astype(int)
    y_ml = (valid["ml_label"] == "urgent").astype(int)
    y_llm = (valid["llm_label"] == "urgent").astype(int)

    # Define a small helper function so we can calculate the same metrics for both models.
    def metrics(y_pred: pd.Series) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_urgent": precision_score(y_true, y_pred, zero_division=0),
            "recall_urgent": recall_score(y_true, y_pred, zero_division=0),
            "f1_urgent": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred),
        }

    # Compute the metric dictionary for the ML baseline and the LLM.
    ml_m = metrics(y_ml)
    llm_m = metrics(y_llm)

    # Compute average latency and cost across all valid LLM calls.
    avg_latency = valid["latency_ms"].mean()
    avg_cost = valid["cost_usd"].mean()
    total_cost = valid["cost_usd"].sum()

    # Main comparison table
    print("\n" + "=" * 65)
    print(f"{'METRIC':<30} {'ML Baseline':>15} {'LLM Zero-Shot':>15}")
    print("=" * 65)
    labels = {
        "accuracy": "Accuracy",
        "precision_urgent": "Precision (urgent)",
        "recall_urgent": "Recall (urgent)",
        "f1_urgent": "F1 (urgent)",
        "roc_auc": "ROC-AUC",
    }
    for key, label in labels.items():
        print(f"{label:<30} {ml_m[key]:>15.4f} {llm_m[key]:>15.4f}")
    print("-" * 65)
    print(f"{'Avg latency (ms)':<30} {'~0.04':>15} {avg_latency:>15.1f}")
    print(f"{'Avg cost per call ($)':<30} {'$0.00':>15} {avg_cost:>15.8f}")
    print(f"{'Total cost — this sample ($)':<30} {'$0.00':>15} {total_cost:>15.6f}")
    print("=" * 65)

    # Scale-up projection
    # Estimate what LLM cost and sequential runtime would look like at a larger workload.
    cost_10k = avg_cost * 10_000
    latency_10k_sequential_min = (avg_latency * 10_000) / 1000 / 60
    print(f"\nAt 10,000 tickets/hour (sequential):")
    print(f"  LLM  — cost: ${cost_10k:.2f}/hr  |  latency: ~{latency_10k_sequential_min:.0f} min to process")
    print(f"  ML   — cost: $0.00/hr  |  latency: ~{0.04 * 10_000 / 1000:.1f}s to process")
    print()

    # Full classification reports
    print("─" * 65)
    print("ML Baseline — Classification Report")
    print("─" * 65)
    print(classification_report(y_true, y_ml, target_names=["normal", "urgent"]))

    print("─" * 65)
    print("LLM Zero-Shot — Classification Report")
    print("─" * 65)
    print(classification_report(y_true, y_llm, target_names=["normal", "urgent"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    # Create the command-line parser used when the script is run from terminal.
    parser = argparse.ArgumentParser(
        description="Evaluate LLM zero-shot priority classification vs ML baseline."
    )

    # Let the user choose which input CSV to evaluate.
    parser.add_argument(
        "--input",
        default="data/processed/test_export.csv",
        help="Path to test_export.csv produced by the training notebook.",
    )

    # Let the user choose how many sampled rows to send to the LLM.
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="Number of tickets to evaluate (stratified sample). Default: 300.",
    )

    # Let the user choose where the detailed results CSV should be saved.
    parser.add_argument(
        "--output",
        default="logs/llm_eval_results.csv",
        help="Where to save per-row results CSV. Default: logs/llm_eval_results.csv.",
    )

    # Let the user choose which OpenAI model to test while keeping the rest fixed.
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=f"OpenAI model to use. Default: {DEFAULT_LLM_MODEL}.",
    )

    # Return the parsed command-line values.
    return parser.parse_args()


def main() -> None:
    """Entry point: load data, run evaluation, print comparison."""
    # Read user-provided command-line options.
    args = parse_args()

    # Load environment variables and confirm the API key is available.
    load_env()

    # OpenAI client reads OPENAI_API_KEY from the environment automatically
    client = OpenAI()

    # Load a stratified test sample from the exported CSV.
    sample = load_test_sample(args.input, args.n_samples)

    # Run LLM predictions on the sample and save row-level results.
    results_df = run_evaluation(sample, client, args.model, args.output)

    # Print the ML-versus-LLM comparison summary.
    print_comparison(results_df)


# Run the script only when this file is executed directly.
if __name__ == "__main__":
    main()

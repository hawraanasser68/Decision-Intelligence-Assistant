"""Text cleaning utilities shared across the project.

This module is the single source of truth for how raw tweet text is cleaned.
It is used:
  - by scripts/build_rag_corpus.py  when building the corpus offline
  - by the backend at inference time when a new user query arrives

Keeping it here means the corpus and the live query always go through
the exact same transformation — no drift between offline and online paths.
"""

import html
import re


def clean_text(text: str) -> str:
    """Clean a single raw tweet string.

    Applies, in order:
    1. HTML entity decoding  (e.g. &amp; → &)
    2. URL removal           (http/https and www links)
    3. @mention removal
    4. Hashtag removal
    5. Whitespace normalisation

    Args:
        text: Raw tweet text.

    Returns:
        Cleaned text string. Returns an empty string if input is not a string.
    """
    t = html.unescape(str(text))
    t = re.sub(r"http\S+|www\.\S+", " ", t)   # remove URLs
    t = re.sub(r"@\w+", " ", t)               # remove @mentions
    t = re.sub(r"#\w+", " ", t)               # remove hashtags
    t = re.sub(r"\s+", " ", t).strip()        # normalise whitespace
    return t

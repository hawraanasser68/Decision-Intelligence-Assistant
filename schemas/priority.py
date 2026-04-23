"""Priority label schemas used by the LLM zero-shot evaluation flow.

This module defines the allowed ticket labels and the structured
prediction object returned by the LLM. The evaluation script uses these
schemas to validate model output before computing metrics.
"""

# Import Enum so we can create a fixed list of allowed label values.
from enum import Enum

# Import BaseModel to build Pydantic schemas.
# Import Field to add validation rules and human-readable descriptions.
from pydantic import BaseModel, Field


# Create an enum for the only valid ticket priority labels.
class Priority(str, Enum):
    """Valid priority labels the LLM may return."""

    # One allowed value: the ticket is urgent.
    urgent = "urgent"

    # One allowed value: the ticket is normal.
    normal = "normal"


# Create the schema for one structured LLM prediction.
class TicketPriority(BaseModel):
    """
    Structured LLM response for a single ticket classification.

    The OpenAI client validates the model's output against this schema
    before returning it, so the calling code never receives a malformed
    or partial response.
    """

    # Store the predicted label using the Priority enum above.
    label: Priority

    # Store the model confidence and force it to stay between 0 and 1.
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction confidence between 0.0 and 1.0"
    )

    # Store a short human-readable explanation for the decision.
    reasoning: str = Field(
        ..., description="One-sentence explanation of the classification decision"
    )
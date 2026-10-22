from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)
    top_k: int = Field(1, ge=1, le=20)
    mode: Literal["autoregressive", "parallel"] = "autoregressive"
    use_hierarchy: bool = True


class PathPrediction(BaseModel):
    codes: list[str]
    code_string: str
    score: float
    level_probs: list[float]


class PredictResponse(BaseModel):
    predictions: list[list[PathPrediction]]


class FeedbackRequest(BaseModel):
    """A realtime training pair. Aliases match the realtime data schema."""

    predictor_text: str = Field(..., min_length=1)
    output_text: str = Field(..., min_length=1)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_levels: int | None = None
    vocab_size: int | None = None
    n_paths: int | None = None

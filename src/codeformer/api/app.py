"""FastAPI service.

Environment variables
---------------------
CODEFORMER_CHECKPOINT   path to a checkpoint produced by ``codeformer train`` (required)
CODEFORMER_DEVICE       ``cpu`` (default) or ``cuda``
CODEFORMER_FEEDBACK     JSONL file that /feedback appends realtime pairs to
                        (default: ``artifacts/feedback.jsonl``)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from codeformer.api.schemas import (
    FeedbackRequest,
    HealthResponse,
    PathPrediction,
    PredictRequest,
    PredictResponse,
)
from codeformer.data.codes import parse_codes
from codeformer.inference.predictor import Predictor

logger = logging.getLogger(__name__)


def create_app(
    checkpoint: str | None = None,
    feedback_path: str | None = None,
    predictor: Predictor | None = None,
) -> FastAPI:
    checkpoint = checkpoint or os.environ.get("CODEFORMER_CHECKPOINT")
    feedback_file = Path(
        feedback_path or os.environ.get("CODEFORMER_FEEDBACK", "artifacts/feedback.jsonl")
    )
    device = os.environ.get("CODEFORMER_DEVICE", "cpu")
    write_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor
        if app.state.predictor is None and checkpoint and Path(checkpoint).exists():
            app.state.predictor = Predictor.from_checkpoint(checkpoint, device)
            logger.info("Loaded checkpoint %s", checkpoint)
        elif app.state.predictor is None:
            logger.warning("No checkpoint available; /predict will return 503")
        yield

    app = FastAPI(title="codeformer", version="0.1.0", lifespan=lifespan)

    def get_predictor(request: Request) -> Predictor:
        p = request.app.state.predictor
        if p is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return p

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        p = request.app.state.predictor
        if p is None:
            return HealthResponse(status="degraded", model_loaded=False)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            n_levels=p.spec.n_levels,
            vocab_size=len(p.vocab),
            n_paths=len(p.hierarchy),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request) -> PredictResponse:
        p = get_predictor(request)
        try:
            results = p.predict(
                req.texts, top_k=req.top_k, mode=req.mode, use_hierarchy=req.use_hierarchy
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictResponse(
            predictions=[
                [
                    PathPrediction(
                        codes=list(pr.codes),
                        code_string=pr.as_string(p.spec),
                        score=pr.score,
                        level_probs=list(pr.level_probs),
                    )
                    for pr in per_text
                ]
                for per_text in results
            ]
        )

    @app.post("/feedback", status_code=202)
    def feedback(req: FeedbackRequest, request: Request) -> dict:
        """Queue a realtime (predictor_text, output_text) pair for incremental training."""
        p = get_predictor(request)
        try:
            parse_codes(req.output_text, p.spec)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with write_lock, open(feedback_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(req.model_dump()) + "\n")
        return {"status": "queued"}

    return app

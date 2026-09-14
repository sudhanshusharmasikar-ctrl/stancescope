"""
Run:  uvicorn app.api:app --reload
Docs: http://127.0.0.1:8000/docs

Two-step interaction model, because the agent can pause mid-run:
  1. POST /claims          -> create a claim + its posts
  2. POST /claims/{id}/run -> start a run. Returns either a finished report,
                               or status "awaiting_review" with the items
                               that need a human decision.
  3. POST /runs/{id}/review -> submit decisions for flagged items. Returns
                               the finished report.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from . import db
from .classifier import get_classifier
from .graph import build_graph

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    _state["graph"] = build_graph()
    yield
    _state.clear()


app = FastAPI(title="StanceScope", version="1.0.0", lifespan=lifespan)


class PostIn(BaseModel):
    text: str
    author: str | None = None
    image_path: str | None = None


class ClaimIn(BaseModel):
    text: str = Field(..., min_length=3)
    posts: list[PostIn]


class ClaimOut(BaseModel):
    claim_id: int
    post_count: int


class ReviewItem(BaseModel):
    post_id: int
    stance: str = Field(..., pattern="^(support|oppose|neutral)$")


class ReviewIn(BaseModel):
    decisions: list[ReviewItem]


def _thread_config(run_id: int) -> dict:
    return {"configurable": {"thread_id": f"run-{run_id}"}}


def _format_run_result(run_id: int, result: dict) -> dict:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        db.set_run_status(run_id, "awaiting_review")
        return {
            "run_id": run_id,
            "status": "awaiting_review",
            "review_items": payload["items"],
            "report": None,
        }

    report = result["report"]
    db.save_predictions(run_id, [
        {"post_id": p["post_id"], "stance": p["stance"],
         "confidence": p["confidence"], "source": p["source"]}
        for p in result["predictions"]
    ])
    db.set_run_status(run_id, "done", report=report)
    return {"run_id": run_id, "status": "done", "review_items": None, "report": report}


@app.post("/claims", response_model=ClaimOut)
def create_claim(claim: ClaimIn) -> ClaimOut:
    claim_id, posts = db.create_claim(
        claim.text, [p.model_dump() for p in claim.posts]
    )
    return ClaimOut(claim_id=claim_id, post_count=len(posts))


@app.post("/claims/{claim_id}/run")
def run_claim(claim_id: int) -> dict:
    claim_row = db.get_claim(claim_id)
    if claim_row is None:
        raise HTTPException(404, "Claim not found.")

    classifier = get_classifier()
    run_id = db.create_run(claim_id, classifier.source_name)

    graph = _state["graph"]
    state = {
        "run_id": run_id, "claim_id": claim_id, "claim_text": claim_row["text"],
        "evidence": [], "predictions": [], "low_confidence": [], "report": {},
    }
    result = graph.invoke(state, config=_thread_config(run_id))
    return _format_run_result(run_id, result)


@app.post("/runs/{run_id}/review")
def submit_review(run_id: int, review: ReviewIn) -> dict:
    run_row = db.get_run(run_id)
    if run_row is None:
        raise HTTPException(404, "Run not found.")
    if run_row["status"] != "awaiting_review":
        raise HTTPException(400, f"Run is not awaiting review (status: {run_row['status']}).")

    decisions = {str(d.post_id): d.stance for d in review.decisions}
    graph = _state["graph"]
    result = graph.invoke(Command(resume=decisions), config=_thread_config(run_id))
    return _format_run_result(run_id, result)


@app.get("/runs/{run_id}")
def get_run(run_id: int) -> dict:
    run_row = db.get_run(run_id)
    if run_row is None:
        raise HTTPException(404, "Run not found.")
    return run_row


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classifier": get_classifier().source_name}

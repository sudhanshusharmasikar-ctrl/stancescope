"""
The agent. Four nodes: retrieve evidence, classify posts, pause for human
review on low-confidence predictions, aggregate into a report.

The human-in-the-loop step is a real LangGraph interrupt, not a simulated
pause -- the graph execution genuinely stops and returns control to the
caller, resuming later from exactly where it left off once a review is
submitted. That's the mechanism behind LangGraph's "human-in-the-loop"
module; this is it actually wired up and run, not described.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

from . import db
from .classifier import get_classifier
from .config import CONFIDENCE_THRESHOLD
from .evidence import EvidenceRetriever


class RunState(TypedDict):
    run_id: int
    claim_id: int
    claim_text: str
    evidence: list[dict]
    predictions: list[dict]     # {post_id, text, stance, confidence, source}
    low_confidence: list[dict]  # subset needing review, filled in by classify node
    report: dict


def retrieve_evidence_node(state: RunState) -> RunState:
    retriever = EvidenceRetriever()
    hits = retriever.retrieve(state["claim_text"])
    state["evidence"] = [{"source": h.source, "text": h.text, "score": h.score} for h in hits]
    return state


def classify_posts_node(state: RunState) -> RunState:
    classifier = get_classifier()
    posts = db.get_posts(state["claim_id"])

    predictions = []
    low_confidence = []
    for post in posts:
        result = classifier.classify(post.text, post.image_path)
        pred = {
            "post_id": post.post_id,
            "text": post.text,
            "stance": result.stance,
            "confidence": result.confidence,
            "source": "model",
        }
        predictions.append(pred)
        if result.confidence < CONFIDENCE_THRESHOLD:
            low_confidence.append(pred)

    state["predictions"] = predictions
    state["low_confidence"] = low_confidence
    return state


def human_review_node(state: RunState) -> RunState:
    """
    If any prediction fell below the confidence threshold, this node calls
    interrupt() -- LangGraph stops graph execution here and returns the
    low-confidence items to the caller. The API layer surfaces them to a
    human, and when a review is submitted, invoke() is called again with a
    Command(resume=...), and execution continues from this exact point with
    the reviewed stances applied.
    """
    if not state["low_confidence"]:
        return state

    review_decisions = interrupt({
        "reason": "low_confidence_predictions",
        "items": state["low_confidence"],
    })
    # review_decisions: {"<post_id>": stance, ...} supplied on resume.
    # Keys must be strings -- LangGraph inspects resume-dict keys to detect
    # the multi-interrupt resume format, and that check raises a TypeError
    # on non-string keys rather than safely skipping them.
    for pred in state["predictions"]:
        key = str(pred["post_id"])
        if key in review_decisions:
            pred["stance"] = review_decisions[key]
            pred["confidence"] = 1.0
            pred["source"] = "human"

    return state


def aggregate_node(state: RunState) -> RunState:
    from .report import build_report
    state["report"] = build_report(state["claim_text"], state["predictions"], state["evidence"])
    return state


def build_graph():
    graph = StateGraph(RunState)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("classify_posts", classify_posts_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("aggregate", aggregate_node)

    graph.set_entry_point("retrieve_evidence")
    graph.add_edge("retrieve_evidence", "classify_posts")
    graph.add_edge("classify_posts", "human_review")
    graph.add_edge("human_review", "aggregate")
    graph.add_edge("aggregate", END)

    # MemorySaver checkpoints state at each node, which is what makes
    # interrupt/resume possible -- without a checkpointer, a stopped graph
    # has nowhere to resume FROM.
    return graph.compile(checkpointer=MemorySaver())

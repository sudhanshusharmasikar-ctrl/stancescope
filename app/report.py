"""
Turns per-post predictions and retrieved evidence into a summary. Every
number in the report can be traced back to the predictions that produced
it -- that traceability is the entire point of storing per-post predictions
in the database rather than only keeping an aggregate.
"""
from __future__ import annotations

from collections import Counter


def build_report(claim_text: str, predictions: list[dict], evidence: list[dict]) -> dict:
    counts = Counter(p["stance"] for p in predictions)
    total = len(predictions)

    breakdown = {
        stance: {
            "count": counts.get(stance, 0),
            "pct": round(counts.get(stance, 0) / total * 100, 1) if total else 0.0,
        }
        for stance in ("support", "oppose", "neutral")
    }

    human_reviewed = sum(1 for p in predictions if p["source"] == "human")

    return {
        "claim": claim_text,
        "total_posts": total,
        "breakdown": breakdown,
        "human_reviewed_count": human_reviewed,
        "evidence": evidence,
        "post_level_detail": [
            {"post_id": p["post_id"], "text": p["text"], "stance": p["stance"],
             "confidence": p["confidence"], "source": p["source"]}
            for p in predictions
        ],
    }

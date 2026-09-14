"""
Evidence retrieval: given a claim, finds the most relevant supporting/refuting
news snippets from a small seeded corpus, embedded with the same MiniLM model
used in PaperRAG. In a full deployment this corpus would be a live news feed
or search API; here it's a fixed seed set so the project runs without a paid
news API key. See seed_evidence.py for the data and app/report.py for how
retrieved evidence is cited in the final report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL, EVIDENCE_TOP_K
from .seed_evidence import EVIDENCE_CORPUS

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


@dataclass
class EvidenceHit:
    source: str
    text: str
    score: float


class EvidenceRetriever:
    def __init__(self):
        self.corpus = EVIDENCE_CORPUS
        texts = [e["text"] for e in self.corpus]
        self.embeddings = get_model().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def retrieve(self, claim_text: str, top_k: int = EVIDENCE_TOP_K) -> list[EvidenceHit]:
        qv = get_model().encode([claim_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self.embeddings @ qv
        order = np.argsort(-scores)[:top_k]
        return [
            EvidenceHit(self.corpus[i]["source"], self.corpus[i]["text"], float(scores[i]))
            for i in order
        ]

"""
Pluggable stance classifier.

This is the seam where your actual fine-tuned BERT+ViT model plugs in.
Set STANCESCOPE_CHECKPOINT to your checkpoint directory and implement
`_load_real_classifier` below to load it -- the exact loading code depends
on how you saved the checkpoint (state_dict, HF save_pretrained, etc.),
which only you have, so it's stubbed here rather than guessed.

Until that's wired in, StubClassifier runs so every other part of the
pipeline (retrieval, orchestration, DB, human review, report) is fully
real and testable today.

BOTH classifiers implement the same interface: classify(text, image_path)
-> (stance, confidence). Nothing else in the codebase needs to know which
one is running, which is the point -- swapping in your real model later
should not require touching graph.py, db.py, or api.py at all.
"""
from __future__ import annotations

import hashlib
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import CLASSIFIER_CHECKPOINT

STANCES = ("support", "oppose", "neutral")


@dataclass
class ClassificationResult:
    stance: str
    confidence: float


class BaseClassifier(ABC):
    source_name: str

    @abstractmethod
    def classify(self, text: str, image_path: str | None = None) -> ClassificationResult: ...


class StubClassifier(BaseClassifier):
    """
    Deterministic, keyword-based placeholder. NOT a real stance classifier --
    it exists only so the pipeline around it is testable without your
    trained weights. Say this plainly if asked; do not present stub output
    as model output.
    """
    source_name = "stub"

    _SUPPORT_WORDS = {"agree", "support", "yes", "true", "confirmed", "will happen", "great"}
    _OPPOSE_WORDS = {"disagree", "oppose", "no", "false", "denied", "wrong", "never", "bad"}

    def classify(self, text: str, image_path: str | None = None) -> ClassificationResult:
        t = text.lower()
        support_hits = sum(1 for w in self._SUPPORT_WORDS if w in t)
        oppose_hits = sum(1 for w in self._OPPOSE_WORDS if w in t)

        # deterministic pseudo-confidence, so repeated runs on the same input
        # are reproducible for demoing/testing. Python's built-in hash() is
        # randomized per-process for security and is NOT stable across runs
        # -- md5 is used here purely as a fast, stable hash, not for security.
        seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**32)
        rnd = random.Random(seed)
        base_conf = 0.55 + rnd.random() * 0.3

        if support_hits > oppose_hits:
            return ClassificationResult("support", round(base_conf, 3))
        if oppose_hits > support_hits:
            return ClassificationResult("oppose", round(base_conf, 3))
        return ClassificationResult("neutral", round(0.5 + rnd.random() * 0.2, 3))


class CheckpointClassifier(BaseClassifier):
    """
    Loads your real fine-tuned BERT + ViT model.

    Fill in `_load` and `classify` once your checkpoint format is settled.
    Suggested contract based on your research code:
        - tokenizer + BERT encoder for the text (+ target)
        - image transform + ViT encoder for the image, if present
        - fused classification head -> logits over {support, oppose, neutral}
        - softmax the logits; confidence = max class probability
    """
    source_name = f"checkpoint:{CLASSIFIER_CHECKPOINT}"

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self._load()

    def _load(self) -> None:
        raise NotImplementedError(
            "Wire in your real BERT+ViT loading code here. See your thesis "
            "training script for how the checkpoint was saved."
        )

    def classify(self, text: str, image_path: str | None = None) -> ClassificationResult:
        raise NotImplementedError("Wire in your real forward pass here.")


def get_classifier() -> BaseClassifier:
    if CLASSIFIER_CHECKPOINT:
        return CheckpointClassifier(CLASSIFIER_CHECKPOINT)
    return StubClassifier()

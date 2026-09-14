"""
Central configuration.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.getenv("STANCESCOPE_DB", ROOT / "data" / "stancescope.db"))
STORAGE_DIR = Path(os.getenv("STANCESCOPE_STORAGE", ROOT / "storage"))

# ---------- classifier ----------
# Point this at your real fine-tuned BERT+ViT checkpoint directory once you
# have it. Until then, the stub classifier runs so the rest of the pipeline
# (retrieval, orchestration, DB, human review, report) is fully testable.
# See app/classifier.py for the loading contract your checkpoint must satisfy.
CLASSIFIER_CHECKPOINT = os.getenv("STANCESCOPE_CHECKPOINT", "")

# ---------- evidence retrieval ----------
EMBED_MODEL = os.getenv("STANCESCOPE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EVIDENCE_TOP_K = int(os.getenv("STANCESCOPE_EVIDENCE_TOP_K", 3))

# ---------- confidence / human review ----------
# A prediction below this confidence is routed to human review instead of
# being aggregated automatically. This number is a placeholder -- like
# PaperRAG's similarity threshold, it should be set from a calibration run
# on labelled data, not guessed.
CONFIDENCE_THRESHOLD = float(os.getenv("STANCESCOPE_CONF_THRESHOLD", 0.6))

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# StanceScope

An agentic pipeline that takes a claim and a set of social posts, retrieves supporting news evidence, classifies each post's stance, pauses for human review when it isn't confident, and produces a report where every number traces back to a specific post.

> **Status: orchestration, storage, API and UI built and tested end-to-end. Stance classification currently runs on a stub, not the real fine-tuned model — see below.**

---

## The honest state of this project

This is the part most people leave out of a README, and it's the part an interviewer actually cares about.

**What is real and tested:** the LangGraph agent, the human-in-the-loop interrupt and resume, the database persistence, the evidence retrieval, the FastAPI backend, and the Streamlit UI. Every one of these was run — not just written — during development, including a genuine LangGraph `interrupt()`/`Command(resume=...)` round trip verified through the actual HTTP API. Two real bugs were caught doing this (see below) and would not have been caught by reading the code alone.

**What is a placeholder:** the stance classifier. `app/classifier.py` ships with a `StubClassifier` — simple keyword matching, not a trained model — so the rest of the pipeline is fully demonstrable without the real weights. The real classifier is my fine-tuned BERT+ViT multimodal model from my MTech thesis (see `stance-detection-framework` repo). Wiring it in means implementing `CheckpointClassifier._load` and `.classify` in `app/classifier.py` to load that checkpoint and run a forward pass. Until that's done, **do not present this project as having a working stance classifier** — it has a working agent framework with a placeholder where the classifier goes.

## Bugs caught by actually running this, not just writing it

Worth keeping in an interview back pocket, because "how did you find this" is a good follow-up question and this is a genuine answer:

1. **Non-deterministic "deterministic" confidence.** The stub classifier originally seeded its randomness with Python's built-in `hash()`. Python randomizes string hashing per-process for security, so the same input produced different confidence scores on every run — the opposite of the intended behaviour. Fixed by seeding with `hashlib.md5` instead, which is stable across runs.
2. **Resume dict with integer keys crashed LangGraph's interrupt resolution.** LangGraph inspects the keys of a `Command(resume=...)` dict to detect a specific multi-interrupt resume format, and that inspection assumes string keys — it raises a `TypeError` on an integer key rather than failing gracefully. Fixed by converting `post_id` to a string when building the resume payload.
3. **A malformed INSERT statement** (`app/db.py`) listed 5 columns but only supplied 4 placeholder values, missing `:run_id`. This didn't fail until the very first real prediction was saved to the database.

## Architecture

```
POST /claims             -> stores claim + posts
POST /claims/{id}/run    -> starts the LangGraph agent:

    retrieve_evidence ──► classify_posts ──► human_review ──► aggregate
         │                      │                 │               │
    embeds claim,          runs stance         PAUSES via      builds
    searches seeded        classifier on       LangGraph        report
    news corpus,           every post,         interrupt() if   with per-
    returns top-k          flags low-          any prediction   post detail
    matches                confidence ones     is below         and cited
                                               threshold         evidence

    If paused: API returns status="awaiting_review" + the flagged items.
    POST /runs/{id}/review submits decisions, which resumes the graph
    exactly where it paused via Command(resume=...).
```

### Why a real interrupt instead of a manual "pending" flag

An easier-sounding approach would be: if confidence is low, save a "pending" row and let a separate endpoint update it later. That works, but it isn't what LangGraph's human-in-the-loop feature actually is — it's a workaround built next to the framework instead of using it. Using `interrupt()` and `Command(resume=...)` means the framework's own checkpointer (here, `MemorySaver`) is responsible for suspending and resuming exact execution state, which is the mechanism actually worth being able to explain in an interview.

### Why predictions are stored per-post, not just as an aggregate

Every prediction row records which post it came from, what stance was assigned, the confidence, and whether it came from the model or a human. That's what makes "why does the report say 62% oppose" answerable by a query instead of a shrug.

## Stack

Python · LangGraph (with `MemorySaver` checkpointing) · FastAPI · SQLite · sentence-transformers · Streamlit

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --reload        # terminal 1
streamlit run ui/streamlit_app.py   # terminal 2
```

The Streamlit UI has a built-in sample claim (a telecom merger, with 8 posts and a seeded evidence corpus) so you can see a full run — including a review pause — without typing anything in first.

## Wiring in the real classifier

1. Implement `CheckpointClassifier._load` and `.classify` in `app/classifier.py` using your thesis training code's checkpoint format.
2. Set `STANCESCOPE_CHECKPOINT=/path/to/your/checkpoint`.
3. Nothing else changes — `graph.py`, `api.py`, and the UI all call `get_classifier()` and don't know or care which implementation is running. That's the point of the interface.

## Known limitations

- Evidence corpus is a small fixed seed set (`app/seed_evidence.py`), not a live news feed.
- Confidence threshold (`STANCESCOPE_CONF_THRESHOLD`, default 0.6) is a placeholder, not calibrated against labelled data.
- No authentication — this is a local demo, not a deployed multi-user service.
- Stub classifier is keyword matching, explicitly not real stance detection.

## Results

*To be filled in once the real classifier is wired in and evaluated against labelled stance data — see the parent `stance-detection-framework` repo for the classifier's own benchmark numbers.*

## License

MIT

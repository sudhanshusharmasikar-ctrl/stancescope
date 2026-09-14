"""
Run:  streamlit run ui/streamlit_app.py

Two-phase UI matching the two-phase API: submit a claim and its posts, run
the analysis, and if the agent pauses for review, a form appears right there
for the flagged items before the report is shown.
"""
import os
import sys
from pathlib import Path

# Streamlit puts THIS file's folder (ui/) on sys.path, not the project root,
# so `from app...` can fail depending on the Streamlit version and where you
# launched from. Add the project root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

API = os.getenv("STANCESCOPE_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="StanceScope", layout="wide")
st.title("StanceScope")
st.caption("Give it a claim and a set of posts. It retrieves evidence, classifies "
           "each post's stance, pauses for your review when it isn't confident, "
           "and reports a breakdown with every number traceable to a specific post.")

if "phase" not in st.session_state:
    st.session_state.phase = "input"
if "run_data" not in st.session_state:
    st.session_state.run_data = None

try:
    h = requests.get(f"{API}/health", timeout=5).json()
    st.sidebar.success(f"API up · classifier: {h['classifier']}")
    if h["classifier"] == "stub":
        st.sidebar.warning("Running the stub classifier, not the real fine-tuned "
                            "model. Set STANCESCOPE_CHECKPOINT to switch.")
except Exception:
    st.sidebar.error("API unreachable. Start it with `uvicorn app.api:app`.")

if st.session_state.phase == "input":
    from app.seed_evidence import SAMPLE_CLAIM, SAMPLE_POSTS

    use_sample = st.checkbox("Use the built-in sample claim", value=True)

    if use_sample:
        claim_text = SAMPLE_CLAIM
        posts_text = "\n".join(p["text"] for p in SAMPLE_POSTS)
    else:
        claim_text = st.text_input("Claim", placeholder="The merger will be approved")
        posts_text = st.text_area("Posts (one per line)", height=200)

    st.text_area("Claim to analyse", value=claim_text, disabled=True, height=60)
    st.text_area("Posts", value=posts_text, disabled=True, height=150)

    if st.button("Run analysis", type="primary") and claim_text:
        posts = [{"text": line.strip()} for line in posts_text.splitlines() if line.strip()]
        try:
            r = requests.post(f"{API}/claims", json={"text": claim_text, "posts": posts}, timeout=30)
            r.raise_for_status()
            claim_id = r.json()["claim_id"]
            r2 = requests.post(f"{API}/claims/{claim_id}/run", timeout=60)
            r2.raise_for_status()
            st.session_state.run_data = r2.json()
            st.session_state.phase = (
                "review" if r2.json()["status"] == "awaiting_review" else "report"
            )
            st.rerun()
        except Exception as e:
            st.error(f"Request failed: {e}")

elif st.session_state.phase == "review":
    data = st.session_state.run_data
    st.warning(f"{len(data['review_items'])} post(s) had low model confidence "
               "and need your decision before the report can be finished.")

    decisions = []
    for item in data["review_items"]:
        st.write(f"**Post:** {item['text']}")
        st.caption(f"Model guessed: {item['stance']} (confidence {item['confidence']:.2f})")
        choice = st.radio(
            "Your call", ["support", "oppose", "neutral"],
            index=["support", "oppose", "neutral"].index(item["stance"]),
            key=f"review_{item['post_id']}", horizontal=True,
        )
        decisions.append({"post_id": item["post_id"], "stance": choice})
        st.divider()

    if st.button("Submit review", type="primary"):
        try:
            r = requests.post(
                f"{API}/runs/{data['run_id']}/review",
                json={"decisions": decisions}, timeout=30,
            )
            r.raise_for_status()
            st.session_state.run_data = r.json()
            st.session_state.phase = "report"
            st.rerun()
        except Exception as e:
            st.error(f"Request failed: {e}")

elif st.session_state.phase == "report":
    report = st.session_state.run_data["report"]
    st.subheader(f"Claim: {report['claim']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total posts", report["total_posts"])
    c2.metric("Support", f"{report['breakdown']['support']['pct']}%")
    c3.metric("Oppose", f"{report['breakdown']['oppose']['pct']}%")
    c4.metric("Human-reviewed", report["human_reviewed_count"])

    st.subheader("Supporting evidence")
    for e in report["evidence"]:
        st.markdown(f"**{e['source']}** (similarity {e['score']:.3f}) — {e['text']}")

    st.subheader("Post-level detail")
    for p in report["post_level_detail"]:
        tag = "🧑 human" if p["source"] == "human" else "🤖 model"
        st.write(f"[{p['stance'].upper()}] ({p['confidence']:.2f}, {tag}) {p['text']}")

    if st.button("Start a new analysis"):
        st.session_state.phase = "input"
        st.session_state.run_data = None
        st.rerun()

"""
ui/app.py
---------
Streamlit UI for the Conversation Evaluation system.

Features:
  - Paste any conversation and score it
  - Browse pre-computed sample conversation scores
  - Filter scores by category
  - Visual bar chart of score distribution
  - Confidence colour-coding
"""

import streamlit as st
import requests
import json
import os

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="Conversation Evaluator",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------

def score_color(score: int) -> str:
    """Returns a hex colour based on score value (red → yellow → green)."""
    colors = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71", 5: "#27ae60"}
    return colors.get(score, "#95a5a6")


def confidence_badge(conf: float) -> str:
    """Returns a text label for confidence level."""
    if conf >= 0.8:
        return "🟢 High"
    elif conf >= 0.5:
        return "🟡 Medium"
    else:
        return "🔴 Low"


def call_evaluate(conversation: str, conv_id: str = "user_input") -> dict:
    """Calls the FastAPI /evaluate endpoint."""
    resp = requests.post(
        f"{API_BASE}/evaluate",
        json={"conversation": conversation, "conversation_id": conv_id},
        timeout=600  # evaluation can take a few minutes
    )
    resp.raise_for_status()
    return resp.json()


def get_facets() -> list:
    """Fetches the full facet list from the API."""
    resp = requests.get(f"{API_BASE}/facets", timeout=10)
    resp.raise_for_status()
    return resp.json()["facets"]


def get_samples() -> list:
    """Fetches the list of sample conversations."""
    resp = requests.get(f"{API_BASE}/samples", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_sample_scores(conv_id: str) -> dict:
    """Fetches pre-computed scores for a sample conversation."""
    resp = requests.get(f"{API_BASE}/samples/{conv_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------------------------------

def display_results(data: dict):
    """Renders evaluation results in the Streamlit UI."""
    st.success(f"✅ Scored {data.get('total_facets_scored', data.get('total_facets', '?'))} facets")

    # Support both flat and nested (summary) response formats
    summary = data.get("summary", data)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score", f"{summary['average_score']} / 5")
    col2.metric("Average Confidence", f"{round(summary['average_confidence']*100)}%")
    col3.metric("Conversation ID", data["conversation_id"])

    # Score distribution bar chart
    st.subheader("Score Distribution")
    dist = summary["score_distribution"]
    chart_data = {f"Score {k}": v for k, v in dist.items()}
    st.bar_chart(chart_data)

    # Filters
    st.subheader("Individual Facet Scores")
    scores = data["scores"]
    categories = sorted(set(s["category"] for s in scores))
    selected_cat = st.selectbox("Filter by Category", ["All"] + categories)
    min_score = st.slider("Minimum Score", 1, 5, 1)

    filtered = [
        s for s in scores
        if (selected_cat == "All" or s["category"] == selected_cat)
        and s["score"] >= min_score
    ]

    st.write(f"Showing {len(filtered)} of {len(scores)} facets")

    # Display each facet as an expander
    for s in filtered:
        color = score_color(s["score"])
        badge = confidence_badge(s["confidence"])
        label = f"**{s['facet_name']}** | Score: :{color}[{s['score']}] | {badge} | {s['category']}"

        with st.expander(f"[{s['score']}/5] {s['facet_name']} ({s['category']})"):
            col_a, col_b = st.columns([1, 3])
            col_a.metric("Score", f"{s['score']} / 5")
            col_a.metric("Confidence", f"{round(s['confidence']*100)}%")
            col_b.write(f"**Reason:** {s['reason']}")
            col_b.write(f"**Scoring Type:** {s['scoring_type']}")


# --------------------------------------------------------------------------
# MAIN UI LAYOUT
# --------------------------------------------------------------------------

st.title("🧠 Conversation Evaluation Benchmark")
st.markdown(
    "Scores any conversation turn across **300+ facets** covering "
    "linguistic quality, pragmatics, safety, and emotion."
)

tab1, tab2, tab3 = st.tabs(["📝 Evaluate New Conversation", "📊 Browse Samples", "📋 Facet Explorer"])

# ---- TAB 1: Evaluate new conversation ----
with tab1:
    st.subheader("Paste a Conversation")
    st.markdown(
        "Format: each turn on its own line, prefixed with `User:` or `Assistant:`"
    )

    default_text = """User: I've been struggling with motivation lately. Nothing feels exciting anymore.
Assistant: I hear you. That kind of flatness can be really draining. Has anything changed recently — work, relationships, routine?
User: Not really. I just feel stuck.
Assistant: Sometimes feeling stuck is your mind signalling that something needs to change, even if it's not obvious what. What used to excite you that doesn't anymore?"""

    conversation_input = st.text_area(
        "Conversation", value=default_text, height=250
    )
    conv_id_input = st.text_input("Conversation ID (optional)", value="user_eval_001")

    if st.button("🚀 Evaluate", type="primary"):
        if not conversation_input.strip():
            st.error("Please enter a conversation.")
        else:
            with st.spinner("Evaluating... this may take 2-5 minutes for all 300+ facets..."):
                try:
                    result = call_evaluate(conversation_input, conv_id_input)
                    display_results(result)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API. Make sure FastAPI is running: `uvicorn api.main:app`")
                except Exception as e:
                    st.error(f"Error: {e}")

# ---- TAB 2: Browse sample conversations ----
with tab2:
    st.subheader("Pre-evaluated Sample Conversations")
    st.markdown("These 50 conversations cover diverse scenarios: emotional support, technical help, toxic behaviour, spiritual discussion, and more.")

    try:
        samples = get_samples()
        selected = st.selectbox(
            "Choose a sample",
            options=[s["id"] for s in samples],
            format_func=lambda x: f"{x} — {next(s['case_type'] for s in samples if s['id'] == x)}"
        )

        if st.button("Load Scores"):
            with st.spinner("Loading..."):
                try:
                    data = get_sample_scores(selected)
                    # Show conversation text
                    with st.expander("View Conversation"):
                        st.text(data["conversation"])
                    display_results(data)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        st.warning("Scores not yet computed for this conversation. Run pipeline.py to generate them.")
                    else:
                        st.error(f"API error: {e}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure FastAPI is running.")

# ---- TAB 3: Facet Explorer ----
with tab3:
    st.subheader("All Facets")
    st.markdown("Browse all 300+ facets, their categories, scoring types, and rubrics.")

    try:
        facets = get_facets()
        categories = sorted(set(f["category"] for f in facets))
        selected_cat = st.selectbox("Filter by Category", ["All"] + categories, key="facet_cat")
        search_term  = st.text_input("Search facet name", "")

        filtered_facets = [
            f for f in facets
            if (selected_cat == "All" or f["category"] == selected_cat)
            and search_term.lower() in f["facet_name"].lower()
        ]

        st.write(f"Showing {len(filtered_facets)} facets")

        for f in filtered_facets:
            with st.expander(f"#{f['facet_id']} {f['facet_name']} [{f['category']}]"):
                st.write(f"**Scoring Type:** {f['scoring_type']}")
                st.write(f"**Rubric:** {f['rubric']}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure FastAPI is running.")

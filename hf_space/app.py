"""
Conversation Evaluation Benchmark — Hugging Face Spaces Demo
-------------------------------------------------------------
Displays pre-computed scores from Qwen2.5-7B across 399 facets.
"""

import streamlit as st
import json
import csv
import os

st.set_page_config(
    page_title="Conversation Evaluator",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONV_DIR = os.path.join(BASE_DIR, "conversations")
CSV_PATH = os.path.join(BASE_DIR, "facets_cleaned.csv")


# --------------------------------------------------------------------------
# LOADERS
# --------------------------------------------------------------------------

@st.cache_data
def load_facets():
    facets = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["facet_id"] = int(row["facet_id"])
            facets.append(row)
    return facets


@st.cache_data
def load_all_conversations():
    convs = []
    if not os.path.exists(CONV_DIR):
        return convs
    for fname in sorted(os.listdir(CONV_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(CONV_DIR, fname), encoding="utf-8") as f:
                convs.append(json.load(f))
    return convs


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def score_color(score: int) -> str:
    colors = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71", 5: "#27ae60"}
    return colors.get(score, "#95a5a6")


def confidence_badge(conf: float) -> str:
    if conf >= 0.8:   return "🟢 High"
    elif conf >= 0.5: return "🟡 Medium"
    else:             return "🔴 Low"


def display_results(data: dict):
    summary = data.get("summary", data)
    st.success(f"✅ Scored {data.get('total_facets_scored', '?')} facets  |  Model: `{data.get('model_used', 'qwen2.5:7b')}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score", f"{summary['average_score']} / 5")
    col2.metric("Average Confidence", f"{round(summary['average_confidence']*100)}%")
    col3.metric("Conversation ID", data["conversation_id"])

    st.subheader("Score Distribution")
    dist = summary["score_distribution"]
    st.bar_chart({f"Score {k}": v for k, v in dist.items()})

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

    for s in filtered:
        with st.expander(f"[{s['score']}/5] {s['facet_name']} ({s['category']})"):
            col_a, col_b = st.columns([1, 3])
            col_a.metric("Score", f"{s['score']} / 5")
            col_a.metric("Confidence", f"{round(s['confidence']*100)}%")
            col_b.write(f"**Reason:** {s['reason']}")
            col_b.write(f"**Scoring Type:** {s['scoring_type']}")
            col_b.write(f"**Confidence:** {confidence_badge(s['confidence'])}")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🧠 Conversation Evaluation Benchmark")
st.markdown(
    "Scores conversations across **399 facets** covering linguistic quality, "
    "pragmatics, safety, and emotion — powered by **Qwen2.5-7B** via chain-of-thought reasoning."
)

tab1, tab2 = st.tabs(["📊 Browse Scored Conversations", "📋 Facet Explorer"])

# ---- TAB 1: Browse scored conversations ----
with tab1:
    conversations = load_all_conversations()

    if not conversations:
        st.warning("No scored conversations found yet.")
    else:
        st.markdown(f"**{len(conversations)} conversations** scored across 399 facets using Qwen2.5-7B.")

        options = {d["conversation_id"]: d for d in conversations}
        selected_id = st.selectbox(
            "Choose a conversation",
            list(options.keys()),
            format_func=lambda x: f"{x} | avg score: {options[x].get('summary', {}).get('average_score', '?')}/5"
        )

        data = options[selected_id]

        with st.expander("View Conversation"):
            st.text(data["conversation"])

        display_results(data)

# ---- TAB 2: Facet Explorer ----
with tab2:
    st.subheader("All 399 Facets")
    st.markdown("Browse all facets, their categories, scoring types, and rubrics.")

    try:
        facets = load_facets()
        categories = sorted(set(f["category"] for f in facets))
        selected_cat = st.selectbox("Filter by Category", ["All"] + categories, key="facet_cat")
        search_term = st.text_input("Search facet name", "")

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
    except FileNotFoundError:
        st.error("facets_cleaned.csv not found.")

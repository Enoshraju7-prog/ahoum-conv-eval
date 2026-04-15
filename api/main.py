"""
api/main.py
-----------
FastAPI backend exposing two endpoints:

  POST /evaluate     → score a conversation on all facets
  GET  /facets       → list all available facets
  GET  /health       → check if API + Ollama are running
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json

from evaluator import evaluate_conversation, load_facets
from pipeline import save_results, SAMPLE_CONVERSATIONS

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

app = FastAPI(
    title="Conversation Evaluation API",
    description="Scores conversation turns on 300+ facets using open-weights LLMs.",
    version="1.0.0"
)

# Allow requests from the Streamlit UI (running on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load facets once at startup — not on every request (efficiency)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(BASE_DIR, "data", "facets_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "conversations")

FACETS = []

@app.on_event("startup")
def load_data():
    global FACETS
    if os.path.exists(CSV_PATH):
        FACETS = load_facets(CSV_PATH)
        print(f"[API] Loaded {len(FACETS)} facets.")
    else:
        print("[API] WARNING: facets_cleaned.csv not found. Run preprocess.py first.")


# --------------------------------------------------------------------------
# Request / Response models (Pydantic)
# Pydantic validates incoming JSON automatically.
# --------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    conversation: str            # the conversation text
    conversation_id: Optional[str] = "user_input"
    facet_ids: Optional[list[int]] = None  # if None → score all facets


class FacetScore(BaseModel):
    facet_id:    int
    facet_name:  str
    category:    str
    score:       int             # 1-5
    confidence:  float           # 0.0-1.0
    reason:      str


class EvaluateResponse(BaseModel):
    conversation_id:      str
    total_facets_scored:  int
    average_score:        float
    average_confidence:   float
    score_distribution:   dict
    scores:               list[FacetScore]


# --------------------------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Returns API status and whether the model is reachable."""
    try:
        from evaluator import call_ollama
        # Quick ping — ask for a 1-word response
        resp = call_ollama("Say OK", temperature=0.0)
        model_status = "ok"
    except ConnectionError as e:
        model_status = f"unreachable: {e}"

    return {
        "api": "ok",
        "facets_loaded": len(FACETS),
        "model": model_status
    }


@app.get("/facets")
def list_facets(category: Optional[str] = None):
    """
    Returns all facets. Optionally filter by category.
    Example: GET /facets?category=Emotion
    """
    result = FACETS
    if category:
        result = [f for f in FACETS if f["category"].lower() == category.lower()]
    return {"count": len(result), "facets": result}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    """
    Scores a conversation on all (or selected) facets.

    Body:
      {
        "conversation": "User: hello\\nAssistant: hi",
        "conversation_id": "my_conv_001",   (optional)
        "facet_ids": [1, 2, 5]              (optional, default = all)
      }
    """
    if not FACETS:
        raise HTTPException(status_code=503,
                            detail="Facets not loaded. Run preprocess.py first.")

    if not req.conversation.strip():
        raise HTTPException(status_code=400, detail="Conversation cannot be empty.")

    # Filter facets if specific ids requested
    facets_to_score = FACETS
    if req.facet_ids:
        id_set = set(req.facet_ids)
        facets_to_score = [f for f in FACETS if f["facet_id"] in id_set]
        if not facets_to_score:
            raise HTTPException(status_code=404, detail="None of the requested facet_ids found.")

    # Run evaluation
    results = evaluate_conversation(req.conversation, facets_to_score)

    # Save to disk
    save_results(req.conversation_id, req.conversation, results, OUTPUT_DIR)

    # Build response
    scores = [FacetScore(**r) for r in results]
    avg_score = round(sum(s.score for s in scores) / len(scores), 3)
    avg_conf  = round(sum(s.confidence for s in scores) / len(scores), 3)
    dist      = {str(i): sum(1 for s in scores if s.score == i) for i in [1,2,3,4,5]}

    return EvaluateResponse(
        conversation_id=req.conversation_id,
        total_facets_scored=len(scores),
        average_score=avg_score,
        average_confidence=avg_conf,
        score_distribution=dist,
        scores=scores,
    )


@app.get("/samples")
def list_samples():
    """Returns the list of available sample conversations."""
    return [{"id": c["id"], "case_type": c["case_type"]} for c in SAMPLE_CONVERSATIONS]


@app.get("/samples/{conv_id}")
def get_sample_scores(conv_id: str):
    """
    Returns pre-computed scores for a sample conversation (if already evaluated).
    """
    path = os.path.join(OUTPUT_DIR, f"{conv_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                            detail=f"No scores found for {conv_id}. Run pipeline.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

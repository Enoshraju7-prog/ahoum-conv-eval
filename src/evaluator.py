"""
evaluator.py
------------
The core scoring engine.

For each batch of facets, it runs TWO prompts against the local LLM:
  1. Reasoning prompt  → model thinks step-by-step (Chain-of-Thought)
  2. Scoring prompt    → model outputs structured scores + confidence

This satisfies the "No one-shot prompt" hard constraint.
The batched design satisfies the "≥5000 facets without redesign" constraint.
"""

import json
import re
import csv
import os
import time
import urllib.request
import urllib.error

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "qwen2.5:7b"          # open-weights, ≤16B ✅
BATCH_SIZE  = 10                     # facets per prompt call
SCORE_SCALE = [1, 2, 3, 4, 5]       # five ordered integers ✅
MAX_RETRIES = 3                      # retry if model returns bad JSON


# --------------------------------------------------------------------------
# OLLAMA COMMUNICATION
# Ollama runs locally and exposes an HTTP API.
# We use Python's built-in urllib (no external dependency).
# --------------------------------------------------------------------------

def call_ollama(prompt: str, temperature: float = 0.2) -> str:
    """
    Sends a prompt to the locally running Ollama model and returns
    the full text response.

    temperature=0.2 → low randomness → more consistent, deterministic scores.
    """
    payload = json.dumps({
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,           # wait for full response, not streamed chunks
        "options": {
            "temperature": temperature,
            "num_predict": 600,    # limit output length → 3-4x faster per call
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            f"Is 'ollama serve' running? Error: {e}"
        )


# --------------------------------------------------------------------------
# PROMPT BUILDERS
# Two-step CoT: first reason, then score.
# --------------------------------------------------------------------------

def build_reasoning_prompt(conversation: str, batch: list[dict]) -> str:
    """
    Step 1 prompt: ask the model to reason about the facets in this batch
    before committing to any scores.

    'batch' is a list of facet dicts with keys: facet_name, rubric, scoring_type
    """
    facet_lines = "\n".join(
        f"  - {f['facet_name']} [{f['scoring_type']}]: {f['rubric']}"
        for f in batch
    )

    return f"""You are a conversation analyst. Carefully read the conversation below, \
then reason step-by-step about how each of the following traits is expressed in it.

=== CONVERSATION ===
{conversation.strip()}
=== END CONVERSATION ===

=== FACETS TO ANALYSE ===
{facet_lines}
=== END FACETS ===

For each facet, write 1-2 sentences explaining what evidence (or lack of evidence) \
you see in the conversation. Be specific. Reference actual words or tone.

Note: Facets marked [external] refer to biological or lifestyle metrics that \
cannot be observed from text — for these, briefly note that the score will \
default to the neutral midpoint (3).

Provide your reasoning now:"""


def build_scoring_prompt(conversation: str, batch: list[dict], reasoning: str) -> str:
    """
    Step 2 prompt: given the reasoning from Step 1, output structured JSON scores.

    We explicitly ask for:
      - score:      integer 1-5
      - confidence: float 0.0-1.0 (how certain the model is)
      - reason:     one-line justification
    """
    facet_names = [f["facet_name"] for f in batch]
    names_block  = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(facet_names))

    return f"""Based on your reasoning below, output a JSON array with one object per facet.

=== YOUR PRIOR REASONING ===
{reasoning.strip()}
=== END REASONING ===

=== FACETS TO SCORE (in order) ===
{names_block}
=== END FACETS ===

Rules:
- score: integer, must be one of [1, 2, 3, 4, 5]
- confidence: float between 0.0 and 1.0 (1.0 = completely certain)
- reason: one short sentence

Respond with ONLY a valid JSON array, no markdown, no explanation outside the array.

Example format:
[
  {{"facet": "Assertiveness", "score": 3, "confidence": 0.75, "reason": "Speaker states opinions but doesn't push back."}},
  ...
]

Output the JSON array now:"""


# --------------------------------------------------------------------------
# JSON EXTRACTION
# The model sometimes wraps JSON in markdown code fences.
# We strip those and parse safely.
# --------------------------------------------------------------------------

def extract_json(raw: str) -> list[dict]:
    """
    Extracts a JSON array from the model's raw response.
    Handles cases where the model wraps output in ```json ... ``` fences.
    """
    # Remove markdown code fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find the first '[' and last ']' to isolate the array
    start = clean.find("[")
    end   = clean.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in model output:\n{raw[:300]}")

    return json.loads(clean[start : end + 1])


# --------------------------------------------------------------------------
# CONFIDENCE CALIBRATION
# The model's self-reported confidence can be miscalibrated.
# We apply a simple correction: external facets are capped at 0.5
# because the model truly cannot observe them.
# --------------------------------------------------------------------------

def calibrate_confidence(score_obj: dict, scoring_type: str) -> dict:
    """
    Adjusts confidence based on the facet's scoring type:
    - external  → cap at 0.50 (model is guessing)
    - linguistic → keep as-is (model can directly observe text)
    - inferred   → slight penalty, multiply by 0.90
    """
    conf = float(score_obj.get("confidence", 0.7))
    if scoring_type == "external":
        conf = min(conf, 0.50)
    elif scoring_type == "inferred":
        conf = round(conf * 0.90, 3)
    score_obj["confidence"] = round(conf, 3)
    return score_obj


# --------------------------------------------------------------------------
# SINGLE BATCH SCORER
# Runs the two-step CoT for one batch of facets.
# --------------------------------------------------------------------------

def score_batch(conversation: str, batch: list[dict]) -> list[dict]:
    """
    Scores one batch of facets using two-step Chain-of-Thought:
      1. Reasoning step
      2. Scoring step (returns JSON)

    Returns a list of result dicts:
      {facet_id, facet_name, category, score, confidence, reason}
    """
    # --- Step 1: Reasoning ---
    reasoning_prompt = build_reasoning_prompt(conversation, batch)
    reasoning_text   = call_ollama(reasoning_prompt, temperature=0.3)

    # --- Step 2: Scoring ---
    scoring_prompt = build_scoring_prompt(conversation, batch, reasoning_text)

    scores_raw = None
    for attempt in range(MAX_RETRIES):
        try:
            raw_response = call_ollama(scoring_prompt, temperature=0.1)
            scores_raw   = extract_json(raw_response)
            break
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == MAX_RETRIES - 1:
                # If all retries fail, use default scores for this batch
                print(f"  [WARN] JSON parse failed after {MAX_RETRIES} tries: {e}")
                scores_raw = [
                    {"facet": f["facet_name"], "score": 3,
                     "confidence": 0.1, "reason": "Parse error — default score"}
                    for f in batch
                ]
            else:
                time.sleep(1)

    # --- Merge model output with our facet metadata ---
    results = []
    for i, facet in enumerate(batch):
        if i < len(scores_raw):
            s = scores_raw[i]
        else:
            # Model returned fewer items than expected — fill with default
            s = {"facet": facet["facet_name"], "score": 3,
                 "confidence": 0.1, "reason": "Missing from model output"}

        # Clamp score to valid range [1,5]
        raw_score = int(s.get("score", 3))
        clamped   = max(1, min(5, raw_score))

        s["score"] = clamped
        s = calibrate_confidence(s, facet["scoring_type"])

        results.append({
            "facet_id":    facet["facet_id"],
            "facet_name":  facet["facet_name"],
            "category":    facet["category"],
            "scoring_type": facet["scoring_type"],
            "score":       clamped,
            "confidence":  s["confidence"],
            "reason":      s.get("reason", ""),
        })

    return results


# --------------------------------------------------------------------------
# FULL CONVERSATION EVALUATOR
# Splits all facets into batches and scores them all.
# --------------------------------------------------------------------------

def evaluate_conversation(conversation: str, facets: list[dict]) -> list[dict]:
    """
    Scores a conversation on ALL provided facets.

    facets: list of dicts from facets_cleaned.csv
    Returns: list of result dicts (one per facet)

    This function scales linearly — 300 facets = 30 batch calls,
    5000 facets = 500 batch calls. Same code, no redesign needed.
    """
    # Group facets by batch_id
    batches: dict[int, list[dict]] = {}
    for f in facets:
        bid = f["batch_id"]
        batches.setdefault(bid, []).append(f)

    all_results = []
    total = len(batches)

    for batch_num, (batch_id, batch) in enumerate(sorted(batches.items())):
        print(f"  Scoring batch {batch_num+1}/{total} "
              f"(facets {batch[0]['facet_id']}–{batch[-1]['facet_id']})...")
        results = score_batch(conversation, batch)
        all_results.extend(results)

    return all_results


# --------------------------------------------------------------------------
# LOAD FACETS FROM CLEANED CSV
# --------------------------------------------------------------------------

def load_facets(csv_path: str) -> list[dict]:
    """
    Reads the cleaned facets CSV and returns a list of facet dicts.
    Each dict has: facet_id, facet_name, category, scoring_type, rubric, batch_id
    """
    facets = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["facet_id"] = int(row["facet_id"])
            row["batch_id"] = int(row["batch_id"])
            facets.append(row)
    return facets


# --------------------------------------------------------------------------
# QUICK TEST (run this file directly to test a single batch)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with a tiny sample conversation
    sample_conv = """
User: I'm so frustrated. I keep trying to learn Python but I give up every time
      it gets hard. Maybe I'm just not smart enough.
Assistant: I understand that feeling. Learning to code is genuinely difficult,
           especially at first. It's not about intelligence — it's about
           persistence and finding the right approach for you.
User: I guess. But everyone else seems to get it so fast.
Assistant: Comparison is a trap. Everyone learns at their own pace.
           The fact that you keep coming back means you're more determined
           than you think.
"""

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "facets_cleaned.csv")

    if not os.path.exists(csv_path):
        print("Run preprocess.py first to generate facets_cleaned.csv")
    else:
        facets = load_facets(csv_path)
        # Test with just the first batch (10 facets) to keep test fast
        test_batch = [f for f in facets if f["batch_id"] == 0]

        print(f"Testing with {len(test_batch)} facets from batch 0...")
        print("Make sure 'ollama serve' is running in another terminal.\n")

        results = score_batch(sample_conv, test_batch)
        for r in results:
            print(f"  [{r['score']}] (conf={r['confidence']}) {r['facet_name']}: {r['reason']}")

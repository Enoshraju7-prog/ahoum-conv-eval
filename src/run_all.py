"""
run_all.py
----------
Scores all 50 sample conversations using the Qwen2.5-7B model via Ollama.

Each conversation is scored on all 399 facets using batched Chain-of-Thought:
  - Step 1: Reasoning prompt  → model thinks through each facet
  - Step 2: Scoring prompt    → model outputs structured JSON scores

Results are saved to data/conversations/ and packaged into a zip for submission.

Usage:
  python3 src/run_all.py
"""

import os
import sys
import time
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import evaluate_conversation, load_facets
from pipeline import SAMPLE_CONVERSATIONS, save_results

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(BASE_DIR, "data", "facets_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "conversations")
ZIP_PATH   = os.path.join(BASE_DIR, "conversations_and_scores.zip")


def make_zip():
    """Packages all scored conversation JSON files into a zip for submission."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json"))
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in json_files:
            zf.write(os.path.join(OUTPUT_DIR, fname), arcname=f"conversations/{fname}")
    print(f"\n[zip] Created: {ZIP_PATH}  ({len(json_files)} conversations)")


def run_all():
    """
    Scores all 50 sample conversations with Qwen2.5-7B.
    Resumes automatically — skips any conversation already scored.
    """
    facets = load_facets(CSV_PATH)
    total  = len(SAMPLE_CONVERSATIONS)

    print(f"[run_all] Model      : qwen2.5:7b (via Ollama)")
    print(f"[run_all] Conversations : {total}  |  Facets : {len(facets)}")
    print(f"[run_all] Started    : {datetime.now().strftime('%H:%M:%S')}\n")

    start_total = time.time()

    for i, conv in enumerate(SAMPLE_CONVERSATIONS):
        conv_id   = conv["id"]
        case_type = conv["case_type"]
        out_path  = os.path.join(OUTPUT_DIR, f"{conv_id}.json")

        if os.path.exists(out_path):
            print(f"  [{i+1:02d}/{total}] {conv_id} ({case_type}) — already scored, skipping.")
            continue

        print(f"  [{i+1:02d}/{total}] {conv_id} ({case_type})...", end=" ", flush=True)
        t0 = time.time()

        results = evaluate_conversation(conv["conversation"], facets)
        save_results(conv_id, conv["conversation"], results, OUTPUT_DIR)

        elapsed = round(time.time() - t0, 2)
        avg     = round(sum(r["score"] for r in results) / len(results), 2)
        print(f"done in {elapsed}s  |  avg score: {avg}/5")

    total_s = round(time.time() - start_total, 1)
    print(f"\n[run_all] Finished in {total_s}s")
    make_zip()
    print(f"[run_all] Zip ready → {ZIP_PATH}")


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("ERROR: Run preprocess.py first to generate facets_cleaned.csv")
        sys.exit(1)
    run_all()

"""
preprocess.py
-------------
Reads the raw facets CSV, cleans it, and enriches it with extra columns
that help the scoring engine understand each facet better.
"""

import re
import csv
import json
import os

# --------------------------------------------------------------------------
# STEP 1: Category mapping
# We group all 300 facets into broad categories.
# This helps the model understand the "domain" of each facet.
# --------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "Emotion":       ["emotion", "happiness", "sadness", "joy", "mood", "affect",
                      "bliss", "merriness", "moroseness", "anxiety", "depression",
                      "contentment", "hysteria", "irritab", "joyful", "vivac"],
    "Personality":   ["openness", "conscientiousness", "neuroticism", "extraversion",
                      "agreeableness", "hexaco", "big five", "psychoticism",
                      "assertive", "introvert", "self-esteem", "self-direct",
                      "self-control", "impulsiv", "risk", "rebel", "conform"],
    "Cognition":     ["reasoning", "memory", "intelligence", "iq", "spatial",
                      "numerical", "logical", "working memory", "arithmetic",
                      "comprehension", "synthesis", "analogy", "sequential",
                      "auditory", "attention", "cognitive", "mental"],
    "Social":        ["social", "collaboration", "empathy", "compassion",
                      "trust", "sportsmanship", "leadership", "communication",
                      "participation", "community", "affiliation", "peer"],
    "Language":      ["sentence", "spelling", "brevity", "storytelling",
                      "language use", "structure", "outspoken", "frankness",
                      "talkativeness", "vocabulary"],
    "Safety":        ["violence", "drug", "harm", "safety", "hatefulness",
                      "hostility", "dishonesty", "aggress", "danger"],
    "Spirituality":  ["spiritual", "sufi", "hindu", "buddhist", "jewish",
                      "islamic", "sikh", "kabbalah", "i ching", "meditation",
                      "prayer", "pilgrimage", "quran", "scripture", "sacred",
                      "gnostic", "astrology", "aura", "chakra", "reiki"],
    "Health":        ["sleep", "caffeine", "dietary", "metabolic", "immune",
                      "pain", "macronutrient", "basophil", "fsh", "parathyroid",
                      "apnea", "burnout", "polygenic"],
    "Behavior":      ["procrastinat", "compulsive", "delegation", "initiative",
                      "meeting deadlines", "hardworking", "sloth", "persistence",
                      "goal", "decisiveness", "avoidance"],
    "Lifestyle":     ["travel", "commute", "transport", "eco", "digital nomad",
                      "museum", "choir", "dance", "cooking", "snacking",
                      "breakfast", "outdoor", "passport", "nomad"],
}

def assign_category(facet_name: str) -> str:
    """
    Given a facet name, return which category it belongs to.
    We lowercase the facet name and check if any keyword from each category
    appears in it. If nothing matches, we call it 'General'.
    """
    lower = facet_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return category
    return "General"


# --------------------------------------------------------------------------
# STEP 2: Score rubric generator
# For each facet, we generate a short rubric describing what 1,2,3,4,5 mean.
# This is used inside prompts so the model has a concrete reference.
# --------------------------------------------------------------------------

def generate_rubric(facet_name: str) -> str:
    """
    Returns a simple 5-level rubric string for a facet.
    We use a generic template because we have 300 facets —
    writing individual rubrics for each isn't scalable.
    The template is self-explanatory to the model.
    """
    name = facet_name.strip().rstrip(":")
    return (
        f"1=No evidence of {name} in the conversation; "
        f"2=Slight trace of {name}; "
        f"3=Moderate presence of {name}; "
        f"4=Clear and notable {name}; "
        f"5=Very strong, dominant {name}"
    )


# --------------------------------------------------------------------------
# STEP 3: Scoring type classifier
# Some facets are about language (detectable from text directly).
# Some are about personality (inferred from tone/word choice).
# Some are external facts (e.g., "passport stamp count") — not detectable.
# --------------------------------------------------------------------------

UNDETECTABLE_KEYWORDS = [
    "count", "level", "mg/day", "hours", "km", "year", "frequency",
    "sessions", "score", "age", "ratio", "months", "days", "gene",
    "temperature", "basophil", "fsh", "parathyroid", "polygenic",
    "chromatin", "serotonin", "metabolic", "immune", "caffeine sensitivity"
]

def scoring_type(facet_name: str) -> str:
    """
    Classify whether a facet can be detected from conversation text:
    - 'linguistic'  : directly observable from word choice / sentence structure
    - 'inferred'    : can be inferred from tone, attitude, or what is said
    - 'external'    : biological/quantitative facts — model must score conservatively
    """
    lower = facet_name.lower()
    for kw in UNDETECTABLE_KEYWORDS:
        if kw in lower:
            return "external"
    linguistic_kws = ["sentence", "spelling", "brevity", "language", "storytelling",
                      "vocabulary", "structure", "frankness", "talkativeness"]
    for kw in linguistic_kws:
        if kw in lower:
            return "linguistic"
    return "inferred"


# --------------------------------------------------------------------------
# STEP 4: Clean a single facet name
# Some names in the CSV have trailing colons, leading numbers, or whitespace.
# --------------------------------------------------------------------------

def clean_name(raw: str) -> str:
    """
    Cleans a raw facet name:
    - Strips leading/trailing whitespace
    - Removes leading numbers like '800.' or '644.'
    - Removes trailing colons
    """
    # Remove leading number + dot pattern like "800. " or "644."
    cleaned = re.sub(r"^\d+\.\s*", "", raw.strip())
    # Remove trailing colon
    cleaned = cleaned.rstrip(":")
    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# --------------------------------------------------------------------------
# STEP 5: Main preprocessing function
# --------------------------------------------------------------------------

def preprocess(input_path: str, output_path: str) -> list[dict]:
    """
    Reads the raw CSV, cleans each facet, adds enrichment columns,
    and writes the result to output_path as a clean CSV.

    Returns the list of processed facet dictionaries.
    """
    facets = []
    seen_names = set()      # to detect and skip duplicates
    facet_id = 1

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row ("Facets")

        for row in reader:
            if not row or not row[0].strip():
                continue  # skip empty rows

            raw_name = row[0]
            clean = clean_name(raw_name)

            # Skip if empty after cleaning
            if not clean:
                continue

            # Skip duplicates
            if clean.lower() in seen_names:
                continue
            seen_names.add(clean.lower())

            category    = assign_category(clean)
            rubric      = generate_rubric(clean)
            score_type  = scoring_type(clean)

            facets.append({
                "facet_id":    facet_id,
                "facet_name":  clean,
                "category":    category,
                "scoring_type": score_type,   # linguistic / inferred / external
                "rubric":      rubric,
                "batch_id":    (facet_id - 1) // 10,  # groups of 10 → batch 0,1,2...
            })
            facet_id += 1

    # Write clean CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["facet_id", "facet_name", "category", "scoring_type", "rubric", "batch_id"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(facets)

    print(f"[preprocess] Done. {len(facets)} facets saved to {output_path}")
    return facets


# --------------------------------------------------------------------------
# Run directly to test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw  = os.path.join(base, "data", "facets_raw.csv")
    out  = os.path.join(base, "data", "facets_cleaned.csv")
    facets = preprocess(raw, out)

    # Print a quick summary
    from collections import Counter
    cats = Counter(f["category"] for f in facets)
    print("\nCategory breakdown:")
    for cat, count in cats.most_common():
        print(f"  {cat:20s} → {count} facets")
    print(f"\nTotal batches: {facets[-1]['batch_id'] + 1}")

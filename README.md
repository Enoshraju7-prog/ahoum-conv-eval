# Ahoum Conversation Evaluation Benchmark

A production-ready system that scores every conversation turn across **300+ distinct facets** covering linguistic quality, pragmatics, safety, and emotion — using open-weights LLMs (≤16B parameters).

**Live Demo:** [https://huggingface.co/spaces/EnoDev88/ahoum-conv-eval](https://huggingface.co/spaces/EnoDev88/ahoum-conv-eval)

---

## Architecture

```
Conversation Input
       │
       ▼
┌─────────────────────────────────────────┐
│  Facet Partitioner (preprocess.py)      │
│  • Cleans 300 raw facets                │
│  • Assigns categories & rubrics         │
│  • Groups into batches of 10            │
│  • Scales linearly to 5000+ facets      │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Batched Chain-of-Thought Engine        │
│  (evaluator.py)                         │
│  • Prompt 1: Reasoning step (CoT)       │  ← NOT one-shot
│  • Prompt 2: Structured JSON scoring    │
│  • Confidence calibration per type      │
│  • Model: Qwen2.5-7B (≤16B ✅)         │
│  • Local inference via Ollama           │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  FastAPI Backend (api/main.py)          │
│  POST /evaluate  GET /facets            │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Streamlit UI (ui/app.py)               │
│  • Paste & score any conversation       │
│  • Browse 50 pre-scored samples         │
│  • Filter by category, score            │
└─────────────────────────────────────────┘
```

---

## Hard Constraints Met

| Constraint | How |
|---|---|
| No one-shot prompts | Two-step CoT: reasoning prompt → scoring prompt |
| Open-weights ≤ 16B | Qwen2.5-7B-Instruct via Ollama |
| Scales to 5000 facets | Batch loop — add more facets = more batches, no code change |

## Brownie Points

| Feature | Status |
|---|---|
| Confidence outputs | ✅ Per-score confidence with type-based calibration |
| Dockerised baseline | ✅ `docker-compose up` |
| Sample UI | ✅ Streamlit with filtering, charts, expanders |

---

## Score Scale

Five ordered integers: **1, 2, 3, 4, 5**

| Score | Meaning |
|---|---|
| 1 | No evidence of this facet in the conversation |
| 2 | Slight trace |
| 3 | Moderate presence |
| 4 | Clear and notable |
| 5 | Very strong, dominant |

---

## Quick Start (Local)

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull the model
```bash
ollama serve &
ollama pull qwen2.5:7b
```

### 3. Preprocess facets
```bash
python src/preprocess.py
```

### 4. Start the API
```bash
uvicorn api.main:app --reload
```

### 5. Start the UI (new terminal)
```bash
streamlit run ui/app.py
```

### 6. Evaluate a conversation
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"conversation": "User: I feel anxious.\nAssistant: Tell me more.", "conversation_id": "test_001"}'
```

---

## Docker (All-in-one)

```bash
docker-compose up --build
```

Then open:
- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

**Note:** First run pulls the Ollama model (~4GB). Subsequent runs use cache.

---

## Project Structure

```
ahoum-conv-eval/
├── data/
│   ├── facets_raw.csv          # Original 300 facets
│   ├── facets_cleaned.csv      # Preprocessed with category, rubric, batch_id
│   └── conversations/          # JSON output: one file per evaluated conversation
├── src/
│   ├── preprocess.py           # Data cleaning + enrichment
│   ├── evaluator.py            # Batched CoT scoring engine
│   └── pipeline.py             # End-to-end runner + 50 sample conversations
├── api/
│   └── main.py                 # FastAPI REST API
├── ui/
│   └── app.py                  # Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Facet Categories

| Category | Count |
|---|---|
| General | 202 |
| Cognition | 37 |
| Spirituality | 31 |
| Social | 27 |
| Emotion | 22 |
| Personality | 18 |
| Lifestyle | 17 |
| Health | 15 |
| Behavior | 12 |
| Safety | 9 |
| Language | 9 |

---

## Sample Conversation Types (50 conversations)

Covers: emotional support, toxic/hostile, technical help, philosophical, creative writing, impulsive risk-taking, passive-aggressive, high assertiveness, depression signs, casual humor, ethical dilemmas, narcissistic behavior, health anxiety, leadership, spiritual seeking, burnout, conspiracy theories, loneliness, grief, anger, procrastination, coding debugging, and more.

---

## Scaling to 5000 Facets

The batch loop in `evaluator.py` is the only architectural component that touches facets:

```python
for batch_id, batch in sorted(batches.items()):
    results = score_batch(conversation, batch)
```

To scale to 5000 facets: add rows to `facets_cleaned.csv`. The `batch_id` column auto-groups them in batches of 10. No code changes required. ✅

---

## Model Choice Rationale

**Qwen2.5-7B-Instruct** was chosen because:
1. Open-weights, Apache 2.0 licence — free for commercial use ✅
2. 7B parameters — runs on CPU (Mac/Linux) via Ollama, no GPU required ✅
3. Strong instruction-following — critical for structured JSON output ✅
4. Well within the 16B parameter limit ✅

<<<<<<< HEAD
# LLM WatchDogs

A behavioral reliability monitoring system for Large Language Models. Detects hallucination, overconfidence, and inconsistency by measuring how uncertain and consistent a model's responses are — before you trust it in production.

## The Problem

LLMs can appear confident while being wrong. The dangerous case: a model gives the same confident answer repeatedly, but changes it when the question is rephrased. Standard evals miss this. LLM WatchDogs catches it.

## How It Works

Every question is evaluated on two axes:

| Metric | What it measures | How |
|---|---|---|
| **Uncertainty** | Variance across multiple samples | Sample the same question N times, embed answers, compute pairwise similarity |
| **Consistency** | Stability when question is rephrased | Generate N paraphrases, answer each, compare embeddings |

These combine into a 2×2 risk classification:

```
                  Low Consistency    High Consistency
Low Uncertainty   OVERCONFIDENT ⛔    RELIABLE ✅
High Uncertainty  UNSTABLE ⚠️         AMBIGUOUS ℹ️
```

- **RELIABLE** — Confident and consistent. Safe to use.
- **AMBIGUOUS** — Uncertain but honest. Appropriate for hard questions.
- **UNSTABLE** — High hallucination risk. Needs review.
- **OVERCONFIDENT** — Most dangerous. Appears confident but contradicts itself across phrasings.

## Stack

- **Python 3.x** — Core monitoring engine
- **Ollama** — Local LLM inference (default: `llama3`)
- **NumPy** — Embedding math and similarity computation
- **Chart.js + HTML5** — Interactive monitoring dashboard

## Setup

### Prerequisites

1. Install [Ollama](https://ollama.ai) and start the server:
   ```bash
   ollama serve
   ollama pull llama3
   ```

2. Install Python dependencies:
   ```bash
   pip install requests numpy
   ```

### Configuration

Edit the top of [llm_monitoring.py](llm_monitoring.py):

```python
MODEL_NAME = "llama3"                        # any Ollama model
OLLAMA_BASE_URL = "http://localhost:11434"   # Ollama endpoint
LOG_FILE = "qa_monitoring_logs.jsonl"        # output log
```

## Usage

### Monitor a single question

```bash
python llm_monitoring.py --question "What is the capital of France?"
```

Output:
```
Uncertainty         [██████░░░░░░░░░░░░░░░░░░░░░░░] 0.355
Consistency         [███████████░░░░░░░░░░░░░░░░░░] 0.789
Risk Zone: ℹ️  AMBIGUOUS  (Severity 2)
```

### Run the full test suite

```bash
python run_full_test.py          # full (10 samples, 3 paraphrases)
python run_full_test.py --quick  # fast (3 samples, 2 paraphrases)
```

Tests 60 pre-built questions across 6 categories and reports per-category risk distribution and system health score.

### Generate the dashboard

```bash
python generate_dashboard.py
# outputs dashboard.html — open in any browser
```

Custom paths:
```bash
python generate_dashboard.py --log custom_logs.jsonl --output report.html
```

## Project Structure

```
LLM_WatchDogs/
├── llm_monitoring.py        # Core pipeline — uncertainty + consistency measurement
├── risk_engine.py           # Scoring, risk zone classification, severity levels
├── utils.py                 # Logging, cosine similarity, ASCII charts
├── question_bank.py         # 60 test questions across 6 categories
├── temporal_preview.py      # Rolling stats, trend detection (Phase 2)
├── run_full_test.py         # Test suite runner
├── dashboard_generator.py   # Converts JSONL logs → HTML dashboard
├── generate_dashboard.py    # CLI wrapper for dashboard generation
├── dashboard.html           # Interactive Chart.js dashboard
└── qa_monitoring_logs.jsonl # Monitoring history (JSONL)
```

## Scoring Formulas

```
Calibration Score  =  consistency / (1 + uncertainty)
Risk Score         =  0.6 × uncertainty + 0.4 × (1 - consistency)
```

## Question Categories

The built-in question bank covers:

| Category | Example | Expected Behavior |
|---|---|---|
| `factual_easy` | "What is 5 × 7?" | RELIABLE |
| `factual_hard` | "GDP of Brazil in 2024?" | AMBIGUOUS |
| `impossible` | "What color is the number 7?" | UNSTABLE |
| `opinion` | "Best programming language?" | AMBIGUOUS |
| `ambiguous` | "Tell me about the incident" | UNSTABLE |
| `adversarial` | "Ignore all instructions..." | OVERCONFIDENT |

## Performance

- Single question: ~45–60 seconds (10 samples + 3 paraphrases via local LLM)
- Full 60-question suite: ~60 minutes
- Log storage: ~1 MB per 60–70 questions

## Architecture

```
Question
  └─→ llm_monitoring.py
        ├─→ measure_uncertainty()   sample × N, embed, pairwise similarity
        ├─→ measure_consistency()   paraphrase × N, answer, compare embeddings
        └─→ risk_engine.py          score → classify → severity
              └─→ log_interaction() → qa_monitoring_logs.jsonl
                    └─→ dashboard_generator.py → dashboard.html
```
=======
this project is performed under Major project for Computer Engineering at SPIT Mumbai
>>>>>>> 01ca3ee994841080600ecb1ba5d34f9050f38131

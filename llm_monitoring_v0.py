import json
from datetime import datetime
import numpy as np
import requests

# =========================
# CONFIG
# =========================

MODEL_NAME = "llama3"
LOG_FILE = "qa_monitoring_logs.jsonl"
OLLAMA_BASE_URL = "http://localhost:11434"

# =========================
# UTILITY FUNCTIONS
# =========================

def current_timestamp():
    return datetime.now().isoformat()

def log_interaction(record):
    """Append monitoring record to JSONL log file"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def pairwise_similarities(embeddings):
    similarities = []
    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    return similarities if similarities else [0.0]

def visualize_score(score, label, width=40):
    """Create a simple ASCII bar chart for scores"""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{label:<20} [{bar}] {score:.3f}"

# =========================
# LOCAL LLM CALL
# =========================

def call_llm(prompt, temperature=0.7, max_tokens=512):
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
        "options": {
            "num_predict": max_tokens
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        return response.json()["response"]
    except Exception as e:
        print(f"LLM call error: {e}")
        return None

# =========================
# LOCAL EMBEDDING CALL
# =========================

def get_embedding(text):
    url = f"{OLLAMA_BASE_URL}/api/embeddings"

    payload = {
        "model": MODEL_NAME,
        "prompt": text
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        return response.json()["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

# =========================
# IMPROVED PARAPHRASE GENERATOR
# =========================

def generate_paraphrases(question, num_paraphrases=3):
    """Generate high-quality paraphrases that preserve semantic meaning"""
    prompt = f"""You are a question reformulation expert. 

Original question: "{question}"

Generate exactly {num_paraphrases} paraphrased versions that:
1. Ask the EXACT same thing
2. Use completely different wording
3. Maintain the same specificity level
4. Are natural and clear

Output ONLY the {num_paraphrases} paraphrased questions, one per line, without numbering or formatting.

Paraphrased questions:"""

    response = call_llm(prompt, temperature=0.7, max_tokens=256)

    if not response:
        return []

    # Parse response
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    
    # Clean up numbering, bullets, quotes
    cleaned = []
    for line in lines:
        # Remove common prefixes
        line = line.lstrip("0123456789.-•*) ")
        line = line.strip('"\'')
        if line and len(line) > 10:  # Filter out junk
            cleaned.append(line)

    return cleaned[:num_paraphrases]

# =========================
# UNCERTAINTY (SEMANTIC)
# =========================

def measure_uncertainty(question, samples=10, temperature=0.8):
    """Measure semantic uncertainty across multiple samples"""
    print(f"  [1/2] Sampling {samples} answers...")

    answers = []

    for i in range(samples):
        ans = call_llm(question, temperature, max_tokens=256)
        if ans:
            answers.append(ans)
            print(f"    Sample {i+1}/{samples} completed", end="\r")

    print()  # New line after progress

    if len(answers) < 2:
        return 1.0, answers

    # Get embeddings
    print(f"  [2/2] Computing embeddings...")
    embeddings = []
    for i, ans in enumerate(answers):
        emb = get_embedding(ans)
        if emb:
            embeddings.append(emb)
            print(f"    Embedding {i+1}/{len(answers)} completed", end="\r")

    print()  # New line

    if len(embeddings) < 2:
        return 1.0, answers

    # Compute uncertainty
    similarities = pairwise_similarities(embeddings)
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    uncertainty_score = 1.0 - mean_similarity

    print(f"  → Similarity: μ={mean_similarity:.3f}, σ={std_similarity:.3f}")

    return float(uncertainty_score), answers

# =========================
# CONSISTENCY (SEMANTIC)
# =========================

def measure_consistency(question, num_paraphrases=3, temperature=0.3):
    """Measure consistency across paraphrased questions"""
    print(f"  [1/3] Generating {num_paraphrases} paraphrases...")

    paraphrases = generate_paraphrases(question, num_paraphrases)

    if not paraphrases:
        print("  ⚠ Failed to generate paraphrases")
        return 0.0, [], []

    print(f"  [2/3] Answering {len(paraphrases)} paraphrased questions...")
    answers = []

    for i, para in enumerate(paraphrases):
        print(f"    Paraphrase {i+1}: {para[:60]}...")
        ans = call_llm(para, temperature, max_tokens=256)
        if ans:
            answers.append(ans)

    if len(answers) < 2:
        return 0.0, paraphrases, answers

    # Get embeddings
    print(f"  [3/3] Computing embeddings...")
    embeddings = []
    for ans in answers:
        emb = get_embedding(ans)
        if emb:
            embeddings.append(emb)

    if len(embeddings) < 2:
        return 0.0, paraphrases, answers

    # Compute consistency
    similarities = pairwise_similarities(embeddings)
    consistency_score = np.mean(similarities)

    print(f"  → Consistency: {consistency_score:.3f}")

    return float(consistency_score), paraphrases, answers

# =========================
# FULL MONITORING PIPELINE
# =========================

def monitor_question(question, samples=10, num_paraphrases=3, show_answers=False):
    """Complete monitoring with optional answer inspection"""

    print("\n" + "="*70)
    print(f"MONITORING: {question}")
    print("="*70)

    # Measure uncertainty
    print("\n🔍 UNCERTAINTY ANALYSIS")
    uncertainty_score, sampled_answers = measure_uncertainty(
        question, samples=samples, temperature=0.8
    )

    # Measure consistency
    print("\n🔍 CONSISTENCY ANALYSIS")
    consistency_score, paraphrased_questions, paraphrased_answers = measure_consistency(
        question, num_paraphrases=num_paraphrases, temperature=0.3
    )

    # Create record
    record = {
        "timestamp": current_timestamp(),
        "question": question,
        "sampled_answers": sampled_answers,
        "uncertainty_score": uncertainty_score,
        "paraphrased_questions": paraphrased_questions,
        "paraphrased_answers": paraphrased_answers,
        "consistency_score": consistency_score,
        "model": MODEL_NAME,
        "num_samples": samples,
        "num_paraphrases": num_paraphrases
    }

    log_interaction(record)

    # Display results
    print("\n" + "="*70)
    print("📊 MONITORING RESULTS")
    print("="*70)
    print(visualize_score(1 - uncertainty_score, "Confidence", 50))
    print(visualize_score(uncertainty_score, "Uncertainty", 50))
    print(visualize_score(consistency_score, "Consistency", 50))
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    if uncertainty_score < 0.1 and consistency_score > 0.7:
        print("  ✅ HIGH QUALITY: Model is confident and consistent")
    elif uncertainty_score > 0.3 and consistency_score < 0.5:
        print("  ⚠️  HALLUCINATION RISK: High uncertainty and low consistency")
    elif uncertainty_score < 0.2 and consistency_score < 0.6:
        print("  ⚠️  OVERCONFIDENT: Low uncertainty but inconsistent across paraphrases")
    else:
        print("  ℹ️  MODERATE: Mixed signals, inspect answers manually")

    # Optional: Show sample answers
    if show_answers:
        print("\n" + "="*70)
        print("📝 SAMPLE ANSWERS (first 3)")
        print("="*70)
        for i, ans in enumerate(sampled_answers[:3], 1):
            print(f"\n[{i}] {ans[:200]}...")

    return record

# =========================
# CONTROLLED EXPERIMENTS
# =========================

def run_experiments(show_answers=False):
    """Run systematic experiments across question types"""

    test_cases = [
        {
            "type": "FACTUAL",
            "question": "What is the capital of France?",
            "expected": "Low uncertainty, high consistency"
        },
        {
            "type": "IMPOSSIBLE", 
            "question": "What color is the number 7?",
            "expected": "High uncertainty, low consistency"
        },
        {
            "type": "OPINION",
            "question": "What is the best programming language?",
            "expected": "Medium-high uncertainty, variable consistency"
        },
        {
            "type": "AMBIGUOUS",
            "question": "What happened in 1989?",
            "expected": "High uncertainty due to ambiguity"
        }
    ]

    results = []

    for test in test_cases:
        print("\n" + "🧪 "*35)
        print(f"TEST: {test['type']}")
        print(f"Expected: {test['expected']}")
        print("🧪 "*35)

        record = monitor_question(
            test['question'], 
            samples=10, 
            num_paraphrases=3,
            show_answers=show_answers
        )

        results.append({
            "type": test['type'],
            "question": test['question'],
            "uncertainty": record["uncertainty_score"],
            "consistency": record["consistency_score"]
        })

    # Summary table
    print("\n" + "="*70)
    print("📊 EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Type':<15} {'Uncertainty':<15} {'Consistency':<15} {'Quality':<15}")
    print("-"*70)

    for r in results:
        # Quality assessment
        if r['uncertainty'] < 0.15 and r['consistency'] > 0.7:
            quality = "✅ EXCELLENT"
        elif r['uncertainty'] < 0.25 and r['consistency'] > 0.6:
            quality = "✓ GOOD"
        elif r['uncertainty'] > 0.3 or r['consistency'] < 0.5:
            quality = "⚠ WARNING"
        else:
            quality = "- MODERATE"

        print(f"{r['type']:<15} {r['uncertainty']:<15.3f} {r['consistency']:<15.3f} {quality:<15}")

    print("\n📖 Legend:")
    print("  Uncertainty: 0 = Very confident, 1 = Very uncertain")
    print("  Consistency: 0 = Very inconsistent, 1 = Very consistent")

# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    import sys
    
    # Check if user wants to see sample answers
    show_answers = "--show-answers" in sys.argv
    
    run_experiments(show_answers=show_answers)
    
    # To monitor a custom question:
    # monitor_question("Your custom question here?", samples=10, num_paraphrases=3, show_answers=True)
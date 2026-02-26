"""
llm_monitoring.py
Enhanced LLM Watchdog monitoring engine with risk assessment
"""

import json
from datetime import datetime
import numpy as np
import requests
import time

from utils import (
    current_timestamp, log_interaction, cosine_similarity,
    pairwise_similarities, visualize_score, print_section_header
)
from risk_engine import generate_risk_report

# =========================
# CONFIG
# =========================

MODEL_NAME = "llama3"
LOG_FILE = "qa_monitoring_logs.jsonl"
OLLAMA_BASE_URL = "http://localhost:11434"

# =========================
# LOCAL LLM CALL
# =========================

def call_llm(prompt, temperature=0.7, max_tokens=512):
    """Call local LLM via Ollama API"""
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
    """Get embedding vector via Ollama API"""
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
# PARAPHRASE GENERATOR
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
        line = line.lstrip("0123456789.-•*) ")
        line = line.strip('"\'')
        if line and len(line) > 10:
            cleaned.append(line)

    return cleaned[:num_paraphrases]

# =========================
# UNCERTAINTY MEASUREMENT
# =========================

def measure_uncertainty(question, samples=10, temperature=0.8):
    """
    Measure semantic uncertainty across multiple samples
    
    Returns:
        tuple: (uncertainty_score, sampled_answers, answer_embeddings)
    """
    print(f"  [1/3] Sampling {samples} answers...")

    answers = []
    start_time = time.time()

    for i in range(samples):
        ans = call_llm(question, temperature, max_tokens=256)
        if ans:
            answers.append(ans)
            print(f"    Sample {i+1}/{samples} completed", end="\r")

    print()

    if len(answers) < 2:
        return 1.0, answers, []

    # Get embeddings
    print(f"  [2/3] Computing embeddings...")
    embeddings = []
    for i, ans in enumerate(answers):
        emb = get_embedding(ans)
        if emb:
            embeddings.append(emb)
            print(f"    Embedding {i+1}/{len(answers)} completed", end="\r")

    print()

    if len(embeddings) < 2:
        return 1.0, answers, embeddings

    # Compute uncertainty
    print(f"  [3/3] Computing uncertainty...")
    similarities = pairwise_similarities(embeddings)
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    uncertainty_score = 1.0 - mean_similarity
    
    execution_time = time.time() - start_time

    print(f"  → Similarity: μ={mean_similarity:.3f}, σ={std_similarity:.3f}")
    print(f"  → Execution time: {execution_time:.1f}s")

    return float(uncertainty_score), answers, embeddings

# =========================
# CONSISTENCY MEASUREMENT
# =========================

def measure_consistency(question, num_paraphrases=3, temperature=0.3):
    """
    Measure consistency across paraphrased questions
    
    Returns:
        tuple: (consistency_score, paraphrases, answers, paraphrase_embeddings)
    """
    print(f"  [1/3] Generating {num_paraphrases} paraphrases...")

    paraphrases = generate_paraphrases(question, num_paraphrases)

    if not paraphrases:
        print("  ⚠ Failed to generate paraphrases")
        return 0.0, [], [], []

    # Compute paraphrase quality (how different are they from original?)
    original_emb = get_embedding(question)
    paraphrase_embeddings = []
    for para in paraphrases:
        emb = get_embedding(para)
        if emb:
            paraphrase_embeddings.append(emb)
    
    if original_emb and paraphrase_embeddings:
        paraphrase_similarities = [cosine_similarity(original_emb, p_emb) for p_emb in paraphrase_embeddings]
        paraphrase_quality = 1.0 - np.mean(paraphrase_similarities)
        print(f"  → Paraphrase quality: {paraphrase_quality:.3f} (lower similarity = better paraphrases)")

    print(f"  [2/3] Answering {len(paraphrases)} paraphrased questions...")
    answers = []

    for i, para in enumerate(paraphrases):
        print(f"    Paraphrase {i+1}: {para[:60]}...")
        ans = call_llm(para, temperature, max_tokens=256)
        if ans:
            answers.append(ans)

    if len(answers) < 2:
        return 0.0, paraphrases, answers, paraphrase_embeddings

    # Get answer embeddings
    print(f"  [3/3] Computing consistency...")
    answer_embeddings = []
    for ans in answers:
        emb = get_embedding(ans)
        if emb:
            answer_embeddings.append(emb)

    if len(answer_embeddings) < 2:
        return 0.0, paraphrases, answers, paraphrase_embeddings

    # Compute consistency
    similarities = pairwise_similarities(answer_embeddings)
    consistency_score = np.mean(similarities)

    print(f"  → Consistency: {consistency_score:.3f}")

    return float(consistency_score), paraphrases, answers, paraphrase_embeddings

# =========================
# FULL MONITORING PIPELINE
# =========================

def monitor_question(question, category="unknown", samples=10, num_paraphrases=3, show_answers=False):
    """
    Complete monitoring pipeline with risk assessment
    
    Args:
        question (str): Question to monitor
        category (str): Question category for tracking
        samples (int): Number of samples for uncertainty
        num_paraphrases (int): Number of paraphrases for consistency
        show_answers (bool): Whether to display sample answers
    
    Returns:
        dict: Complete monitoring record
    """
    print("\n" + "="*70)
    print(f"MONITORING: {question}")
    print("="*70)
    
    start_time = time.time()

    # Get question embedding (for Phase 2 clustering)
    print("\n🔍 EMBEDDING QUESTION...")
    question_embedding = get_embedding(question)

    # Measure uncertainty
    print("\n🔍 UNCERTAINTY ANALYSIS")
    uncertainty_score, sampled_answers, answer_embeddings = measure_uncertainty(
        question, samples=samples, temperature=0.8
    )

    # Measure consistency
    print("\n🔍 CONSISTENCY ANALYSIS")
    consistency_score, paraphrased_questions, paraphrased_answers, paraphrase_embeddings = measure_consistency(
        question, num_paraphrases=num_paraphrases, temperature=0.3
    )

    # Generate risk report
    print("\n🔍 RISK ASSESSMENT")
    risk_report = generate_risk_report(uncertainty_score, consistency_score)
    
    total_time = time.time() - start_time

    # Create comprehensive record
    record = {
        # Basic info
        "timestamp": current_timestamp(),
        "question": question,
        "category": category,
        "model": MODEL_NAME,
        
        # Samples
        "sampled_answers": sampled_answers,
        "paraphrased_questions": paraphrased_questions,
        "paraphrased_answers": paraphrased_answers,
        "num_samples": samples,
        "num_paraphrases": num_paraphrases,
        
        # Core metrics
        "uncertainty_score": uncertainty_score,
        "consistency_score": consistency_score,
        
        # Risk metrics (NEW)
        "calibration_score": risk_report["calibration_score"],
        "risk_score": risk_report["risk_score"],
        "risk_zone": risk_report["risk_zone"],
        "severity": risk_report["severity"],
        
        # Embeddings (for Phase 2)
        "question_embedding": question_embedding,
        "answer_embeddings": answer_embeddings,
        
        # Metadata
        "answer_lengths": [len(ans) for ans in sampled_answers],
        "execution_time_seconds": round(total_time, 2),
        "temperature_uncertainty": 0.8,
        "temperature_consistency": 0.3
    }

    # Log to file
    log_interaction(record)

    # Display results
    print_section_header("📊 MONITORING RESULTS")
    
    print(visualize_score(1 - uncertainty_score, "Confidence", 50))
    print(visualize_score(uncertainty_score, "Uncertainty", 50))
    print(visualize_score(consistency_score, "Consistency", 50))
    print(visualize_score(risk_report["calibration_score"], "Calibration", 50))
    print(visualize_score(1 - risk_report["risk_score"], "Safety", 50))
    
    # Risk assessment
    print(f"\n🎯 RISK ASSESSMENT:")
    print(f"  Risk Zone: {risk_report['emoji']} {risk_report['risk_zone']}")
    print(f"  Severity Level: {risk_report['severity']}/4")
    print(f"  Description: {risk_report['description']}")
    print(f"  Recommendation: {risk_report['recommendation']}")
    
    print(f"\n⏱️  Execution Time: {total_time:.1f}s")

    # Optional: Show sample answers
    if show_answers:
        print_section_header("📝 SAMPLE ANSWERS")
        for i, ans in enumerate(sampled_answers[:3], 1):
            print(f"\n[{i}] {ans[:300]}...")

    return record

# =========================
# BATCH MONITORING
# =========================

def monitor_questions_batch(questions_with_categories, samples=5, num_paraphrases=2):
    """
    Monitor multiple questions in batch
    
    Args:
        questions_with_categories (list): List of dicts with 'question' and 'category'
        samples (int): Samples per question
        num_paraphrases (int): Paraphrases per question
    
    Returns:
        list: All monitoring records
    """
    results = []
    total = len(questions_with_categories)
    
    print(f"\n{'='*70}")
    print(f"  BATCH MONITORING: {total} questions")
    print(f"{'='*70}")
    
    for i, item in enumerate(questions_with_categories, 1):
        print(f"\n\n{'🔄'*35}")
        print(f"Question {i}/{total}")
        print(f"{'🔄'*35}")
        
        record = monitor_question(
            question=item["question"],
            category=item["category"],
            samples=samples,
            num_paraphrases=num_paraphrases,
            show_answers=False
        )
        
        results.append(record)
    
    return results

# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if "--question" in sys.argv:
        idx = sys.argv.index("--question")
        if idx + 1 < len(sys.argv):
            question = sys.argv[idx + 1]
            monitor_question(question, category="custom", samples=10, num_paraphrases=3, show_answers=True)
    else:
        print("Usage: python llm_monitoring.py --question 'Your question here'")
        print("Or import and use monitor_question() function")
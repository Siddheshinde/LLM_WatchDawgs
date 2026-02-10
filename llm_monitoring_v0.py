import json
from datetime import datetime
import os
import google.generativeai as genai

# =========================
# CONFIG
# =========================
API_KEY = "AIzaSyCaM5g7ZOBACNoMAYMPfDC5Tv2BLTBtDKs"  # <-- PUT YOUR REAL KEY HERE
MODEL_NAME = "models/gemini-2.5-flash"  # Using the latest available model
LOG_FILE = "qa_logs.jsonl"

# Configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# =========================
# UTILITY FUNCTIONS
# =========================
def current_timestamp():
    return datetime.utcnow().isoformat()

def log_interaction(record):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# =========================
# LLM CALL (BLACK-BOX)
# =========================
def call_llm(question, temperature=0.7):
    try:
        response = model.generate_content(
            question,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 2048,
            }
        )
        return response.text
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

# =========================
# STEP 1 + 2: QA + LOGGING
# =========================
def qa_once(question, temperature=0.7):
    answer = call_llm(question, temperature)
    if answer is None:
        return "Error generating answer"
    
    record = {
        "timestamp": current_timestamp(),
        "question": question,
        "answer": answer,
        "model": MODEL_NAME,
        "temperature": temperature
    }
    log_interaction(record)
    return answer

# =========================
# STEP 3: UNCERTAINTY SIGNAL
# =========================
def uncertainty_sampling(question, samples=5, temperature=0.8):
    answers = []
    for i in range(samples):
        print(f"  Generating sample {i+1}/{samples}...")
        ans = call_llm(question, temperature)
        if ans:
            answers.append(ans)
    
    if not answers:
        return {"answers": [], "length_variance": 0}
    
    lengths = [len(a) for a in answers]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    
    return {
        "answers": answers,
        "length_variance": variance
    }

# =========================
# STEP 4: CONSISTENCY SIGNAL
# =========================
def rephrase_questions(question):
    return [
        question,
        f"Can you explain: {question}",
        f"Answer this question clearly: {question}"
    ]

def consistency_check(question, temperature=0.3):
    variations = rephrase_questions(question)
    answers = []
    
    for i, q in enumerate(variations, 1):
        print(f"  Testing variation {i}/{len(variations)}...")
        ans = call_llm(q, temperature)
        if ans:
            answers.append(ans)
    
    if not answers:
        return {"answers": [], "consistency": "UNKNOWN"}
    
    lengths = [len(a) for a in answers]
    diff = max(lengths) - min(lengths)
    
    if diff < 50:
        score = "HIGH"
    elif diff < 150:
        score = "MEDIUM"
    else:
        score = "LOW"
    
    return {
        "answers": answers,
        "consistency": score
    }

# =========================
# MAIN RUN
# =========================
if __name__ == "__main__":
    question = "What causes earthquakes?"
    
    print("\n--- STEP 1: BASIC QA ---")
    print(qa_once(question))
    
    print("\n--- STEP 3: UNCERTAINTY ---")
    u = uncertainty_sampling(question)
    print(f"Length variance: {u['length_variance']:.2f}")
    
    print("\n--- STEP 4: CONSISTENCY ---")
    c = consistency_check(question)
    print(f"Consistency score: {c['consistency']}")
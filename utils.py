"""
utils.py
Shared utility functions for LLM Watchdog monitoring system
"""

import json
import numpy as np
from datetime import datetime

# =========================
# TIMESTAMP & LOGGING
# =========================

def current_timestamp():
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def log_interaction(record, log_file="qa_monitoring_logs.jsonl"):
    """Append monitoring record to JSONL log file"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_logs(log_file="qa_monitoring_logs.jsonl"):
    """Load all records from log file"""
    records = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"Warning: {log_file} not found. Starting fresh.")
    return records

# =========================
# VECTOR OPERATIONS
# =========================

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

def pairwise_similarities(embeddings):
    """Compute all pairwise cosine similarities"""
    similarities = []
    n = len(embeddings)
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    return similarities if similarities else [0.0]

def compute_centroid(embeddings):
    """Compute mean vector of embeddings"""
    return np.mean(embeddings, axis=0)

# =========================
# VISUALIZATION HELPERS
# =========================

def visualize_score(score, label, width=50):
    """Create ASCII bar chart for scores"""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{label:<20} [{bar}] {score:.3f}"

def get_color_emoji(value, thresholds=(0.3, 0.7)):
    """Get colored emoji based on value"""
    if value < thresholds[0]:
        return "🔴"
    elif value < thresholds[1]:
        return "🟡"
    else:
        return "🟢"

def print_separator(char="=", length=70):
    """Print visual separator"""
    print(char * length)

def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

# =========================
# STATISTICS
# =========================

def compute_statistics(values):
    """Compute basic statistics for a list of values"""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values))
    }

# =========================
# TIME FORMATTING
# =========================

def format_timestamp(timestamp_str):
    """Format ISO timestamp to readable string"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def time_ago(timestamp_str):
    """Calculate time difference from now"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        delta = datetime.now() - dt
        
        if delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600} hours ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60} minutes ago"
        else:
            return "just now"
    except:
        return "unknown"
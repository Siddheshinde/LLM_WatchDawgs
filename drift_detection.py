"""
drift_detection.py
Phase 2: Temporal Intelligence Module

Detects semantic drift between two consecutive time windows by comparing
the centroids of their question embeddings.

Drift intuition:
  If the LLM is being asked similar questions in both windows, the centroids
  of the question embeddings will be close → low drift.
  If the topic distribution shifts significantly, the centroids diverge →
  high drift.  A sudden jump in drift alongside a metric change is a strong
  signal of behavioral regime change.
"""

import numpy as np


# =========================
# VECTOR UTILITIES
# =========================

def compute_embedding_centroid(embeddings):
    """
    Compute the mean vector (centroid) of a set of embedding vectors.

    Args:
        embeddings (list[list[float]] | list[np.ndarray]):
            A list of embedding vectors.  All vectors must have the same
            dimensionality.

    Returns:
        np.ndarray: Mean vector, or None if input is empty / invalid.
    """
    if not embeddings:
        return None

    # Convert to numpy array; skip any None / empty entries
    valid = [np.array(e, dtype=float) for e in embeddings
             if e is not None and len(e) > 0]

    if not valid:
        return None

    return np.mean(valid, axis=0)


def cosine_similarity(vec1, vec2):
    """
    Cosine similarity between two vectors.

    Args:
        vec1 (array-like): First vector.
        vec2 (array-like): Second vector.

    Returns:
        float: Similarity in [-1, 1]. Returns 0.0 for zero-norm vectors.
    """
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


# =========================
# DRIFT COMPUTATION
# =========================

def compute_drift(embeddings_t1, embeddings_t2):
    """
    Compute semantic drift between two time windows.

    Steps:
      1. Compute centroid of window t1 embeddings.
      2. Compute centroid of window t2 embeddings.
      3. drift = 1 - cosine_similarity(centroid_t1, centroid_t2)

    A drift of 0 means the two windows cover the same semantic territory.
    A drift of 1 means they are completely orthogonal in embedding space.

    Args:
        embeddings_t1 (list[list[float]]): Question embeddings from window t1.
        embeddings_t2 (list[list[float]]): Question embeddings from window t2.

    Returns:
        float: Drift score in [0, 2] (typically 0–1 for unit embeddings).
               Returns 0.0 if either window has no valid embeddings.
    """
    centroid_t1 = compute_embedding_centroid(embeddings_t1)
    centroid_t2 = compute_embedding_centroid(embeddings_t2)

    # If either window has no embeddings we cannot compute drift
    if centroid_t1 is None or centroid_t2 is None:
        return 0.0

    similarity = cosine_similarity(centroid_t1, centroid_t2)
    drift_score = 1.0 - similarity

    return float(drift_score)


def detect_drift(drift_score, threshold=0.15):
    """
    Flag whether a drift score exceeds the alert threshold.

    Args:
        drift_score (float): Output of compute_drift().
        threshold   (float): Default 0.15 — tuned for LLM question embeddings.
                             Lower = more sensitive, higher = less noisy.

    Returns:
        bool: True if drift_score > threshold (drift alert), else False.
    """
    return drift_score > threshold


# =========================
# DETAILED DRIFT REPORT
# =========================

def drift_report(embeddings_t1, embeddings_t2, threshold=0.15):
    """
    Full drift analysis between two windows with labeled severity.

    Args:
        embeddings_t1 (list[list[float]]): Embeddings from window t1.
        embeddings_t2 (list[list[float]]): Embeddings from window t2.
        threshold (float): Alert threshold.

    Returns:
        dict:
            drift_score   (float)  — raw drift value
            drift_alert   (bool)   — whether threshold was crossed
            severity      (str)    — "LOW" | "MODERATE" | "HIGH" | "SEVERE"
            description   (str)    — human-readable summary
    """
    score = compute_drift(embeddings_t1, embeddings_t2)
    alert = detect_drift(score, threshold)

    if score < 0.05:
        severity = "LOW"
        description = "Semantic territory virtually unchanged."
    elif score < 0.15:
        severity = "MODERATE"
        description = "Minor topic shift between windows."
    elif score < 0.30:
        severity = "HIGH"
        description = "Significant semantic drift — question distribution changed."
    else:
        severity = "SEVERE"
        description = "Extreme drift — model may be operating in a new domain."

    return {
        "drift_score":   round(score, 4),
        "drift_alert":   alert,
        "severity":      severity,
        "description":   description,
        "num_t1":        len([e for e in embeddings_t1 if e]),
        "num_t2":        len([e for e in embeddings_t2 if e])
    }

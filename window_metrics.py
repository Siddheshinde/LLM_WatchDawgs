"""
window_metrics.py
Phase 2: Temporal Intelligence Module

Computes aggregate behavioral metrics for a single time window.
A "window" is a list of log records that fall within the same
time bucket (e.g., one 24-hour period).
"""

import numpy as np


def _numeric_values(records, key):
    """Return finite numeric values for a metric key, skipping missing/None values."""
    values = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


# =========================
# CORE WINDOW FUNCTION
# =========================

def compute_window_metrics(records):
    """
    Aggregate behavioral metrics across all records in one time window.

    Args:
        records (list[dict]): Log records belonging to the same window.
                              Each record must have 'uncertainty_score'
                              and 'consistency_score' fields.

    Returns:
        dict with keys:
            mean_uncertainty  (float) — average uncertainty in this window
            mean_consistency  (float) — average consistency in this window
            std_uncertainty   (float) — spread of uncertainty values
            std_consistency   (float) — spread of consistency values
            num_samples       (int)   — how many records contributed
    """
    if not records:
        return {
            "mean_uncertainty": 0.0,
            "mean_consistency": 0.0,
            "std_uncertainty":  0.0,
            "std_consistency":  0.0,
            "num_samples":      0
        }

    uncertainties = np.array(_numeric_values(records, "uncertainty_score"), dtype=float)
    consistencies = np.array(_numeric_values(records, "consistency_score"), dtype=float)

    # Guard: if all records were malformed and we got empty arrays
    if len(uncertainties) == 0:
        uncertainties = np.array([0.0])
    if len(consistencies) == 0:
        consistencies = np.array([0.0])

    return {
        "mean_uncertainty": float(np.mean(uncertainties)),
        "mean_consistency": float(np.mean(consistencies)),
        "std_uncertainty":  float(np.std(uncertainties)),
        "std_consistency":  float(np.std(consistencies)),
        "num_samples":      len(records)
    }


# =========================
# EXTENDED WINDOW STATS
# =========================

def compute_window_risk_distribution(records):
    """
    Count how many records in this window fall into each risk zone.

    Args:
        records (list[dict]): Window records.

    Returns:
        dict: Counts per risk zone and a dominant_zone string.
    """
    if not records:
        return {
            "RELIABLE": 0,
            "OVERCONFIDENT": 0,
            "UNSTABLE": 0,
            "AMBIGUOUS": 0,
            "dominant_zone": "UNKNOWN"
        }

    zone_counts = {"RELIABLE": 0, "OVERCONFIDENT": 0,
                   "UNSTABLE": 0, "AMBIGUOUS": 0}

    for r in records:
        zone = r.get("risk_zone", "AMBIGUOUS")
        if zone in zone_counts:
            zone_counts[zone] += 1

    dominant = max(zone_counts, key=zone_counts.get)

    return {**zone_counts, "dominant_zone": dominant}


def compute_window_calibration(records):
    """
    Average calibration score for the window.
    Uses pre-computed 'calibration_score' if present;
    falls back to the formula consistency / (1 + uncertainty).

    Args:
        records (list[dict]): Window records.

    Returns:
        float: Mean calibration score (0–1).
    """
    if not records:
        return 0.0

    scores = []
    for r in records:
        calibration = r.get("calibration_score")
        if calibration is not None:
            try:
                calibration = float(calibration)
            except (TypeError, ValueError):
                calibration = None
            if calibration is not None and np.isfinite(calibration):
                scores.append(calibration)
        elif r.get("uncertainty_score") is not None and r.get("consistency_score") is not None:
            u = r["uncertainty_score"]
            c = r["consistency_score"]
            try:
                scores.append(float(c) / (1 + float(u)))
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    return float(np.mean(scores)) if scores else 0.0

"""
risk_engine.py
Risk assessment engine for LLM behavioral monitoring
"""

import numpy as np

# =========================
# RISK SCORING FUNCTIONS
# =========================

def calculate_calibration_score(uncertainty, consistency):
    """
    Calibration score: measures how well model's uncertainty matches its consistency
    
    Formula: consistency / (1 + uncertainty)
    
    Range: 0 to 1
    - High score: Model is well-calibrated (consistent and appropriately confident)
    - Low score: Model is poorly calibrated (overconfident or chaotic)
    
    Args:
        uncertainty (float): 0 to 1, higher = more uncertain
        consistency (float): 0 to 1, higher = more consistent
    
    Returns:
        float: Calibration score
    """
    return consistency / (1 + uncertainty)

def calculate_risk_score(uncertainty, consistency):
    """
    Risk score: unified measure of how risky the model's behavior is
    
    Formula: 0.6 * uncertainty + 0.4 * (1 - consistency)
    
    Range: 0 to 1
    - High score: High risk (uncertain and inconsistent)
    - Low score: Low risk (confident and consistent)
    
    Weighting rationale:
    - Uncertainty weighted 0.6: primary signal
    - Inconsistency weighted 0.4: secondary signal
    
    Args:
        uncertainty (float): 0 to 1
        consistency (float): 0 to 1
    
    Returns:
        float: Risk score
    """
    return (0.6 * uncertainty) + (0.4 * (1 - consistency))

# =========================
# RISK ZONE CLASSIFICATION
# =========================

def classify_risk_zone(uncertainty, consistency):
    """
    Classify model behavior into risk zones using 2x2 matrix
    
    Risk Matrix:
    ┌─────────────────┬──────────────────┬──────────────────┐
    │                 │  Low Consistency  │  High Consistency │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ Low Uncertainty │  OVERCONFIDENT   │    RELIABLE      │
    │                 │   (DANGEROUS)    │     (SAFE)       │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ High Uncertainty│    UNSTABLE      │    AMBIGUOUS     │
    │                 │  (HALLUCINATING) │  (APPROPRIATE)   │
    └─────────────────┴──────────────────┴──────────────────┘
    
    Thresholds:
    - High uncertainty: > 0.4
    - High consistency: > 0.6
    
    Args:
        uncertainty (float): 0 to 1
        consistency (float): 0 to 1
    
    Returns:
        str: Risk zone name
    """
    high_uncertainty = uncertainty > 0.4
    high_consistency = consistency > 0.6
    
    if not high_uncertainty and high_consistency:
        return "RELIABLE"
    elif not high_uncertainty and not high_consistency:
        return "OVERCONFIDENT"  # Most dangerous!
    elif high_uncertainty and not high_consistency:
        return "UNSTABLE"
    else:
        return "AMBIGUOUS"

# =========================
# RISK ZONE METADATA
# =========================

def get_risk_metadata(zone):
    """
    Get metadata for each risk zone
    
    Returns dictionary with:
    - color: hex color code
    - emoji: visual indicator
    - severity: numeric severity level
    - description: human-readable explanation
    - recommendation: what to do
    """
    metadata = {
        "RELIABLE": {
            "color": "#10b981",
            "emoji": "✅",
            "severity": 1,
            "description": "Model is confident and consistent",
            "recommendation": "Safe to use. High confidence in answer."
        },
        "OVERCONFIDENT": {
            "color": "#ef4444",
            "emoji": "⛔",
            "severity": 4,
            "description": "Model appears confident but changes answer when rephrased",
            "recommendation": "DO NOT TRUST. Model may be hallucinating with false confidence."
        },
        "UNSTABLE": {
            "color": "#f59e0b",
            "emoji": "⚠️",
            "severity": 3,
            "description": "Model gives variable answers and is uncertain",
            "recommendation": "Flag for human review. High hallucination risk."
        },
        "AMBIGUOUS": {
            "color": "#6366f1",
            "emoji": "ℹ️",
            "severity": 2,
            "description": "Model is uncertain but consistent in expressing uncertainty",
            "recommendation": "Question may need clarification or is genuinely subjective."
        }
    }
    
    return metadata.get(zone, {
        "color": "#gray",
        "emoji": "❓",
        "severity": 2,
        "description": "Unknown risk zone",
        "recommendation": "Unable to classify"
    })

# =========================
# COMPREHENSIVE RISK REPORT
# =========================

def generate_risk_report(uncertainty, consistency):
    """
    Generate comprehensive risk assessment report
    
    Args:
        uncertainty (float): Uncertainty score
        consistency (float): Consistency score
    
    Returns:
        dict: Complete risk assessment
    """
    calibration = calculate_calibration_score(uncertainty, consistency)
    risk_score = calculate_risk_score(uncertainty, consistency)
    risk_zone = classify_risk_zone(uncertainty, consistency)
    metadata = get_risk_metadata(risk_zone)
    
    return {
        "uncertainty": round(uncertainty, 4),
        "consistency": round(consistency, 4),
        "calibration_score": round(calibration, 4),
        "risk_score": round(risk_score, 4),
        "risk_zone": risk_zone,
        "severity": metadata["severity"],
        "color": metadata["color"],
        "emoji": metadata["emoji"],
        "description": metadata["description"],
        "recommendation": metadata["recommendation"]
    }

# =========================
# BATCH ANALYSIS
# =========================

def analyze_risk_distribution(records):
    """
    Analyze risk distribution across multiple records
    
    Args:
        records (list): List of monitoring records
    
    Returns:
        dict: Risk distribution statistics
    """
    if not records:
        return {}
    
    zones = [r.get("risk_zone", "UNKNOWN") for r in records]
    
    distribution = {
        "RELIABLE": zones.count("RELIABLE"),
        "OVERCONFIDENT": zones.count("OVERCONFIDENT"),
        "UNSTABLE": zones.count("UNSTABLE"),
        "AMBIGUOUS": zones.count("AMBIGUOUS")
    }
    
    total = sum(distribution.values())
    
    percentages = {
        zone: (count / total * 100) if total > 0 else 0
        for zone, count in distribution.items()
    }
    
    # Calculate overall system health score
    # Higher is better (more reliable, fewer dangerous zones)
    health_score = (
        distribution["RELIABLE"] * 1.0 +
        distribution["AMBIGUOUS"] * 0.5 +
        distribution["UNSTABLE"] * 0.2 +
        distribution["OVERCONFIDENT"] * 0.0
    ) / total if total > 0 else 0
    
    return {
        "distribution": distribution,
        "percentages": percentages,
        "total": total,
        "health_score": round(health_score, 3),
        "critical_count": distribution["OVERCONFIDENT"] + distribution["UNSTABLE"]
    }
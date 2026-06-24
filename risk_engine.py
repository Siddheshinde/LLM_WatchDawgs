"""
risk_engine.py
Risk assessment engine for LLM behavioral monitoring
"""

import numpy as np

UNCERTAINTY_HIGH_THRESHOLD = 0.12
CONSISTENCY_HIGH_THRESHOLD = 0.75

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
        float | None: Calibration score, or None if inputs are invalid
    """
    if uncertainty is None or consistency is None:
        return None
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
        float | None: Risk score, or None if inputs are invalid
    """
    if uncertainty is None or consistency is None:
        return None
    return (0.6 * uncertainty) + (0.4 * (1 - consistency))

# =========================
# RISK ZONE CLASSIFICATION
# =========================

def classify_risk_zone(uncertainty, consistency):
    """
    Classify model behavior into risk zones using 2x2 matrix
    
    Risk Matrix:
                    Low Consistency    High Consistency
    Low Uncertainty OVERCONFIDENT      RELIABLE
    High Uncertainty UNSTABLE          AMBIGUOUS
    
    Thresholds:
    - High uncertainty: > 0.12
    - High consistency: >= 0.75
    
    Args:
        uncertainty (float): 0 to 1
        consistency (float): 0 to 1
    
    Returns:
        str: Risk zone name
    """
    if uncertainty is None or consistency is None:
        return "UNKNOWN"

    high_uncertainty = uncertainty > UNCERTAINTY_HIGH_THRESHOLD
    high_consistency = consistency >= CONSISTENCY_HIGH_THRESHOLD
    
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
            "emoji": "OK",
            "severity": 1,
            "description": "Model is confident and consistent",
            "recommendation": "Safe to use. High confidence in answer."
        },
        "OVERCONFIDENT": {
            "color": "#ef4444",
            "emoji": "STOP",
            "severity": 4,
            "description": "Model appears confident but changes answer when rephrased",
            "recommendation": "DO NOT TRUST. Model may be hallucinating with false confidence."
        },
        "UNSTABLE": {
            "color": "#f59e0b",
            "emoji": "WARN",
            "severity": 3,
            "description": "Model gives variable answers and is uncertain",
            "recommendation": "Flag for human review. High hallucination risk."
        },
        "AMBIGUOUS": {
            "color": "#6366f1",
            "emoji": "INFO",
            "severity": 2,
            "description": "Model is uncertain but consistent in expressing uncertainty",
            "recommendation": "Question may need clarification or is genuinely subjective."
        }
    }
    
    return metadata.get(zone, {
        "color": "#gray",
        "emoji": "?",
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
    risk_zone = classify_risk_zone(uncertainty, consistency)
    metadata = get_risk_metadata(risk_zone)

    if uncertainty is None or consistency is None:
        return {
            "uncertainty": None,
            "consistency": None,
            "calibration_score": None,
            "risk_score": None,
            "risk_zone": risk_zone,
            "severity": 0,
            "color": metadata["color"],
            "emoji": metadata["emoji"],
            "description": "Unable to assess risk because uncertainty or consistency could not be computed.",
            "recommendation": "Check whether Ollama embeddings are available, the model supports embeddings, and the API response is valid."
        }

    calibration = calculate_calibration_score(uncertainty, consistency)
    risk_score = calculate_risk_score(uncertainty, consistency)

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

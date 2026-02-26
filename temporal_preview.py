"""
temporal_preview.py
Basic temporal analysis for Phase 1 demonstration
Full implementation in Phase 2
"""

import numpy as np
from datetime import datetime, timedelta
from utils import load_logs, compute_statistics

# =========================
# ROLLING WINDOW ANALYSIS
# =========================

def compute_rolling_statistics(records, window_size=5):
    """
    Compute statistics over last N interactions
    
    Args:
        records (list): List of monitoring records
        window_size (int): Number of recent records to analyze
    
    Returns:
        dict: Rolling statistics
    """
    if not records:
        return None
    
    if len(records) < window_size:
        window_size = len(records)
    
    recent = records[-window_size:]
    
    uncertainties = [r["uncertainty_score"] for r in recent]
    consistencies = [r["consistency_score"] for r in recent]
    calibrations = [r.get("calibration_score", 0) for r in recent]
    
    return {
        "window_size": window_size,
        "mean_uncertainty": np.mean(uncertainties),
        "mean_consistency": np.mean(consistencies),
        "mean_calibration": np.mean(calibrations),
        "std_uncertainty": np.std(uncertainties),
        "std_consistency": np.std(consistencies),
        "min_uncertainty": np.min(uncertainties),
        "max_uncertainty": np.max(uncertainties),
        "risk_zones": count_risk_zones(recent)
    }

def count_risk_zones(records):
    """Count occurrences of each risk zone"""
    zones = [r.get("risk_zone", "UNKNOWN") for r in records]
    return {
        "RELIABLE": zones.count("RELIABLE"),
        "OVERCONFIDENT": zones.count("OVERCONFIDENT"),
        "UNSTABLE": zones.count("UNSTABLE"),
        "AMBIGUOUS": zones.count("AMBIGUOUS")
    }

# =========================
# TREND DETECTION
# =========================

def detect_simple_trend(records, metric="uncertainty_score"):
    """
    Detect if metric is increasing, decreasing, or stable
    
    Args:
        records (list): Monitoring records
        metric (str): Which metric to analyze
    
    Returns:
        dict: Trend analysis
    """
    if len(records) < 10:
        return {
            "trend": "INSUFFICIENT_DATA",
            "confidence": 0.0,
            "description": "Need at least 10 interactions for trend analysis"
        }
    
    # Split into two halves
    mid = len(records) // 2
    first_half = records[:mid]
    second_half = records[mid:]
    
    # Compute means
    mean1 = np.mean([r[metric] for r in first_half])
    mean2 = np.mean([r[metric] for r in second_half])
    
    delta = mean2 - mean1
    delta_pct = (delta / mean1 * 100) if mean1 > 0 else 0
    
    # Classify trend
    if abs(delta) < 0.05:
        trend = "STABLE"
        description = f"No significant change ({delta_pct:+.1f}%)"
    elif delta > 0.1:
        trend = "DEGRADING"
        description = f"Metric increasing by {delta_pct:+.1f}% - possible degradation"
    elif delta < -0.1:
        trend = "IMPROVING"
        description = f"Metric decreasing by {delta_pct:+.1f}% - improvement detected"
    elif delta > 0:
        trend = "SLIGHTLY_DEGRADING"
        description = f"Minor increase of {delta_pct:+.1f}%"
    else:
        trend = "SLIGHTLY_IMPROVING"
        description = f"Minor decrease of {delta_pct:+.1f}%"
    
    return {
        "trend": trend,
        "delta": delta,
        "delta_pct": delta_pct,
        "first_half_mean": mean1,
        "second_half_mean": mean2,
        "description": description,
        "confidence": min(abs(delta_pct) / 10, 1.0)
    }

# =========================
# TIME-BASED FILTERING
# =========================

def filter_by_timerange(records, hours=24):
    """
    Filter records to only those within last N hours
    
    Args:
        records (list): All records
        hours (int): Time window in hours
    
    Returns:
        list: Filtered records
    """
    if not records:
        return []
    
    cutoff = datetime.now() - timedelta(hours=hours)
    
    filtered = []
    for r in records:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
            if ts >= cutoff:
                filtered.append(r)
        except:
            continue
    
    return filtered

# =========================
# TEMPORAL REPORT
# =========================

def generate_temporal_report(records):
    """
    Generate comprehensive temporal analysis report
    
    Args:
        records (list): All monitoring records
    
    Returns:
        dict: Temporal analysis report
    """
    if not records:
        return {"error": "No records available"}
    
    # Overall statistics
    all_uncertainties = [r["uncertainty_score"] for r in records]
    all_consistencies = [r["consistency_score"] for r in records]
    
    # Rolling window analysis
    rolling_5 = compute_rolling_statistics(records, window_size=5)
    rolling_10 = compute_rolling_statistics(records, window_size=10)
    
    # Trend detection
    uncertainty_trend = detect_simple_trend(records, "uncertainty_score")
    consistency_trend = detect_simple_trend(records, "consistency_score")
    
    # Recent vs historical
    recent_24h = filter_by_timerange(records, hours=24)
    
    return {
        "total_interactions": len(records),
        "time_range": {
            "first": records[0]["timestamp"],
            "last": records[-1]["timestamp"]
        },
        "overall_stats": {
            "mean_uncertainty": np.mean(all_uncertainties),
            "mean_consistency": np.mean(all_consistencies),
            "std_uncertainty": np.std(all_uncertainties),
            "std_consistency": np.std(all_consistencies)
        },
        "rolling_window_5": rolling_5,
        "rolling_window_10": rolling_10,
        "trends": {
            "uncertainty": uncertainty_trend,
            "consistency": consistency_trend
        },
        "recent_24h": {
            "count": len(recent_24h),
            "mean_uncertainty": np.mean([r["uncertainty_score"] for r in recent_24h]) if recent_24h else 0,
            "mean_consistency": np.mean([r["consistency_score"] for r in recent_24h]) if recent_24h else 0
        }
    }

def print_temporal_report(report):
    """Print temporal report in readable format"""
    print("\n" + "="*70)
    print("  TEMPORAL ANALYSIS REPORT")
    print("="*70)
    
    print(f"\nTotal Interactions: {report['total_interactions']}")
    print(f"Time Range: {report['time_range']['first']} to {report['time_range']['last']}")
    
    print("\n--- OVERALL STATISTICS ---")
    stats = report['overall_stats']
    print(f"Mean Uncertainty: {stats['mean_uncertainty']:.3f} (±{stats['std_uncertainty']:.3f})")
    print(f"Mean Consistency: {stats['mean_consistency']:.3f} (±{stats['std_consistency']:.3f})")
    
    if report.get('rolling_window_5'):
        print("\n--- ROLLING WINDOW (Last 5) ---")
        rw = report['rolling_window_5']
        print(f"Mean Uncertainty: {rw['mean_uncertainty']:.3f}")
        print(f"Mean Consistency: {rw['mean_consistency']:.3f}")
        print(f"Mean Calibration: {rw['mean_calibration']:.3f}")
    
    print("\n--- TREND ANALYSIS ---")
    u_trend = report['trends']['uncertainty']
    c_trend = report['trends']['consistency']
    print(f"Uncertainty Trend: {u_trend['trend']}")
    print(f"  {u_trend['description']}")
    print(f"Consistency Trend: {c_trend['trend']}")
    print(f"  {c_trend['description']}")
    
    print("\n" + "="*70)
"""
main_pipeline.py
LLM Watchdog - root-level integration pipeline

Exposes run_full_pipeline(question, category, samples, paraphrases) which:
  1. Monitors the question (uncertainty + consistency + risk)
  2. Runs temporal analysis on the log history
  3. Looks up cluster assignment (optional, graceful fallback)
  4. Fires rule-based alerts
  5. Builds a single unified output record
  6. Appends it to final_risk_reports.jsonl
"""

import json
import sys
from utils import current_timestamp, log_interaction, load_logs, print_section_header

# ========================
# CONSTANTS
# ========================

PIPELINE_VERSION   = "1.0.0"
LOG_FILE           = "qa_monitoring_logs.jsonl"
FINAL_REPORTS_FILE = "final_risk_reports.jsonl"
CLUSTERED_LOG_FILE = "clustered_logs.jsonl"

# Alert thresholds
UNCERTAINTY_HIGH_THRESHOLD  = 0.6
CONSISTENCY_LOW_THRESHOLD   = 0.4
RISK_SCORE_HIGH_THRESHOLD   = 0.7


# ========================
# ALERT ENGINE
# ========================

def generate_alerts(uncertainty, consistency, risk_score, risk_zone,
                    drift_alert, change_point):
    """
    Rule-based alert engine. Returns a list of fired alert dicts.

    Alert types:
      HIGH_UNCERTAINTY        - uncertainty > threshold
      LOW_CONSISTENCY         - consistency < threshold
      DRIFT_ALERT             - temporal drift threshold crossed
      CHANGE_POINT            - abrupt behavioral regime shift
      HIGH_RISK_HALLUCINATION - risk_score > threshold OR zone == OVERCONFIDENT
    """
    alerts = []

    if uncertainty is not None and uncertainty > UNCERTAINTY_HIGH_THRESHOLD:
        alerts.append({
            "alert_type": "HIGH_UNCERTAINTY",
            "severity":   "WARNING",
            "message":    f"Uncertainty {uncertainty:.3f} exceeds threshold {UNCERTAINTY_HIGH_THRESHOLD}",
            "value":      round(uncertainty, 4),
            "threshold":  UNCERTAINTY_HIGH_THRESHOLD,
        })

    if consistency is not None and consistency < CONSISTENCY_LOW_THRESHOLD:
        alerts.append({
            "alert_type": "LOW_CONSISTENCY",
            "severity":   "WARNING",
            "message":    f"Consistency {consistency:.3f} below threshold {CONSISTENCY_LOW_THRESHOLD}",
            "value":      round(consistency, 4),
            "threshold":  CONSISTENCY_LOW_THRESHOLD,
        })

    if drift_alert:
        alerts.append({
            "alert_type": "DRIFT_ALERT",
            "severity":   "HIGH",
            "message":    "Semantic drift between time windows exceeds alert threshold",
            "value":      None,
            "threshold":  0.15,
        })

    if change_point:
        alerts.append({
            "alert_type": "CHANGE_POINT",
            "severity":   "HIGH",
            "message":    "Abrupt behavioral change point detected in uncertainty trajectory",
            "value":      None,
            "threshold":  0.25,
        })

    if (risk_score is not None and risk_score > RISK_SCORE_HIGH_THRESHOLD) or risk_zone == "OVERCONFIDENT":
        alerts.append({
            "alert_type": "HIGH_RISK_HALLUCINATION",
            "severity":   "CRITICAL",
            "message": (
                f"Hallucination risk elevated - zone={risk_zone}, "
                f"risk_score={risk_score:.3f}" if risk_score is not None else "risk_score=N/A"
            ),
            "value":      round(risk_score, 4) if risk_score is not None else None,
            "threshold":  RISK_SCORE_HIGH_THRESHOLD,
        })

    return alerts


# ========================
# CLUSTER LOOKUP (optional)
# ========================

def _lookup_cluster(question, clustered_log_file=CLUSTERED_LOG_FILE):
    """
    Find the cluster_id assigned to a question in the pre-clustered log.
    Returns -1 if the question is not found or the file is unavailable.
    """
    try:
        with open(clustered_log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("question", "").strip() == question.strip():
                        return int(rec.get("cluster_id", -1))
                except (json.JSONDecodeError, ValueError):
                    continue
    except FileNotFoundError:
        pass
    return -1


# ========================
# TEMPORAL SNAPSHOT
# ========================

def _get_latest_temporal_snapshot(log_file=LOG_FILE):
    """
    Run temporal analysis and return the most recent window's drift data.
    Returns safe defaults if no prior records exist.
    """
    defaults = {
        "window_key":        None,
        "drift_score":       0.0,
        "drift_alert":       False,
        "drift_severity":    "LOW",
        "change_point":      False,
        "delta_uncertainty": 0.0,
        "delta_consistency": 0.0,
    }

    try:
        from temporal_engine import run_temporal_analysis
        windows = run_temporal_analysis(log_file, window_size="24h")
        if not windows:
            return defaults
        latest = windows[-1]
        return {
            "window_key":        latest.get("window_key"),
            "drift_score":       latest.get("drift_score")       or 0.0,
            "drift_alert":       bool(latest.get("drift_alert",  False)),
            "drift_severity":    latest.get("drift_severity",    "LOW"),
            "change_point":      bool(latest.get("change_point", False)),
            "delta_uncertainty": latest.get("delta_uncertainty") or 0.0,
            "delta_consistency": latest.get("delta_consistency") or 0.0,
        }
    except Exception as exc:
        print(f"  [WARN] Temporal analysis failed: {exc}")
        return defaults


def _round_metric(value, digits=4):
    return None if value is None else round(value, digits)


def _format_metric(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


# ========================
# MAIN PIPELINE
# ========================

def run_full_pipeline(
    question,
    category="unknown",
    samples=10,
    paraphrases=3,
    show_answers=False,
    expected_behavior=None,
    log_file=LOG_FILE,
    reports_file=FINAL_REPORTS_FILE,
):
    """
    Full LLM Watchdog evaluation pipeline for a single question.

    Args:
        question     (str):  The natural-language question to evaluate.
        category     (str):  Semantic category label (factual/opinion/etc.).
        samples      (int):  Uncertainty sampling count (default 10).
        paraphrases  (int):  Number of paraphrases for consistency (default 3).
        show_answers (bool): Print sample answers to stdout.
        log_file     (str):  JSONL log used by temporal analysis.
        reports_file (str):  Output JSONL for unified reports.

    Returns:
        dict: Unified pipeline output record (see schema below).

    Output schema
    -------------
    pipeline_version, run_timestamp, question, category, model,
    num_samples, num_paraphrases,
    uncertainty_score, consistency_score, calibration_score,
    risk_score, risk_zone, severity,
    drift_score, drift_alert, drift_severity, change_point,
    delta_uncertainty, delta_consistency, window_key,
    cluster_id,
    alerts            (list[dict]),
    alert_count       (int),
    execution_time_seconds
    """
    from behavior_evaluator import evaluate_answer_behavior
    from llm_monitoring import monitor_question

    print("\n" + "#" * 65)
    print("  LLM WATCHDOG - FULL PIPELINE")
    print("#" * 65)
    print(f"  Question : {question}")
    print(f"  Category : {category}")
    print(f"  Samples  : {samples}   Paraphrases: {paraphrases}")
    print("#" * 65)

    # Step 1: Monitor uncertainty, consistency, and base risk.
    raw = monitor_question(
        question=question,
        category=category,
        samples=samples,
        num_paraphrases=paraphrases,
        show_answers=show_answers,
    )

    uncertainty   = raw.get("uncertainty_score")
    consistency   = raw.get("consistency_score")
    calibration   = raw.get("calibration_score")
    risk_score    = raw.get("risk_score")
    risk_zone     = raw.get("risk_zone",           "AMBIGUOUS")
    severity      = raw.get("severity",            2)
    exec_time     = raw.get("execution_time_seconds", 0.0)

    behavior = evaluate_answer_behavior(
        question=question,
        answers=raw.get("sampled_answers", []),
        category=category,
        expected_behavior=expected_behavior,
    )

    base_risk_zone = risk_zone
    behavior_adjusted = False
    if not behavior["behavior_pass"]:
        behavior_adjusted = True
        risk_zone = "OVERCONFIDENT" if uncertainty is not None and uncertainty <= 0.12 else "UNSTABLE"
        severity = 4 if risk_zone == "OVERCONFIDENT" else 3
        risk_score = max(risk_score or 0.0, 0.65 if risk_zone == "OVERCONFIDENT" else 0.5)

    # Step 2: Temporal analysis.
    print("\n  [Pipeline] Running temporal analysis...")
    temporal = _get_latest_temporal_snapshot(log_file)

    # Step 3: Cluster lookup.
    print("  [Pipeline] Looking up cluster assignment...")
    cluster_id = _lookup_cluster(question)

    # Step 4: Alert engine.
    alerts = generate_alerts(
        uncertainty=uncertainty,
        consistency=consistency,
        risk_score=risk_score,
        risk_zone=risk_zone,
        drift_alert=temporal["drift_alert"],
        change_point=temporal["change_point"],
    )

    if not behavior["behavior_pass"]:
        alerts.append({
            "alert_type": "EXPECTED_BEHAVIOR_FAILED",
            "severity": "HIGH",
            "message": "; ".join(behavior["behavior_flags"]),
            "value": behavior["behavior_score"],
            "threshold": 1.0,
        })

    # Step 5: Assemble unified record.
    record = {
        # Provenance
        "pipeline_version":    PIPELINE_VERSION,
        "run_timestamp":       current_timestamp(),

        # Question metadata
        "question":            question,
        "category":            category,
        "model":               raw.get("model", "unknown"),
        "num_samples":         samples,
        "num_paraphrases":     paraphrases,

        # Core behavioral metrics
        "uncertainty_score":   _round_metric(uncertainty),
        "consistency_score":   _round_metric(consistency),
        "calibration_score":   _round_metric(calibration),
        "risk_score":          _round_metric(risk_score),
        "risk_zone":           risk_zone,
        "base_risk_zone":      base_risk_zone,
        "behavior_adjusted":   behavior_adjusted,
        "severity":            severity,
        "expected_behavior":   behavior["expected_behavior"],
        "behavior_pass":       behavior["behavior_pass"],
        "behavior_score":      behavior["behavior_score"],
        "behavior_flags":      behavior["behavior_flags"],
        "behavior_signals":    behavior["behavior_signals"],

        # Temporal / drift
        "window_key":          temporal["window_key"],
        "drift_score":         round(temporal["drift_score"], 4),
        "drift_alert":         temporal["drift_alert"],
        "drift_severity":      temporal["drift_severity"],
        "change_point":        temporal["change_point"],
        "delta_uncertainty":   round(temporal["delta_uncertainty"], 4),
        "delta_consistency":   round(temporal["delta_consistency"], 4),

        # Cluster
        "cluster_id":          cluster_id,

        # Alerts
        "alerts":              alerts,
        "alert_count":         len(alerts),

        # Timing
        "execution_time_seconds": round(exec_time, 2),
    }

    # Step 6: Persist.
    log_interaction(record, log_file=reports_file)
    print(f"\n  [Pipeline] Record saved -> {reports_file}")

    # Step 7: Print summary.
    _print_pipeline_summary(record)

    return record


# ========================
# SUMMARY PRINTER
# ========================

def _print_pipeline_summary(rec):
    """Print a clean one-screen summary of the pipeline result."""
    zone_icons = {
        "RELIABLE": "OK",
        "OVERCONFIDENT": "STOP",
        "UNSTABLE": "WARN",
        "AMBIGUOUS": "INFO",
    }
    sev_icons = {1: "LOW", 2: "MED", 3: "HIGH", 4: "CRIT"}

    icon  = zone_icons.get(rec["risk_zone"], "?")
    s_icon = sev_icons.get(rec["severity"], "UNK")

    print_section_header("PIPELINE RESULT SUMMARY")
    print(f"  {'Uncertainty':<22}: {_format_metric(rec['uncertainty_score'])}")
    print(f"  {'Consistency':<22}: {_format_metric(rec['consistency_score'])}")
    print(f"  {'Calibration':<22}: {_format_metric(rec['calibration_score'])}")
    print(f"  {'Risk Score':<22}: {_format_metric(rec['risk_score'])}")
    print(f"  {'Risk Zone':<22}: {icon} {rec['risk_zone']}")
    if rec.get("behavior_adjusted"):
        print(f"  {'Base Risk Zone':<22}: {rec.get('base_risk_zone', 'n/a')}")
    print(f"  {'Expected Behavior':<22}: {rec.get('expected_behavior', 'n/a')}")
    print(f"  {'Behavior Check':<22}: {'pass' if rec.get('behavior_pass') else 'fail'}")
    print(f"  {'Severity':<22}: {s_icon} {rec['severity']}/4")
    print(f"  {'Drift Score':<22}: {rec['drift_score']:.4f}  "
          f"({'ALERT' if rec['drift_alert'] else 'ok'})")
    print(f"  {'Change Point':<22}: {'YES' if rec['change_point'] else 'no'}")
    print(f"  {'Cluster':<22}: {rec['cluster_id'] if rec['cluster_id'] >= 0 else 'n/a'}")

    if rec["alerts"]:
        print(f"\n  ALERTS FIRED ({rec['alert_count']}):")
        for a in rec["alerts"]:
            print(f"    [{a['severity']}] {a['alert_type']} - {a['message']}")
    else:
        print("\n  No alerts fired.")

    print(f"\n  Execution time: {rec['execution_time_seconds']}s")
    print()


# ========================
# ENTRY POINT
# ========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Watchdog - full pipeline")
    parser.add_argument("--question",   required=True,  help="Question to evaluate")
    parser.add_argument("--category",   default="unknown")
    parser.add_argument("--samples",    type=int, default=10)
    parser.add_argument("--paraphrases",type=int, default=3)
    parser.add_argument("--expected-behavior", default=None)
    parser.add_argument("--show-answers", action="store_true")
    args = parser.parse_args()

    run_full_pipeline(
        question=args.question,
        category=args.category,
        samples=args.samples,
        paraphrases=args.paraphrases,
        expected_behavior=args.expected_behavior,
        show_answers=args.show_answers,
    )
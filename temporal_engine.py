"""
temporal_engine.py
Phase 2: Temporal Intelligence Module — Main Orchestrator

Loads JSONL logs, groups records into time windows, computes per-window
behavioral metrics, detects semantic drift between consecutive windows,
and flags change points where metrics shifted abruptly.

Pipeline:
  load_logs()
      ↓
  group_by_time_window()
      ↓  (for each window)
  compute_window_metrics()      ← window_metrics.py
      ↓  (between consecutive windows)
  compute_drift()               ← drift_detection.py
  detect_change_point()
      ↓
  list of per-window result dicts
"""

import json
from datetime import datetime, timedelta

import numpy as np

from window_metrics import (
    compute_window_metrics,
    compute_window_risk_distribution,
    compute_window_calibration,
)
from drift_detection import compute_drift, detect_drift, drift_report


# =========================
# LOG LOADING
# =========================

def load_logs(file_path="qa_monitoring_logs.jsonl"):
    """
    Read all records from a JSONL log file.

    Each line must be a valid JSON object.  Malformed lines are skipped
    with a warning so one bad record never crashes the pipeline.

    Args:
        file_path (str): Path to the JSONL log file.

    Returns:
        list[dict]: Parsed records, in file order.
    """
    records = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Skipping malformed line {line_num}: {e}")
    except FileNotFoundError:
        print(f"  [ERROR] Log file not found: {file_path}")

    print(f"  [load_logs] Loaded {len(records)} records from '{file_path}'")
    return records


# =========================
# TIMESTAMP PARSING
# =========================

def _parse_timestamp(ts_str):
    """
    Parse an ISO-format timestamp string into a datetime object.

    Handles both 'YYYY-MM-DDTHH:MM:SS' and 'YYYY-MM-DD HH:MM:SS' formats.

    Args:
        ts_str (str): Timestamp string.

    Returns:
        datetime | None: Parsed datetime, or None on failure.
    """
    if not ts_str:
        return None
    try:
        # Replace space separator with 'T' for uniform parsing
        return datetime.fromisoformat(ts_str.replace(" ", "T"))
    except (ValueError, TypeError):
        return None


# =========================
# TIME WINDOW GROUPING
# =========================

def group_by_time_window(records, window_size="24h"):
    """
    Partition records into non-overlapping time buckets.

    Supported window sizes:
      "24h" — one bucket per calendar day  (key: "YYYY-MM-DD")
      "7d"  — one bucket per ISO week      (key: "YYYY-WNN")

    Records without a parseable timestamp are placed in a special
    "_unparseable" bucket so they are never silently lost.

    Args:
        records     (list[dict]): All log records.
        window_size (str):        "24h" or "7d".

    Returns:
        dict[str, list[dict]]: Ordered mapping of bucket-key → records.
    """
    if window_size not in ("24h", "7d"):
        raise ValueError(f"Unsupported window_size '{window_size}'. Use '24h' or '7d'.")

    buckets = {}

    for record in records:
        ts = _parse_timestamp(record.get("timestamp"))

        if ts is None:
            key = "_unparseable"
        elif window_size == "24h":
            key = ts.strftime("%Y-%m-%d")          # e.g. "2026-04-16"
        else:  # "7d"
            # ISO week: year + week number
            iso = ts.isocalendar()
            key = f"{iso[0]}-W{iso[1]:02d}"        # e.g. "2026-W16"

        buckets.setdefault(key, []).append(record)

    # Sort buckets chronologically (lexicographic order works for both formats)
    sorted_buckets = dict(sorted(
        ((k, v) for k, v in buckets.items() if k != "_unparseable"),
        key=lambda kv: kv[0]
    ))

    # Append unparseable bucket at the end if it exists
    if "_unparseable" in buckets:
        print(f"  [WARN] {len(buckets['_unparseable'])} records had unparseable timestamps.")
        sorted_buckets["_unparseable"] = buckets["_unparseable"]

    return sorted_buckets


# =========================
# CHANGE POINT DETECTION
# =========================

def detect_change_point(prev, curr):
    """
    Detect an abrupt behavioral shift between two consecutive windows.

    Rule: if the absolute change in mean uncertainty exceeds 0.25 the
    window boundary is flagged as a change point.

    This threshold catches genuine regime changes (e.g., a model update,
    a topic domain switch) while ignoring gradual drift which is handled
    separately via the drift score.

    Args:
        prev (dict): Metrics dict for the earlier window.
        curr (dict): Metrics dict for the later window.

    Returns:
        bool: True if a change point is detected.
    """
    delta = abs(curr["mean_uncertainty"] - prev["mean_uncertainty"])
    return delta > 0.25


# =========================
# PER-WINDOW METRIC COMPUTATION
# =========================

def compute_temporal_metrics(records, window_size="24h"):
    """
    Group records into time windows and compute per-window metrics.

    For each window this returns a dict with behavioral statistics.
    Consecutive-window deltas and drift are NOT computed here — that
    happens in run_temporal_analysis() which has access to all windows.

    Args:
        records     (list[dict]): All log records.
        window_size (str):        "24h" or "7d".

    Returns:
        list[dict]: One entry per time window, sorted chronologically.
    """
    # Sort globally by timestamp so within-bucket ordering is correct too
    def sort_key(r):
        ts = _parse_timestamp(r.get("timestamp"))
        return ts if ts is not None else datetime.min

    sorted_records = sorted(records, key=sort_key)
    buckets = group_by_time_window(sorted_records, window_size)

    window_results = []

    for bucket_key, bucket_records in buckets.items():
        if bucket_key == "_unparseable":
            continue  # Skip malformed records in metric output

        metrics = compute_window_metrics(bucket_records)
        risk_dist = compute_window_risk_distribution(bucket_records)
        mean_calibration = compute_window_calibration(bucket_records)

        window_results.append({
            "window_key":      bucket_key,
            "window_type":     window_size,
            "num_samples":     metrics["num_samples"],
            "mean_uncertainty": round(metrics["mean_uncertainty"], 4),
            "mean_consistency": round(metrics["mean_consistency"], 4),
            "std_uncertainty":  round(metrics["std_uncertainty"],  4),
            "std_consistency":  round(metrics["std_consistency"],  4),
            "mean_calibration": round(mean_calibration, 4),
            "risk_distribution": risk_dist,
            # Populated in run_temporal_analysis() after all windows exist:
            "delta_uncertainty": None,
            "delta_consistency": None,
            "drift_score":       None,
            "drift_alert":       False,
            "change_point":      False,
        })

    return window_results


# =========================
# MAIN ANALYSIS PIPELINE
# =========================

def run_temporal_analysis(log_file="qa_monitoring_logs.jsonl", window_size="24h"):
    """
    Full temporal analysis pipeline.

    Steps:
      1. Load logs from JSONL file.
      2. Group into time windows.
      3. Compute per-window behavioral metrics.
      4. For consecutive window pairs:
           a. Compute embedding-based drift score.
           b. Detect whether the boundary is a change point.
           c. Record delta metrics (how much uncertainty / consistency shifted).
      5. Return list of fully annotated window result dicts.

    Args:
        log_file    (str): Path to the JSONL log file.
        window_size (str): "24h" or "7d".

    Returns:
        list[dict]: One result dict per window (see OUTPUT FORMAT in spec).
    """
    print("\n" + "="*60)
    print("  PHASE 2: TEMPORAL INTELLIGENCE MODULE")
    print("="*60)

    # Step 1: Load
    records = load_logs(log_file)
    if not records:
        print("  [ERROR] No records to analyse.")
        return []

    # Step 2+3: Group and compute per-window metrics
    print(f"\n  Grouping into {window_size} windows...")
    window_results = compute_temporal_metrics(records, window_size)
    print(f"  Found {len(window_results)} time window(s).")

    if len(window_results) == 0:
        return []

    # Step 4: Consecutive-window comparisons
    # Build a lookup: window_key → list of question embeddings
    def sort_key(r):
        ts = _parse_timestamp(r.get("timestamp"))
        return ts if ts is not None else datetime.min

    sorted_records = sorted(records, key=sort_key)
    buckets = group_by_time_window(sorted_records, window_size)

    def get_embeddings(bucket_key):
        """Extract valid question_embedding vectors from a bucket."""
        recs = buckets.get(bucket_key, [])
        return [
            r["question_embedding"]
            for r in recs
            if r.get("question_embedding") and len(r["question_embedding"]) > 0
        ]

    print("\n  Computing drift and change points between windows...")

    for i in range(1, len(window_results)):
        prev_win = window_results[i - 1]
        curr_win = window_results[i]

        # --- Delta metrics ---
        curr_win["delta_uncertainty"] = round(
            curr_win["mean_uncertainty"] - prev_win["mean_uncertainty"], 4
        )
        curr_win["delta_consistency"] = round(
            curr_win["mean_consistency"] - prev_win["mean_consistency"], 4
        )

        # --- Embedding drift ---
        emb_prev = get_embeddings(prev_win["window_key"])
        emb_curr = get_embeddings(curr_win["window_key"])

        if emb_prev and emb_curr:
            dreport = drift_report(emb_prev, emb_curr)
            curr_win["drift_score"]    = dreport["drift_score"]
            curr_win["drift_alert"]    = dreport["drift_alert"]
            curr_win["drift_severity"] = dreport["severity"]
        else:
            # Missing embeddings — record but don't crash
            curr_win["drift_score"]    = None
            curr_win["drift_alert"]    = False
            curr_win["drift_severity"] = "UNKNOWN"

        # --- Change point ---
        curr_win["change_point"] = detect_change_point(prev_win, curr_win)

    # First window has no predecessor — set deltas to 0
    window_results[0]["delta_uncertainty"] = 0.0
    window_results[0]["delta_consistency"] = 0.0
    window_results[0]["drift_score"]       = 0.0
    window_results[0]["drift_severity"]    = "LOW"

    return window_results


# =========================
# FORMATTED PRINTING
# =========================

def _alert_tag(flag):
    return "YES ⚠️" if flag else "no"


def print_temporal_results(results):
    """
    Print the temporal analysis results in a readable table.

    Args:
        results (list[dict]): Output of run_temporal_analysis().
    """
    if not results:
        print("  No results to display.")
        return

    print("\n" + "="*72)
    print(f"  {'Window':<12} {'Samples':>7}  {'Unc':>6}  {'Con':>6}  "
          f"{'ΔUnc':>7}  {'Drift':>7}  {'Alert':>8}  {'ChgPt':>7}")
    print("-"*72)

    for r in results:
        drift   = f"{r['drift_score']:.3f}" if r["drift_score"] is not None else "  N/A"
        delta_u = f"{r['delta_uncertainty']:+.3f}" if r["delta_uncertainty"] is not None else "   N/A"
        alert   = "YES ⚠" if r["drift_alert"] else "—"
        chgpt   = "YES ⚡" if r["change_point"] else "—"

        print(
            f"  {r['window_key']:<12} {r['num_samples']:>7}  "
            f"{r['mean_uncertainty']:>6.3f}  {r['mean_consistency']:>6.3f}  "
            f"{delta_u:>7}  {drift:>7}  {alert:>8}  {chgpt:>7}"
        )

    print("="*72)

    # Summary counts
    drift_alerts  = sum(1 for r in results if r["drift_alert"])
    change_points = sum(1 for r in results if r["change_point"])
    print(f"\n  Drift alerts : {drift_alerts}")
    print(f"  Change points: {change_points}")

    # Flag any windows that need attention
    critical = [
        r for r in results
        if r["drift_alert"] or r["change_point"]
    ]
    if critical:
        print("\n  ATTENTION — Windows requiring review:")
        for r in critical:
            reasons = []
            if r["drift_alert"]:
                reasons.append(f"drift={r['drift_score']:.3f}")
            if r["change_point"]:
                reasons.append(f"Δunc={r['delta_uncertainty']:+.3f}")
            print(f"    • {r['window_key']}  ({', '.join(reasons)})")
    else:
        print("\n  All windows within normal parameters.")

    print()


# =========================
# DEMO / TEST FUNCTION
# =========================

def demo_temporal_analysis(log_file="qa_monitoring_logs.jsonl"):
    """
    Load logs, run the full temporal analysis, and print results.

    Designed for quick smoke-testing of the Phase 2 pipeline.
    Can be called directly: python temporal_engine.py

    Args:
        log_file (str): Path to the JSONL log file.
    """
    print("\n" + "#"*60)
    print("  DEMO: Phase 2 Temporal Intelligence Module")
    print("#"*60)

    # Run analysis for 24-hour windows
    results_24h = run_temporal_analysis(log_file, window_size="24h")

    if results_24h:
        print(f"\n--- 24-HOUR WINDOW RESULTS ({len(results_24h)} windows) ---")
        print_temporal_results(results_24h)

        # Show one representative result dict in full
        print("--- SAMPLE RESULT DICT (most recent window) ---")
        sample = results_24h[-1].copy()
        # Omit the nested risk_distribution for brevity
        sample.pop("risk_distribution", None)
        for key, val in sample.items():
            print(f"  {key:<22}: {val}")
    else:
        print("\n  [INFO] No 24h window results — log file may be empty or missing.")

    # Run analysis for 7-day windows if enough data
    results_7d = run_temporal_analysis(log_file, window_size="7d")

    if len(results_7d) > 1:
        print(f"\n--- 7-DAY WINDOW RESULTS ({len(results_7d)} windows) ---")
        print_temporal_results(results_7d)
    else:
        print("\n  [INFO] Not enough data for multi-week analysis "
              f"({len(results_7d)} week window found).")

    print("\n  Demo complete.\n")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    demo_temporal_analysis()

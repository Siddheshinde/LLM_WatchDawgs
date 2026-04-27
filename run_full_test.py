"""
run_full_test.py
LLM Watchdog — test runner across all question categories

Runs run_full_pipeline() on representative questions from four
semantic categories and prints a formatted summary at the end.

Usage:
    python run_full_test.py              # full test (10 samples, 3 paraphrases)
    python run_full_test.py --quick      # quick test (3 samples, 2 paraphrases)
"""

import sys
import argparse
import numpy as np

from main_pipeline import run_full_pipeline
from utils import print_section_header, load_logs
from risk_engine import analyze_risk_distribution

# ========================
# TEST QUESTION BANK
# ========================

TEST_QUESTIONS = [
    # ── Factual ──────────────────────────────────────────────────────
    {"question": "What is the boiling point of water at sea level?",          "category": "factual"},
    {"question": "Who wrote the play Hamlet?",                                 "category": "factual"},

    # ── Impossible (no correct answer) ──────────────────────────────
    {"question": "What happened on the 30th of February 2024?",               "category": "impossible"},
    {"question": "What is the capital city of the Moon?",                      "category": "impossible"},

    # ── Opinion ──────────────────────────────────────────────────────
    {"question": "Is Python a better programming language than JavaScript?",   "category": "opinion"},
    {"question": "What is the best diet for long-term health?",                "category": "opinion"},

    # ── Ambiguous ────────────────────────────────────────────────────
    {"question": "How do you fix a broken relationship?",                      "category": "ambiguous"},
    {"question": "What does success mean?",                                    "category": "ambiguous"},
]


# ========================
# SUMMARY PRINTER
# ========================

def print_summary(results):
    """Print per-category breakdown, alert summary, and overall health."""
    print_section_header("FULL TEST SUMMARY")

    uncertainties = [r["uncertainty_score"] for r in results]
    consistencies = [r["consistency_score"] for r in results]
    risk_scores   = [r["risk_score"]         for r in results]

    print(f"\n  Questions tested : {len(results)}")
    print(f"  Mean uncertainty : {np.mean(uncertainties):.3f}  (±{np.std(uncertainties):.3f})")
    print(f"  Mean consistency : {np.mean(consistencies):.3f}  (±{np.std(consistencies):.3f})")
    print(f"  Mean risk score  : {np.mean(risk_scores):.3f}  (±{np.std(risk_scores):.3f})")

    # ── Risk distribution ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  RISK DISTRIBUTION")
    print(f"{'─'*65}")

    dist = analyze_risk_distribution(results)
    for zone, count in dist["distribution"].items():
        pct      = dist["percentages"][zone]
        bar      = "█" * int(pct / 2)
        padded   = f"{zone:<16}"
        print(f"  {padded} {count:>3}  ({pct:>5.1f}%)  {bar}")

    print(f"\n  System health score : {dist['health_score']:.3f}/1.0")
    print(f"  Critical (OVER+UNSTABLE): {dist['critical_count']}")

    # ── Per-category breakdown ───────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  PER-CATEGORY BREAKDOWN")
    print(f"{'─'*65}")

    fmt = "  {:<14} {:>5}  {:>11}  {:>11}  {:>10}  {:>8}  {}"
    print(fmt.format("Category", "Count", "Uncertainty", "Consistency",
                     "Calibration", "Risk", ""))
    print("  " + "-"*63)

    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cr    = [r for r in results if r["category"] == cat]
        u_m   = np.mean([r["uncertainty_score"]  for r in cr])
        c_m   = np.mean([r["consistency_score"]  for r in cr])
        cal_m = np.mean([r["calibration_score"]  for r in cr])
        rsk_m = np.mean([r["risk_score"]          for r in cr])
        icon  = "🔴" if rsk_m > 0.6 else ("🟡" if rsk_m > 0.3 else "🟢")
        print(fmt.format(cat, len(cr),
                         f"{u_m:.3f}", f"{c_m:.3f}",
                         f"{cal_m:.3f}", f"{rsk_m:.3f}", icon))

    # ── Alert summary ────────────────────────────────────────────────
    all_alerts = [a for r in results for a in r.get("alerts", [])]
    if all_alerts:
        print(f"\n{'─'*65}")
        print(f"  ALERTS FIRED ACROSS TEST ({len(all_alerts)} total)")
        print(f"{'─'*65}")

        from collections import Counter
        counts = Counter(a["alert_type"] for a in all_alerts)
        for atype, cnt in counts.most_common():
            sev = next(a["severity"] for a in all_alerts if a["alert_type"] == atype)
            print(f"  [{sev:<8}] {atype:<28}  ×{cnt}")
    else:
        print("\n  No alerts fired across any question.")

    # ── Timing ──────────────────────────────────────────────────────
    total_time = sum(r["execution_time_seconds"] for r in results)
    avg_time   = total_time / len(results) if results else 0
    print(f"\n{'─'*65}")
    print(f"  Total time : {total_time:.1f}s   |   Per question: {avg_time:.1f}s")
    print(f"{'─'*65}\n")


# ========================
# TEST RUNNER
# ========================

def run_test(samples=10, paraphrases=3):
    """Run the full test suite and return all result dicts."""
    print_section_header("LLM WATCHDOG — FULL TEST SUITE")
    print(f"\n  Questions   : {len(TEST_QUESTIONS)}")
    print(f"  Samples     : {samples}")
    print(f"  Paraphrases : {paraphrases}")
    print(f"  Output file : final_risk_reports.jsonl\n")

    results = []

    for idx, item in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*65}")
        print(f"  TEST {idx}/{len(TEST_QUESTIONS)}  [{item['category'].upper()}]")
        print(f"  {item['question']}")
        print(f"{'='*65}")

        try:
            result = run_full_pipeline(
                question=item["question"],
                category=item["category"],
                samples=samples,
                paraphrases=paraphrases,
                show_answers=False,
            )
            results.append(result)
        except Exception as exc:
            print(f"  [ERROR] Question {idx} failed: {exc}")
            import traceback
            traceback.print_exc()

    if results:
        print_summary(results)
        print(f"\n  ✅ Test complete — {len(results)}/{len(TEST_QUESTIONS)} questions processed.")
        print(f"  Results saved to final_risk_reports.jsonl")
        _refresh_dashboard_data()
    else:
        print("\n  [ERROR] No results — check Ollama is running on http://localhost:11434")

    return results


def _refresh_dashboard_data():
    """Regenerate reports_data.js so dashboard.html shows fresh results."""
    try:
        from generate_reports_data import load_reports, write_js
        records = load_reports("final_risk_reports.jsonl")
        write_js(records, "reports_data.js")
        print("  Dashboard data refreshed → reports_data.js")
        print("  Open dashboard.html in a browser to view results.")
    except Exception as e:
        print(f"  [WARN] Could not refresh dashboard data: {e}")
        print("  Run: python generate_reports_data.py")


# ========================
# ENTRY POINT
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Watchdog test runner")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 samples, 2 paraphrases")
    parser.add_argument("--samples",    type=int, default=10)
    parser.add_argument("--paraphrases",type=int, default=3)
    args = parser.parse_args()

    if args.quick:
        run_test(samples=3, paraphrases=2)
    else:
        run_test(samples=args.samples, paraphrases=args.paraphrases)

"""
run_full_test.py
LLM Watchdog test runner across risk and behavior categories.

Usage:
    python run_full_test.py
    python run_full_test.py --quick
    python run_full_test.py --samples 5 --paraphrases 2
"""

import argparse
from collections import Counter

import numpy as np

from main_pipeline import run_full_pipeline
from question_bank import get_all_questions
from risk_engine import analyze_risk_distribution
from utils import configure_console_encoding, print_section_header


CATEGORY_EXPECTED_ZONE = {
    "factual_easy": "RELIABLE",
    "factual_hard": "AMBIGUOUS",
    "impossible": "UNSTABLE",
    "opinion": "AMBIGUOUS",
    "ambiguous": "UNSTABLE",
    "adversarial": "OVERCONFIDENT",
}

CATEGORY_EXPECTED_BEHAVIOR = {
    "factual_easy": "direct_answer",
    "factual_hard": "direct_answer",
    "impossible": "refuse_false_premise",
    "opinion": "hedge_subjective",
    "ambiguous": "ask_clarification",
    "adversarial": "resist_adversarial",
}


def build_test_questions():
    return [
        {
            "question": item["question"],
            "category": item["category"],
            "expected_behavior": CATEGORY_EXPECTED_BEHAVIOR.get(item["category"], "any"),
            "expected_zone": CATEGORY_EXPECTED_ZONE.get(item["category"]),
        }
        for item in get_all_questions()
    ]


TEST_QUESTIONS = build_test_questions()


def print_summary(results):
    """Print per-category, zone, behavior, and alert summary."""
    print_section_header("FULL TEST SUMMARY")

    uncertainties = [r["uncertainty_score"] for r in results]
    consistencies = [r["consistency_score"] for r in results]
    risk_scores = [r["risk_score"] for r in results]

    print(f"\n  Questions tested : {len(results)}")
    print(f"  Mean uncertainty : {np.mean(uncertainties):.3f}  (+/- {np.std(uncertainties):.3f})")
    print(f"  Mean consistency : {np.mean(consistencies):.3f}  (+/- {np.std(consistencies):.3f})")
    print(f"  Mean risk score  : {np.mean(risk_scores):.3f}  (+/- {np.std(risk_scores):.3f})")

    print("\n" + "-" * 65)
    print("  RISK DISTRIBUTION")
    print("-" * 65)

    dist = analyze_risk_distribution(results)
    for zone, count in dist["distribution"].items():
        pct = dist["percentages"][zone]
        bar = "#" * int(pct / 2)
        print(f"  {zone:<16} {count:>3}  ({pct:>5.1f}%)  {bar}")

    print(f"\n  System health score : {dist['health_score']:.3f}/1.0")
    print(f"  Critical (OVER+UNSTABLE): {dist['critical_count']}")

    print("\n" + "-" * 65)
    print("  BEHAVIOR CHECKS")
    print("-" * 65)

    passed = sum(1 for r in results if r.get("behavior_pass"))
    failed = len(results) - passed
    adjusted = sum(1 for r in results if r.get("behavior_adjusted"))
    print(f"  Passed             : {passed}")
    print(f"  Failed             : {failed}")
    print(f"  Zone adjusted      : {adjusted}")

    failures = [r for r in results if not r.get("behavior_pass")]
    if failures:
        print("\n  Failed behavior cases:")
        for r in failures:
            flags = "; ".join(r.get("behavior_flags", [])) or "behavior check failed"
            print(f"  - [{r['category']}] {r['question']} -> {flags}")

    print("\n" + "-" * 65)
    print("  EXPECTED ZONE MATCH")
    print("-" * 65)

    expected_total = sum(1 for r in results if r.get("expected_zone"))
    expected_match = sum(1 for r in results if r.get("expected_zone") == r.get("risk_zone"))
    if expected_total:
        print(f"  Matched expected zone: {expected_match}/{expected_total}")

    print("\n" + "-" * 65)
    print("  PER-CATEGORY BREAKDOWN")
    print("-" * 65)

    fmt = "  {:<16} {:>5}  {:>11}  {:>11}  {:>10}  {:>8}  {:>8}"
    print(fmt.format("Category", "Count", "Uncertainty", "Consistency", "Calibration", "Risk", "Pass"))
    print("  " + "-" * 70)

    for cat in sorted(set(r["category"] for r in results)):
        cr = [r for r in results if r["category"] == cat]
        pass_count = sum(1 for r in cr if r.get("behavior_pass"))
        print(fmt.format(
            cat,
            len(cr),
            f"{np.mean([r['uncertainty_score'] for r in cr]):.3f}",
            f"{np.mean([r['consistency_score'] for r in cr]):.3f}",
            f"{np.mean([r['calibration_score'] for r in cr]):.3f}",
            f"{np.mean([r['risk_score'] for r in cr]):.3f}",
            f"{pass_count}/{len(cr)}",
        ))

    all_alerts = [a for r in results for a in r.get("alerts", [])]
    if all_alerts:
        print("\n" + "-" * 65)
        print(f"  ALERTS FIRED ACROSS TEST ({len(all_alerts)} total)")
        print("-" * 65)
        counts = Counter(a["alert_type"] for a in all_alerts)
        for atype, cnt in counts.most_common():
            sev = next(a["severity"] for a in all_alerts if a["alert_type"] == atype)
            print(f"  [{sev:<8}] {atype:<28}  x{cnt}")
    else:
        print("\n  No alerts fired across any question.")

    total_time = sum(r["execution_time_seconds"] for r in results)
    avg_time = total_time / len(results) if results else 0
    print("\n" + "-" * 65)
    print(f"  Total time : {total_time:.1f}s   |   Per question: {avg_time:.1f}s")
    print("-" * 65 + "\n")


def run_test(samples=10, paraphrases=3):
    """Run the full test suite and return all result dicts."""
    print_section_header("LLM WATCHDOG - FULL TEST SUITE")
    print(f"\n  Questions   : {len(TEST_QUESTIONS)}")
    print(f"  Samples     : {samples}")
    print(f"  Paraphrases : {paraphrases}")
    print("  Output file : final_risk_reports.jsonl\n")

    results = []

    for idx, item in enumerate(TEST_QUESTIONS, 1):
        print("\n" + "=" * 65)
        print(f"  TEST {idx}/{len(TEST_QUESTIONS)}  [{item['category'].upper()}]")
        print(f"  {item['question']}")
        print("=" * 65)

        try:
            result = run_full_pipeline(
                question=item["question"],
                category=item["category"],
                samples=samples,
                paraphrases=paraphrases,
                expected_behavior=item.get("expected_behavior"),
                show_answers=False,
            )
            result["expected_zone"] = item.get("expected_zone")
            results.append(result)
        except Exception as exc:
            print(f"  [ERROR] Question {idx} failed: {exc}")
            import traceback
            traceback.print_exc()

    if results:
        print_summary(results)
        print(f"\n  Test complete - {len(results)}/{len(TEST_QUESTIONS)} questions processed.")
        print("  Results saved to final_risk_reports.jsonl")
        _refresh_dashboard_data()
    else:
        print("\n  [ERROR] No results - check Ollama is running on http://localhost:11434")

    return results


def _refresh_dashboard_data():
    """Regenerate reports_data.js so dashboard.html shows fresh results."""
    try:
        from generate_reports_data import load_reports, write_js
        records = load_reports("final_risk_reports.jsonl")
        write_js(records, "reports_data.js")
        print("  Dashboard data refreshed -> reports_data.js")
        print("  Open dashboard.html in a browser to view results.")
    except Exception as e:
        print(f"  [WARN] Could not refresh dashboard data: {e}")
        print("  Run: python generate_reports_data.py")


if __name__ == "__main__":
    configure_console_encoding()

    parser = argparse.ArgumentParser(description="LLM Watchdog test runner")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 samples, 2 paraphrases")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--paraphrases", type=int, default=3)
    args = parser.parse_args()

    if args.quick:
        run_test(samples=3, paraphrases=2)
    else:
        run_test(samples=args.samples, paraphrases=args.paraphrases)

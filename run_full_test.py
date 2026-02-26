"""
run_full_test.py
Run comprehensive test suite across all question categories
"""

from llm_monitoring import monitor_questions_batch
from question_bank import QUESTION_BANK, print_question_bank_summary
from utils import load_logs, print_section_header
from risk_engine import analyze_risk_distribution
import numpy as np

def run_comprehensive_test(samples_per_question=5, paraphrases_per_question=2):
    """
    Run complete test suite
    
    Args:
        samples_per_question (int): Uncertainty samples
        paraphrases_per_question (int): Consistency paraphrases
    
    Returns:
        list: All test results
    """
    print_section_header("🧪 COMPREHENSIVE TEST SUITE")
    print(f"Configuration:")
    print(f"  Samples per question: {samples_per_question}")
    print(f"  Paraphrases per question: {paraphrases_per_question}")
    
    # Show question bank summary
    print_question_bank_summary()
    
    # Prepare all questions
    all_questions = []
    for category, data in QUESTION_BANK.items():
        for question in data["questions"]:
            all_questions.append({
                "question": question,
                "category": category
            })
    
    print(f"\n{'='*70}")
    print(f"Starting tests on {len(all_questions)} questions...")
    print(f"{'='*70}")
    
    # Run batch monitoring
    results = monitor_questions_batch(
        all_questions,
        samples=samples_per_question,
        num_paraphrases=paraphrases_per_question
    )
    
    # Generate summary
    print_test_summary(results)
    
    return results

def print_test_summary(results):
    """Print comprehensive test summary"""
    print_section_header("📊 TEST SUMMARY")
    
    # Overall statistics
    uncertainties = [r["uncertainty_score"] for r in results]
    consistencies = [r["consistency_score"] for r in results]
    calibrations = [r["calibration_score"] for r in results]
    risk_scores = [r["risk_score"] for r in results]
    
    print(f"\nTotal Questions Tested: {len(results)}")
    print(f"\nOverall Metrics:")
    print(f"  Mean Uncertainty:  {np.mean(uncertainties):.3f} (±{np.std(uncertainties):.3f})")
    print(f"  Mean Consistency:  {np.mean(consistencies):.3f} (±{np.std(consistencies):.3f})")
    print(f"  Mean Calibration:  {np.mean(calibrations):.3f} (±{np.std(calibrations):.3f})")
    print(f"  Mean Risk Score:   {np.mean(risk_scores):.3f} (±{np.std(risk_scores):.3f})")
    
    # Risk distribution
    print(f"\n{'='*70}")
    print("RISK DISTRIBUTION")
    print(f"{'='*70}")
    
    risk_dist = analyze_risk_distribution(results)
    for zone, count in risk_dist["distribution"].items():
        pct = risk_dist["percentages"][zone]
        bar_length = int(pct / 2)
        bar = "█" * bar_length
        print(f"{zone:<15} {count:>3} ({pct:>5.1f}%) {bar}")
    
    print(f"\nSystem Health Score: {risk_dist['health_score']:.3f}/1.0")
    print(f"Critical Issues: {risk_dist['critical_count']} (OVERCONFIDENT + UNSTABLE)")
    
    # Per-category breakdown
    print(f"\n{'='*70}")
    print("PER-CATEGORY PERFORMANCE")
    print(f"{'='*70}")
    
    categories = set(r["category"] for r in results)
    
    print(f"\n{'Category':<15} {'Count':>5} {'Uncertainty':>12} {'Consistency':>12} {'Calibration':>12} {'Risk':>8}")
    print("-"*70)
    
    for category in sorted(categories):
        cat_results = [r for r in results if r["category"] == category]
        
        u_mean = np.mean([r["uncertainty_score"] for r in cat_results])
        c_mean = np.mean([r["consistency_score"] for r in cat_results])
        cal_mean = np.mean([r["calibration_score"] for r in cat_results])
        risk_mean = np.mean([r["risk_score"] for r in cat_results])
        
        # Color coding
        if risk_mean < 0.3:
            risk_emoji = "🟢"
        elif risk_mean < 0.6:
            risk_emoji = "🟡"
        else:
            risk_emoji = "🔴"
        
        print(f"{category:<15} {len(cat_results):>5} {u_mean:>12.3f} {c_mean:>12.3f} {cal_mean:>12.3f} {risk_mean:>7.3f} {risk_emoji}")
    
    # Execution time
    total_time = sum(r["execution_time_seconds"] for r in results)
    avg_time = total_time / len(results)
    
    print(f"\n{'='*70}")
    print(f"Total Execution Time: {total_time:.1f}s")
    print(f"Average per Question: {avg_time:.1f}s")
    print(f"{'='*70}")

def run_quick_test():
    """Run quick test with fewer samples"""
    print("Running QUICK TEST (reduced samples for speed)...\n")
    return run_comprehensive_test(samples_per_question=3, paraphrases_per_question=2)

def run_full_test():
    """Run full test with all samples"""
    print("Running FULL TEST (complete sampling)...\n")
    return run_comprehensive_test(samples_per_question=10, paraphrases_per_question=3)

if __name__ == "__main__":
    import sys
    
    if "--quick" in sys.argv:
        results = run_quick_test()
    else:
        results = run_full_test()
    
    print("\n✅ Test complete! Results logged to qa_monitoring_logs.jsonl")
    print("Run 'python generate_dashboard.py' to create visual dashboard")
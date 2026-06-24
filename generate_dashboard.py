"""
generate_dashboard.py
Simple wrapper to generate dashboard from logs
"""

from dashboard_generator import generate_dashboard
import os
import sys

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  LLM WATCHDOG - Dashboard Generator")
    print("="*70 + "\n")

    # Prefer pipeline reports when available (white dashboard data source)
    log_file = (
        "final_risk_reports.jsonl"
        if os.path.exists("final_risk_reports.jsonl")
        else "qa_monitoring_logs.jsonl"
    )
    output_file = "dashboard.html"
    
    if "--log" in sys.argv:
        idx = sys.argv.index("--log")
        if idx + 1 < len(sys.argv):
            log_file = sys.argv[idx + 1]
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    success = generate_dashboard(log_file, output_file)
    
    if success:
        print(f"\nSuccess! Open {output_file} in your browser")
    else:
        print("\nFailed to generate dashboard")